"""Adapter for Synalinks LanguageModel to RLM interface."""

import asyncio
from typing import Any
from typing import Optional
from typing import Union

from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.language_models import LanguageModel


class SynalinksLMClient:
    """Adapts Synalinks LanguageModel to RLM-compatible interface.

    Provides both sync and async completion methods, converting
    between RLM's string/dict prompt format and Synalinks' ChatMessages.

    Supports multi-client architecture for cost optimization:
    - Create separate clients for root LM (expensive) and sub LM (cheap)
    - Each client tracks its own usage for cost analysis

    Args:
        language_model: Synalinks LanguageModel instance

    Example:
        >>> # Multi-model setup for cost optimization
        >>> lm_root = synalinks.LanguageModel(model="zai/glm-4.7")
        >>> lm_sub = synalinks.LanguageModel(model="groq/openai/gpt-oss-20b")
        >>>
        >>> client_root = SynalinksLMClient(lm_root)
        >>> client_sub = SynalinksLMClient(lm_sub)
        >>>
        >>> # Root for orchestration
        >>> response = client_root.completion("Plan analysis...")
        >>>
        >>> # Sub for recursive calls (cheaper)
        >>> chunk_result = client_sub.completion(f"Summarize: {chunk}")
    """

    def __init__(self, language_model: LanguageModel):
        self.language_model = language_model
        self.model_name = language_model.model
        self._usage_total: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }
        self._usage_last: dict[str, Any] = {}

    def _convert_prompt(self, prompt: Union[str, list[dict]]) -> ChatMessages:
        """Convert RLM prompt format to Synalinks ChatMessages."""
        if isinstance(prompt, str):
            return ChatMessages(
                messages=[ChatMessage(role=ChatRole.USER, content=prompt)]
            )
        elif isinstance(prompt, list):
            messages = []
            for msg in prompt:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                messages.append(ChatMessage(role=role, content=content))
            return ChatMessages(messages=messages)
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    def _extract_response(self, result: Any) -> str:
        """Extract string response from LanguageModel result."""
        if isinstance(result, dict):
            return result.get("content", str(result))
        if hasattr(result, "content"):
            return result.content
        return str(result)

    def _track_usage(self, usage: Optional[dict]):
        """Track usage statistics for cost analysis."""
        if usage:
            self._usage_last = usage.copy()
            self._usage_total["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self._usage_total["completion_tokens"] += usage.get("completion_tokens", 0)
            self._usage_total["total_tokens"] += usage.get("total_tokens", 0)
        self._usage_total["calls"] += 1

    def completion(self, prompt: Union[str, list[dict]]) -> str:
        """Synchronous completion - runs async LM in event loop.

        This is the primary method called by LMHandler when routing
        REPL llm_query() requests.
        """
        messages = self._convert_prompt(prompt)

        try:
            loop = asyncio.get_running_loop()

            future = asyncio.run_coroutine_threadsafe(self.language_model(messages), loop)
            result = future.result(timeout=300)
        except RuntimeError:
            result = asyncio.run(self.language_model(messages))

        # Track usage for cost analysis
        if hasattr(result, "usage"):
            self._track_usage(result.usage)

        return self._extract_response(result)

    async def acompletion(self, prompt: Union[str, list[dict]]) -> str:
        """Async completion - direct async call to LanguageModel."""
        messages = self._convert_prompt(prompt)
        result = await self.language_model(messages)

        if hasattr(result, "usage"):
            self._track_usage(result.usage)

        return self._extract_response(result)

    def get_usage_summary(self) -> dict:
        """Get cumulative usage/cost summary.

        Useful for comparing costs between root and sub clients.
        """
        return self._usage_total.copy()

    def get_last_usage(self) -> dict:
        """Get usage from last call."""
        return self._usage_last.copy()

    def reset_usage(self):
        """Reset usage counters (for new session/experiment)."""
        self._usage_total = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }
        self._usage_last = {}
