"""Model-specific prompt templates for RLM.

This module provides prompt templates optimized for different LLM model families.
Templates are selected automatically based on the model prefix or can be set explicitly.
"""

from typing import Optional

# Registry of prompt templates by model family
PROMPT_TEMPLATES = {}


def _default_template():
    """Default RLM prompt template.

    Used when no model-specific template is available. This template provides
    a general-purpose structure that works across most model families.

    Returns:
        str: The default prompt template
    """
    return """
# Instructions
{{ instructions }}
{% if inputs_schema %}
# Input Schema
{{ inputs_schema }}
{% endif %}{% if outputs_schema %}
# Output Schema
{{ outputs_schema }}
{% endif %}{% if examples %}
# Examples
{% for example in examples %}
## Input:
{{ example[0] }}
## Output:
{{ example[1] }}
{% endfor %}
{% endif %}
# REPL Environment
You have access to a Python REPL with the following functions:
- llm_query(prompt: str) -> str: Make a recursive LLM call
- FINAL(value): Terminate with the final output value

Use the REPL to break down complex tasks into smaller steps.
""".strip()


def _zai_template():
    """Prompt template optimized for zai/glm-4.7 model.

    This template is tuned for the zai model family's specific behavior patterns:
    - Prefers clear, structured instructions
    - Works well with explicit step markers
    - Benefits from explicit formatting guidance

    Returns:
        str: The zai-optimized prompt template
    """
    return """
# Task Instructions
{{ instructions }}
{% if inputs_schema %}
# Input Format
{{ inputs_schema }}
{% endif %}{% if outputs_schema %}
# Expected Output Format
{{ outputs_schema }}
{% endif %}{% if examples %}
# Reference Examples
{% for example in examples %}
### Example Input:
{{ example[0] }}
### Example Output:
{{ example[1] }}
{% endfor %}
{% endif %}
# Available Tools
You can use these Python functions in your REPL:
- llm_query(prompt: str) -> str: Query the language model recursively
- FINAL(value): Return the final result

## Approach
1. Break down the task into logical steps
2. Use llm_query() for sub-problems that need reasoning
3. Combine results and call FINAL() when complete

Follow this structure for reliable results.
""".strip()


def _groq_template():
    """Prompt template optimized for groq/* models.

    This template is tuned for groq model family characteristics:
    - Benefits from concise, direct instructions
    - Performs well with minimal formatting
    - Prefers action-oriented guidance

    Returns:
        str: The groq-optimized prompt template
    """
    return """
# Instructions
{{ instructions }}
{% if inputs_schema %}
# Input Schema
{{ inputs_schema }}
{% endif %}{% if outputs_schema %}
# Output Schema
{{ outputs_schema }}
{% endif %}{% if examples %}
# Examples
{% for example in examples %}
Input: {{ example[0] }}
Output: {{ example[1] }}
{% endfor %}
{% endif %}
# REPL Functions
- llm_query(prompt: str) -> str: Make recursive LLM calls
- FINAL(value): Return final result

Break complex tasks into steps. Use llm_query() for sub-tasks. Call FINAL() when done.
""".strip()


# Register templates in the registry
PROMPT_TEMPLATES["default"] = _default_template()
PROMPT_TEMPLATES["zai"] = _zai_template()
PROMPT_TEMPLATES["groq"] = _groq_template()


def get_prompt_template(model: Optional[str] = None) -> str:
    """Get appropriate prompt template for a model.

    Selects the optimal template based on the model prefix. Falls back to
    default template for unknown model families. This enables automatic
    optimization across different LLM providers.

    Args:
        model: Model identifier (e.g., "zai/glm-4.7", "groq/openai/gpt-oss-20b").
               If None, returns default template.

    Returns:
        str: The selected prompt template

    Example:
        >>> template = get_prompt_template("zai/glm-4.7")
        >>> # Returns zai-optimized template
        >>> template = get_prompt_template("groq/llama-3.1-70b")
        >>> # Returns groq-optimized template
        >>> template = get_prompt_template("unknown/model")
        >>> # Returns default template
        >>> template = get_prompt_template(None)
        >>> # Returns default template
    """
    if model is None:
        return PROMPT_TEMPLATES["default"]

    # Extract model family prefix (e.g., "zai" from "zai/glm-4.7")
    prefix = model.split("/")[0] if "/" in model else model

    # Return family-specific template or default
    return PROMPT_TEMPLATES.get(prefix, PROMPT_TEMPLATES["default"])
