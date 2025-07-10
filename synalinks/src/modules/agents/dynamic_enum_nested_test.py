import openai
import os
import json

from dotenv import load_dotenv

from synalinks.src.modules.agents.parallel_react_agent_redo import dynamic_enum_nested, MultiDecisionAnswer
from synalinks.src import testing

load_dotenv()

class DynamicEnumNestedTest(testing.TestCase):
    def setUp(self):
        self.sample_tools = ["calulcate", "word_count"]
        self.base_schema = MultiDecisionAnswer.get_schema()

    def test_openai_api_call(self):
        """Test that the generated schema works with OpenAI's API"""
        decision_schema = dynamic_enum_nested(
            self.base_schema,
            "tool_choices/items/properties/tool",
            self.sample_tools,
            description="Available tools to choose from"
        )

        api_key = os.getenv("OPENAI_API_KEY")

        client = openai.OpenAI(api_key=api_key)

        tool_choices_ref = decision_schema["properties"]["tool_choices"]["items"]["$ref"]
        self.assertEqual(tool_choices_ref, "#/$defs/ToolChoice")

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that analyzes tasks and selects appropriate tools."
                },
                {
                    "role": "user",
                    "content": "I have a document about Python frameworks and I need to count how many words it contains, then calculate the average words per section. What tools should I use?"
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "tool_selection",
                    "schema": decision_schema
                }
            }
        )

        response_content = completion.choices[0].message.content
        self.assertIsNotNone(response_content)
