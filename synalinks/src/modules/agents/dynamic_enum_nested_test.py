import openai
import os
import json
import jsonschema

from dotenv import load_dotenv

from synalinks.src.modules.agents.prebuilt import dynamic_enum_on_nested_property, ToolDecision
from synalinks.src import testing

load_dotenv()

class DynamicEnumNestedTest(testing.TestCase):
    def setUp(self):
        self.sample_tools = ["calulcate", "word_count"]
        self.base_schema = ToolDecision.get_schema()

    def test_openai_api_call(self):
        """Test that the generated schema works with OpenAI's API"""
        decision_schema = dynamic_enum_on_nested_property(
            self.base_schema,
            "ToolChoice/properties/tool",
            self.sample_tools,
            description="The name of the tool to run from the available toolkit."
        )

        api_key = os.getenv("OPENAI_API_KEY", "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")

        client = openai.OpenAI(api_key=api_key)

        # tool_choices_ref = decision_schema["properties"]["tool_choices"]["items"]["$ref"]
        # self.assertEqual(tool_choices_ref, "#/$defs/ToolChoice")

        completion = client.beta.chat.completions.parse(
            model="gpt-4.1-nano",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that analyzes tasks and selects appropriate tools reasoning step by step."
                },
                {
                    "role": "user",
                    "content": "I have a document about Python frameworks and I need to count how many words it contains, then calculate the average words per section. What tools should I use?"
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "tool_decision",
                    "schema": decision_schema,
                    "strict": True
                }
            }
        )

        response = completion.choices[0].message.content
        response_content = json.loads(response)
        print(response_content)
        
        # Validate that response_content is compliant with decision_schema
        try:
            jsonschema.validate(instance=response_content, schema=decision_schema)
            print("✓ Response is compliant with decision_schema")
        except jsonschema.ValidationError as e:
            self.fail(f"Response is not compliant with decision_schema: {e}")
        except jsonschema.SchemaError as e:
            self.fail(f"Invalid decision_schema: {e}")
