import json

from .wrapper_base import LLMChatCompletionWrapper
from ..json_parser import clean_json
from ...errors import LLMJSONParsingError


class Dolphin21MistralWrapper(LLMChatCompletionWrapper):
    """Wrapper for Dolphin 2.1 Mistral 7b: https://huggingface.co/ehartford/dolphin-2.1-mistral-7b

    Note: this wrapper formats a prompt that only generates JSON, no inner thoughts
    """

    def __init__(
        self,
        simplify_json_content=True,
        clean_function_args=True,
        include_assistant_prefix=True,
        include_opening_brace_in_prefix=True,
        include_section_separators=False,
    ):
        self.simplify_json_content = simplify_json_content
        self.clean_func_args = clean_function_args
        self.include_assistant_prefix = include_assistant_prefix
        self.include_opening_brance_in_prefix = include_opening_brace_in_prefix
        self.include_section_separators = include_section_separators

    def chat_completion_to_prompt(self, messages, functions):
        """Example for airoboros: https://huggingface.co/jondurbin/airoboros-l2-70b-2.1#prompt-format

        <|im_start|>system
        You are Dolphin, a helpful AI assistant.<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant

        Do function spec Airoboros style inside the system message:
        Functions support: https://huggingface.co/jondurbin/airoboros-l2-70b-2.1#agentfunction-calling

            As an AI assistant, please select the most suitable function and parameters from the list of available functions below, based on the user's input. Provide your response in JSON format.

            Input: I want to know how many times 'Python' is mentioned in my text file.

            Available functions:
            file_analytics:
              description: This tool performs various operations on a text file.
              params:
                action: The operation we want to perform on the data, such as "count_occurrences", "find_line", etc.
                filters:
                  keyword: The word or phrase we want to search for.

        OpenAI functions schema style:

            {
                "name": "send_message",
                "description": "Sends a message to the human user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # https://json-schema.org/understanding-json-schema/reference/array.html
                        "message": {
                            "type": "string",
                            "description": "Message contents. All unicode (including emojis) are supported.",
                        },
                    },
                    "required": ["message"],
                }
            },
        """
        prompt = ""

        # <|im_start|>system
        # You are Dolphin, a helpful AI assistant.<|im_end|>

        IM_START_TOKEN = "<|im_start|>"
        IM_END_TOKEN = "<|im_end|>"

        # System instructions go first
        assert messages[0]["role"] == "system"
        prompt += f"{IM_START_TOKEN}system"
        prompt += f"\n{messages[0]['content']}"

        # Next is the functions preamble
        def create_function_description(schema):
            # airorobos style
            func_str = ""
            func_str += f"{schema['name']}:"
            func_str += f"\n  description: {schema['description']}"
            func_str += f"\n  params:"
            for param_k, param_v in schema["parameters"]["properties"].items():
                # TODO we're ignoring type
                func_str += f"\n    {param_k}: {param_v['description']}"
            # TODO we're ignoring schema['parameters']['required']
            return func_str

        # prompt += f"\nPlease select the most suitable function and parameters from the list of available functions below, based on the user's input. Provide your response in JSON format."
        prompt += f"\nPlease select the most suitable function and parameters from the list of available functions below, based on the ongoing conversation. Provide your response in JSON format."
        prompt += f"\nAvailable functions:"
        for function_dict in functions:
            prompt += f"\n{create_function_description(function_dict)}"

        # Put functions INSIDE system message (TODO experiment with this)
        prompt += IM_END_TOKEN
    def clean_function_args(self, function_name, function_args):
        """Some basic MemGPT-specific cleaning of function args"""
        cleaned_function_name = function_name
        cleaned_function_args = function_args.copy()

        if function_name == "send_message":
            # strip request_heartbeat
            cleaned_function_args.pop("request_heartbeat", None)

        # TODO more cleaning to fix errors LLM makes
        return cleaned_function_name, cleaned_function_args

        def create_function_call(self, function_call):
        """
        Converts a function call structure to a format suitable for inclusion in the prompt.
        """
        airo_func_call = {
            "function": function_call["name"],
            "params": json.loads(function_call["arguments"]),
        }
        return json.dumps(airo_func_call, indent=2)

    def output_to_chat_completion_response(self, raw_llm_output):
        """
        Converts the raw output from the Vertex AI model into a ChatCompletion-style response.
        """
        if raw_llm_output and raw_llm_output[0] != "{":
            raw_llm_output = "{" + raw_llm_output  # Adjust based on the actual output format

        try:
            function_json_output = clean_json(raw_llm_output)
        except Exception as e:
            raise LLMJSONParsingError(f"Failed to decode JSON from LLM output:\n{raw_llm_output} - error\n{str(e)}")

        try:
            function_name = function_json_output["function"]
            function_parameters = function_json_output["params"]

            if self.clean_func_args:
                function_name, function_parameters = self.clean_function_args(function_name, function_parameters)

            message = {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": function_name,
                    "arguments": json.dumps(function_parameters),
                },
            }
            return message
        except KeyError as e:
            raise LLMJSONParsingError(f"Received valid JSON from LLM, but JSON was missing fields: {str(e)}")

    