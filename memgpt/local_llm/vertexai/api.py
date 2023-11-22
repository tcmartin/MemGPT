import os
import logging
from vertexai.language_models import ChatModel
import json

from ..utils import count_tokens, load_grammar_file  # Assuming these utilities are available
import vertexai
from google.oauth2.service_account import Credentials as ServiceAccountCredentials


DEBUG = False
# DEBUG = True

def get_palm_completion(endpoint, prompt, context_window, model_name="chat-bison",model=None, settings=None, grammar=None):
    """
    Function to get a completion from Vertex AI's PaLM model (chat-bison), with support for grammar and settings.
    """
    # Initialize VertexAI with appropriate credentials
    from memgpt.config import MemGPTConfig

    # load config
    config = MemGPTConfig.load()

    #cred_dict = json.loads(config.google_credentials)
    creds = ServiceAccountCredentials.from_service_account_info(config.google_credentials, scopes=["https://www.googleapis.com/auth/cloud-platform"])
    vertexai.init(config.google_project, config.google_location, credentials=creds)

    prompt_tokens = count_tokens(prompt)
    if prompt_tokens > context_window:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens} > {context_window} tokens)")

    # Apply grammar to the prompt if specified
    if grammar is not None:
        grammar_rules = load_grammar_file(grammar)
        prompt = apply_grammar_to_prompt(prompt, grammar_rules)

    # Prepare settings for the chat model
    if settings is None:
        settings = {
            "max_output_tokens": 100,
            "temperature": 0.9,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }

    try:
        chat_model = ChatModel.from_pretrained(model_name)
        chat = chat_model.start_chat(context=prompt)

        response = chat.send_message(prompt, **settings)
        result = response.text

        if DEBUG:
            print(f"PaLM API response.text: {result}")

        return result
    except Exception as e:
        logging.error(f"Error in get_palm_completion: {e}")
        raise

    # Helper functions:
def apply_grammar_to_prompt(prompt, grammar_rules):
    # Implement logic to apply grammar rules to the prompt
    # This is a placeholder function and should be tailored to your specific grammar application needs
    return prompt + "\n" + grammar_rules
