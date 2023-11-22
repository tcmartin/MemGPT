import os
import logging

import json

from ..utils import count_tokens, load_grammar_file  # Assuming these utilities are available

import google.generativeai as palm






#DEBUG = False
DEBUG = True

def get_palm_completion(endpoint, prompt, context_window, model_name="chat-bison-32k",model=None, settings=None, grammar=None):
    """
    Function to get a completion from Vertex AI's PaLM model (chat-bison), with support for grammar and settings.
    """
    # Initialize VertexAI with appropriate credentials
    from memgpt.config import MemGPTConfig

    # load config
    config = MemGPTConfig.load()

    palm.configure(api_key=config.google_key)
    prompt_tokens = count_tokens(prompt)
    if DEBUG:
        print("PaLM API prompt_tokens: {prompt_tokens}")
        print('PaLM API prompt: '+prompt)
    if prompt_tokens > context_window:
        raise Exception(f"Request exceeds maximum context length ({prompt_tokens} > {context_window} tokens)")

    # Apply grammar to the prompt if specified
    if grammar is not None:
        grammar_rules = load_grammar_file(grammar)
        prompt = apply_grammar_to_prompt(prompt, grammar_rules)

    # Prepare settings for the chat model
    if settings is None:
        settings = {
            "max_output_tokens": 400,
            "temperature": 0.9,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }

    try:
        
        response = palm.chat(context="",messages=[prompt])
        result = response.last
        if DEBUG:
            print(f"PaLM API response: {response}")
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
