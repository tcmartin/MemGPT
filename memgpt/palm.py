import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair, ChatMessage

import os


def chat_completion(sys_prompt, chat_prompt, max_tokens, api_key=os.environ["OPENAI_API_KEY"], model="chat-bison", temperature=0.9):
    
    from memgpt.config import MemGPTConfig

    # load config
    config = MemGPTConfig.load()
    vertexai.init(os.environ.get('GOOGLE'), os.environ.get('GOOGLE_LOCATION'), credentials=os.environ.get('GOOGLE_CREDENTIALS'))
    try:
        if chat_prompt == "":
            chat_prompt = "Please do what the context says"
        parameters={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.8,
            "top_k": 40
        }
        chat_model = ChatModel.from_pretrained("chat-bison")
        chat = chat_model.start_chat(
            context=sys_prompt,
        )
        response = chat.send_message(chat_prompt, **parameters)
        return response.text
    except Exception as e:
        logger.info(e)
        raise e