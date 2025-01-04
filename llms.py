import os
from dotenv import load_dotenv
import dspy


load_dotenv()

provider_to_api_key = {
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
}

model_to_provider = {
    "meta-llama/llama-3.1-8b-instruct": "openrouter",
    "openrouter/meta-llama/llama-3.1-8b-instruct": "openrouter",
    "openrouter/anthropic/claude-3.5-sonnet-20240620": "openrouter",
    "deepseek/deepseek-chat": "deepseek",
    "deepseek-chat": "deepseek",
}

provider_to_base_url = {
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com",
}

def get_lm(model: str):
    lm = dspy.LM(
        model=model,
        api_key=provider_to_api_key[model_to_provider[model]],
        api_base=provider_to_base_url[model_to_provider[model]],
        max_tokens=3000
        #cache=False
    )
    return lm

