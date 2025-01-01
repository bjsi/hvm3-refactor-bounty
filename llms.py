import os
from dotenv import load_dotenv


load_dotenv()

provider_to_api_key = {
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
}

model_to_provider = {
    "meta-llama/llama-3.1-8b-instruct": "openrouter",
    "openrouter/meta-llama/llama-3.1-8b-instruct": "openrouter",
    "deepseek/deepseek-chat": "deepseek",
    "deepseek-chat": "deepseek",
}

provider_to_base_url = {
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com",
}