import os
from dotenv import load_dotenv
import dspy


load_dotenv()

provider_to_api_key = {
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY"),
}

model_to_provider = {
    "gemini/gemini-1.5-flash-8b": "gemini",
    "gemini/gemini-1.5-flash": "gemini",
    "gemini/gemini-1.5-pro": "gemini",
    "meta-llama/llama-3.1-8b-instruct": "openrouter",
    "openrouter/meta-llama/llama-3.1-8b-instruct": "openrouter",
    "openrouter/anthropic/claude-3.5-sonnet-20240620": "openrouter",
    "openrouter/openai/gpt-4o": "openai",
    "deepseek/deepseek-chat": "deepseek",
    "deepseek-chat": "deepseek",
    "openrouter/openai/gpt-4o-mini": "openai",
}

provider_to_base_url = {
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com",
    "gemini": None,
    "openai": None,
}

def get_lm(model: str):
    lm = dspy.LM(
        model=model,
        api_key=provider_to_api_key[model_to_provider[model]],
        api_base=provider_to_base_url[model_to_provider[model]],
        max_tokens=3000
    )
    return lm

gemini_8b = get_lm("gemini/gemini-1.5-flash-8b")
gemini_flash = get_lm("gemini/gemini-1.5-flash")
gemini_pro = get_lm("gemini/gemini-1.5-pro")
deepseek_chat = get_lm("deepseek/deepseek-chat")
claude_sonnet = get_lm("openrouter/anthropic/claude-3.5-sonnet-20240620")
gpt_4o = get_lm("openrouter/openai/gpt-4o")