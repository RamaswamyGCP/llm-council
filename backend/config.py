"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "arcee-ai/trinity-mini:free",
    "amazon/nova-2-lite-v1:free",
    "allenai/olmo-3-32b-think:free",
    "tngtech/tng-r1t-chimera:free",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "arcee-ai/trinity-mini:free"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
