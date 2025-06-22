import os
from functools import lru_cache
from urllib.parse import urlparse

import httpx
import structlog
from langchain_community.chat_models.ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI

logger = structlog.get_logger(__name__)


@lru_cache(maxsize=4)
def get_openai_llm(model: str = "gpt-3.5-turbo", azure: bool = False):
    proxy_url = os.getenv("PROXY_URL")
    http_client = None
    if proxy_url:
        parsed_url = urlparse(proxy_url)
        if parsed_url.scheme and parsed_url.netloc:
            http_client = httpx.AsyncClient(proxies=proxy_url)
        else:
            logger.warn("Invalid proxy URL provided. Proceeding without proxy.")

    if not azure:
        try:
            openai_model = model
            llm = ChatOpenAI(
                # http_client=http_client,
                openai_api_key=os.environ["OPENAI_API_KEY"],
                openai_api_base=os.environ["OPENAI_API_BASE"],
                model=openai_model,
                temperature=0,
            )
        except Exception as e:
            logger.error(
                f"Failed to instantiate ChatOpenAI due to: {str(e)}. Falling back to AzureChatOpenAI."
            )
            llm = AzureChatOpenAI(
                http_client=http_client,
                temperature=0,
                deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
                azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
            )
    else:
        llm = AzureChatOpenAI(
            http_client=http_client,
            temperature=0,
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            azure_endpoint=os.environ["AZURE_OPENAI_API_BASE"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
        )
    return llm


@lru_cache(maxsize=1)
def get_ollama_llm():
    model_name = os.environ.get("OLLAMA_MODEL")
    if not model_name:
        model_name = "llama2"
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL")
    if not ollama_base_url:
        ollama_base_url = "http://localhost:11434"

    return ChatOllama(model=model_name, base_url=ollama_base_url)
