from langchain_openai import ChatOpenAI


def get_model(api_key: str, model: str):
    llm = ChatOpenAI(
        model=model,
        max_retries=2,
        openai_api_key=api_key,
        openai_api_base="https://api.proxyapi.ru/openrouter/v1",
    )
    llm.verbose = False
    
    return llm
