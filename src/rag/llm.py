from langchain_mistralai import ChatMistralAI


def get_model(api_key: str, model: str):
    llm = ChatMistralAI(
        model=model,
        max_retries=2,
        api_key=api_key,
    )
    llm.verbose = False
    
    return llm
