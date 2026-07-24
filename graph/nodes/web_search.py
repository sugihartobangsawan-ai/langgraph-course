from typing import Any, Dict

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("-- WEB SEARCH --")
    question = state['question']
    documents = state['documents']

    tavily_result = web_search_tool.invoke(
        {'query': question},
    )

    joined_tavily_result = "\n".join(
        item["content"] for item in tavily_result["results"]
    )
    # print(joined_tavily_result)

    web_result = Document(page_content = joined_tavily_result)
    if documents is not None:
        documents.append(web_result)
    else:
        documents = [web_result]
    return {'documents': documents, 'question': question}


if __name__ == "__main__":
    web_search(
        state={
            'question': 'agent memory',
            'documents':None
        }
    )

    result = web_search_tool.invoke({"query": "agent memory"})
    print(result)
