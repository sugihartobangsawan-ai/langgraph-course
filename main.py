from typing import TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing import TypedDict

from chains import generate_chain, reflection_chain


#new state
class State(TypedDict):
    original_request: str
    draft: str
    critique: str
    iteration: int

REFLECT='reflect'
GENERATE='generate'

def generation_node(state: State):
    result = generate_chain.invoke({
        "original_request": state["original_request"],
        "draft": state["draft"],
        "critique": state["critique"],
    })

    return {
        "draft": result.text
    }

def reflection_node(state: State):
    result = reflection_chain.invoke({
        "draft": state["draft"]
    })

    return {
        "critique": result.text,
        'iteration': state["iteration"] + 1
    }

builder = StateGraph(state_schema=State)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state: State):
    if state['iteration'] >=3:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue, path_map={
    END:END,
    REFLECT:REFLECT,
})
builder.add_edge(REFLECT, GENERATE)

app = builder.compile()
# app.get_graph().draw_mermaid_png(output_file_path='flow.png')
graph = builder.compile()

first_prompt = """
Make this tweet better:
@LangChainAI - newly Tool Calling feature is seriously underrated.
After a long wait, it's here - making the implementation of agents across different models with function calling - super easy.
Made a video covering their newest blog post
"""

if __name__ == "__main__":
    print("Hello LangGraph")
    inputs = {
        "original_request": first_prompt,
        "draft": "",
        "critique": "",
        'iteration': 0
    }
    response=graph.invoke(inputs)

    print("=" * 60)
    print("FINAL RESULT")
    print("=" * 60)

    print("\nOriginal Request:")
    print(response["original_request"])

    print("\nFinal Draft:")
    print(response["draft"])

    print("\nLatest Critique:")
    print(response["critique"])

    print("\nReflection Iterations:")
    print(response["iteration"])

