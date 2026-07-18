from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END,START, StateGraph, MessagesState

from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 2

def draft_node(state: MessagesState):
    """Draft the initial response."""
    response = first_responder.invoke({'messages':state['messages']})
    return {'messages': [response]}

def revise_node(state:MessagesState):
    """Revise the answer based on tool results"""
    response = revisor.invoke({'messages': state['messages']})
    return {'messages': [response]}

def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
    """Determine whether to continue or end based on iteration count"""
    tool_calls = sum(
        isinstance(msg, ToolMessage)
        for msg in state["messages"]
    )
    if tool_calls >= MAX_ITERATIONS:
        return END
    return "execute_tools"


builder = StateGraph(MessagesState)
builder.add_node('draft', draft_node)
builder.add_node('execute_tools', execute_tools)
builder.add_node('revise', revise_node)


builder.add_edge(START, 'draft')
builder.add_edge('draft', 'execute_tools')
builder.add_edge('execute_tools','revise')
builder.add_conditional_edges('revise', event_loop, ['execute_tools',END])

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

res= graph.invoke(
    {
        'messages': [
            {
                'role': 'user',
                'content': 'Write about shrimp startup in Indonesia and list top 3 startup that got invested'
            }
        ]
    }
)

last_message = res['messages'][-1]