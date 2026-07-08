from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph,MessagesState, END
from pprint import pprint

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON="agent_reason"
ACT="act"
LAST=-1

def should_continue(state: MessagesState) -> str:
    if not state['messages'][LAST].tool_calls:
        return END
    return ACT


flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue,{
    END:END,
    ACT:ACT
})

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path='flow.png')

if __name__ == '__main__':
    print("Hello React LangGraph  Function Calling")
    res = app.invoke({
        'messages':[HumanMessage(content='What is the current temprature in Tokyo? list it and triple it.')],
    })
    # for learning purpose
    for msg in res["messages"]:
        print("=" * 80)
        print(type(msg).__name__)
        print(msg.pretty_repr())

    #final answer
    print(res["messages"][-1].text)

