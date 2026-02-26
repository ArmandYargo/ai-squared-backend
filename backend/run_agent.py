# run_agent.py
import uuid
from agent.graph import get_graph


def main():
    print("RAM LangGraph Agent")
    print("Type 'exit' to quit.")
    print("Tips:")
    print("  - Start the wizard:  create input sheet")
    print("  - Ask a question:    ? What is MTBF?")
    print("  - RAM commands:      /ram status | /ram reset | /ram cancel\n")

    graph = get_graph()
    thread_id = str(uuid.uuid4())

    state = {
        "messages": [],
        "ram_wizard": {"active": False, "step": "machine"},
        "intent": "qa",
    }

    while True:
        user = input("You: ").strip()
        if user.lower() in {"exit", "quit"}:
            break

        state["messages"] = state.get("messages", []) + [{"role": "user", "content": user}]
        out = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
        state = out

        msgs = out.get("messages") or []
        if msgs and msgs[-1].get("role") == "assistant":
            last = msgs[-1]
            speaker = last.get("speaker", "ASSISTANT")
            print(f"\nAssistant ({speaker}): {last.get('content','')}\n")


if __name__ == "__main__":
    main()
