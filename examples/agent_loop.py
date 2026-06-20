"""Give any chat loop long-term memory in ~10 lines.

The pattern: before answering, *recall* relevant memories and prepend them to the prompt;
after answering, *remember* anything worth keeping. agentrecall is the persistent store —
bring whichever LLM you like (this demo fakes the model so it runs offline).

Run with:  python examples/agent_loop.py
"""

from agentrecall import Memory


def build_prompt(memory: Memory, user_message: str) -> str:
    recalled = memory.search(user_message, k=3)
    context = "\n".join(f"- {hit.content}" for hit in recalled)
    return (
        f"Relevant things you remember about the user:\n{context or '- (nothing yet)'}\n\n"
        f"User: {user_message}\nAssistant:"
    )


def fake_llm(prompt: str) -> str:  # swap for Claude / OpenAI / a local model
    return "(this is where your LLM reply would go)"


def main() -> None:
    with Memory("agent_loop.db", namespace="user:demo") as memory:
        # Seed a couple of durable facts.
        memory.add("The user is vegetarian", tags=["diet"])
        memory.add("The user is learning Spanish", tags=["goal"])

        turns = [
            "Can you suggest a recipe for dinner?",
            "How's my language learning going?",
        ]
        for message in turns:
            prompt = build_prompt(memory, message)
            print(prompt)
            reply = fake_llm(prompt)
            print(reply, "\n" + "-" * 60)
            # Persist the exchange so future turns can recall it.
            memory.add(f"User asked: {message}", tags=["history"])


if __name__ == "__main__":
    main()
