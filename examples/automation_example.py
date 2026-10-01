"""
Example: non-interactive research script that uses the library API.

Goal: run a fixed prompt sequence, insert memory, and search memory.
"""

from letta_research_chat.config import LettaConfig
from letta_research_chat.http import HttpClient
from letta_research_chat.agent import AgentClient
from letta_research_chat.conversations import ConversationClient
from letta_research_chat.memory import MemoryClient
from letta_research_chat.render import extract_assistant_reply


def run():
    cfg = LettaConfig()
    http = HttpClient()
    agent = AgentClient(cfg.base_url, http)
    convos = ConversationClient(cfg.base_url, http)
    mem = MemoryClient(cfg.base_url, http)

    agent_id, _ = agent.get_or_create_agent_id(cfg.agent_name, model=cfg.agent_model)

    # Start (or reuse) a conversation explicitly for batch experiments
    conv = convos.create_conversation(agent_id)
    cid = conv.get("id") or conv.get("conversation_id")
    if not cid and isinstance(conv.get("data"), dict):
        cid = conv["data"].get("id") or conv["data"].get("conversation_id")
    if not cid:
        raise RuntimeError("No conversation id returned")

    # Insert a memory directly (no LLM call)
    mem.insert_archival_memory(agent_id, "User likes pizza and ice cream.")

    prompts = [
        "Hi!",
        "Can you summarize what you know about my preferences?",
        "Search archival memory and answer based on it.",
    ]

    for p in prompts:
        resp = convos.send_conversation_message(cid, p)
        ans = extract_assistant_reply(resp) or "<no assistant_message>"
        print(f"\nUSER: {p}\nASSISTANT: {ans}")

    # Search archival memory
    res = mem.search_archival_memory(agent_id, "preferences", limit=5)
    print("\n" + mem.format_archival(res))


if __name__ == "__main__":
    run()
