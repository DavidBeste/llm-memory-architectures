from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any

from colorama import Fore, Style, init

init(autoreset=True)


@dataclass(frozen=True)
class Colors:
    USER: str = Fore.CYAN + Style.BRIGHT
    ASSISTANT: str = Fore.GREEN + Style.BRIGHT
    SYSTEM: str = Fore.MAGENTA + Style.BRIGHT
    TOOL: str = Fore.YELLOW + Style.BRIGHT
    META: str = Fore.WHITE + Style.DIM
    ERROR: str = Fore.RED + Style.BRIGHT
    RESET: str = Style.RESET_ALL


C = Colors()


def fmt_ts(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S")
    except Exception:
        return ""


def print_block(label: str, text: str, color: str, timestamp: str | None = None) -> None:
    ts = fmt_ts(timestamp)
    prefix = f"[{ts}] " if ts else ""
    print(f"{color}{prefix}{label}:{C.RESET}")
    print(f"{color}{text}{C.RESET}")
    print()


def print_meta(text: str) -> None:
    print(f"{C.META}{text}{C.RESET}")


def strip_hidden_thinking(text: str) -> str:
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def extract_assistant_reply(resp_json: dict[str, Any]) -> str | None:
    # Matches your observed schema: assistant_message with string content
    for m in resp_json.get("messages", []):
        if m.get("message_type") == "assistant_message":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return strip_hidden_thinking(c)
    return None


def iter_items(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        for key in ("data", "passages", "results", "conversations", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return []
    if isinstance(payload, list):
        return payload
    return []
