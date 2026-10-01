from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any
import requests


def _server_timeout_from_env() -> int:
    raw = os.getenv("LETTA_SERVER_TIMEOUT", "120").strip()
    try:
        timeout = int(raw)
    except ValueError as exc:
        raise ValueError("LETTA_SERVER_TIMEOUT must be a positive integer.") from exc
    if timeout < 1:
        raise ValueError("LETTA_SERVER_TIMEOUT must be a positive integer.")
    return timeout


@dataclass
class HttpClient:
    """
    Thin wrapper around requests to keep the rest of the code testable and clean.
    """
    timeout_default: int = field(default_factory=_server_timeout_from_env)

    def request(
        self,
        method: str,
        url: str,
        payload: Any | None = None,
        timeout: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        t = timeout if timeout is not None else self.timeout_default
        r = requests.request(method, url, json=payload, timeout=t, headers=headers)
        if not r.ok:
            # Raise but keep response text visible for debugging.
            raise requests.HTTPError(f"HTTP {r.status_code}: {r.text}", response=r)

        if not r.text:
            return None
        return r.json()

    def get(self, url: str, timeout: int | None = None, headers: dict[str, str] | None = None) -> Any:
        return self.request("GET", url, payload=None, timeout=timeout, headers=headers)

    def post(
        self,
        url: str,
        payload: Any | None = None,
        timeout: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return self.request("POST", url, payload=payload, timeout=timeout, headers=headers)

    def post_sse_json(
        self,
        url: str,
        payload: Any | None = None,
        timeout: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> list[Any]:
        """POST and decode JSON values from a server-sent event stream."""
        t = timeout if timeout is not None else self.timeout_default
        with requests.post(
            url,
            json=payload,
            timeout=t,
            headers=headers,
            stream=True,
        ) as response:
            if not response.ok:
                raise requests.HTTPError(
                    f"HTTP {response.status_code}: {response.text}",
                    response=response,
                )
            chunks: list[Any] = []
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = str(raw_line).strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if not data or data == "[DONE]":
                    continue
                chunks.append(json.loads(data))
            return chunks

    def patch(
        self,
        url: str,
        payload: Any | None = None,
        timeout: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return self.request("PATCH", url, payload=payload or {}, timeout=timeout, headers=headers)
