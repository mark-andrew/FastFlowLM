#!/usr/bin/env python3
"""Anthropic Messages API: text, required tools, and streaming."""

from __future__ import annotations

import flm_api
from flm_api import Api, case, expect


def messages_url(api: Api) -> str:
    return api.url.replace("/v1/chat/completions", "/v1/messages")


def tool() -> dict:
    return {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }


def anthropic_body(api: Api, messages: list[dict], **fields) -> dict:
    body = {
        "model": api.model,
        "max_tokens": 256,
        "messages": messages,
    }
    body.update(fields)
    return body


def require_supported(response, endpoint: str) -> dict:
    if response.status != 200:
        flm_api.raise_for_capability(response.error_message())
        raise AssertionError(f"HTTP {response.status} from {endpoint}: {response.error_message()}")
    return response.body


@case
def anthropic_messages_text(api: Api) -> None:
    """Text messages and Claude Code's in-history system reminder are accepted."""
    response = api.post(anthropic_body(
        api, [
            {"role": "user", "content": "Reply with exactly: pong"},
            {"role": "system", "content": [
                {"type": "text", "text": "Keep the response concise."},
            ]},
        ],
    ), url=messages_url(api))
    body = require_supported(response, "/v1/messages")
    expect(body.get("type") == "message", f"unexpected response type: {body}")
    expect(body.get("role") == "assistant", f"unexpected response role: {body}")
    expect(body.get("content") and body["content"][0].get("type") == "text",
           f"missing Anthropic text block: {body}")
    expect("pong" in body["content"][0].get("text", "").lower(),
           f"unexpected model text: {body['content']!r}")


@case
def anthropic_messages_required_tool(api: Api) -> None:
    """tool_choice any returns an Anthropic tool_use block."""
    response = api.post(anthropic_body(
        api,
        [{"role": "user", "content": "Call get_weather for Paris."}],
        tools=[tool()],
        tool_choice={"type": "any"},
        temperature=0,
    ), url=messages_url(api))
    body = require_supported(response, "/v1/messages")
    calls = [block for block in body.get("content", []) if block.get("type") == "tool_use"]
    expect(bool(calls), f"expected tool_use block, got {body}")
    expect(calls[0].get("name") == "get_weather", f"unexpected tool: {calls[0]}")
    expect(calls[0].get("input", {}).get("city") == "Paris", f"unexpected tool input: {calls[0]}")
    expect(body.get("stop_reason") == "tool_use", f"unexpected stop_reason: {body}")


@case
def anthropic_messages_stream_tool(api: Api) -> None:
    """The Messages SSE stream emits tool_use block events."""
    stream = api.stream(anthropic_body(
        api,
        [{"role": "user", "content": "Call get_weather for Paris."}],
        tools=[tool()],
        tool_choice={"type": "any"},
        temperature=0,
    ), url=messages_url(api))
    if stream.status != 200:
        flm_api.raise_for_capability(stream.error_message())
        raise AssertionError(f"HTTP {stream.status} from streamed /v1/messages: {stream.error_message()}")
    event_types = [chunk.get("type") for chunk in stream.chunks]
    expect("message_start" in event_types, f"missing message_start: {event_types}")
    expect("content_block_start" in event_types, f"missing content_block_start: {event_types}")
    expect("content_block_delta" in event_types, f"missing tool argument deltas: {event_types}")
    expect("message_stop" in event_types, f"missing message_stop: {event_types}")


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
