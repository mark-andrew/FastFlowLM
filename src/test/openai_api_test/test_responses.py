#!/usr/bin/env python3
"""Stateless OpenAI Responses API: text, function calls, streaming, and tool results."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import flm_api
from flm_api import Api, case, expect

TEST_TOOL = {
    "type": "function",
    "name": "test_tool",
    "description": "Return a test value.",
    "parameters": {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    },
}

TOOL_PROMPT = "Call test_tool with value hello."


def output_budget(api: Api, base: int) -> int:
    """Leave room for models such as gpt-oss that reason before answering."""
    return base + (api.reasoning_budget if api.always_thinks else 0)


def responses_url(api: Api) -> str:
    return api.url.replace("/v1/chat/completions", "/v1/responses")


def post_responses(api: Api, body: dict) -> flm_api.Response:
    request_body = dict(body)
    request_body["stream"] = False
    try:
        with api._open(request_body, responses_url(api)) as response:
            return flm_api.Response(response.status, json.loads(response.read().decode("utf-8")))
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"error": {"message": raw}}
        return flm_api.Response(error.code, parsed)


def stream_responses(api: Api, body: dict) -> tuple[int, list[dict], str | None]:
    request_body = dict(body)
    request_body["stream"] = True
    request = urllib.request.Request(
        responses_url(api),
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=api.timeout)
    except urllib.error.HTTPError as error:
        return error.code, [], error.read().decode("utf-8", "replace")

    events: list[dict] = []
    with response:
        if not response.headers.get("Content-Type", "").startswith("text/event-stream"):
            return response.status, [], response.read().decode("utf-8", "replace")
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                continue
            events.append(json.loads(payload))
    return response.status, events, None


def expect_response_tool_call(response: dict) -> dict:
    expect(response.get("object") == "response", f"object is {response.get('object')!r}")
    expect(response.get("status") in ("completed", "incomplete"),
           f"status is {response.get('status')!r}")
    calls = [item for item in response.get("output", []) if item.get("type") == "function_call"]
    expect(len(calls) == 1, f"expected one function_call, got {response.get('output')!r}")
    call = calls[0]
    expect(call.get("name") == "test_tool", f"wrong function: {call!r}")
    expect(isinstance(call.get("call_id"), str) and call["call_id"].startswith("call_"),
           f"invalid call_id: {call.get('call_id')!r}")
    try:
        args = json.loads(call.get("arguments", ""))
    except (TypeError, json.JSONDecodeError) as error:
        raise AssertionError(f"function arguments are not valid JSON: {call.get('arguments')!r}") from error
    expect(args == {"value": "hello"}, f"wrong arguments: {args!r}")
    return call


def body(api: Api, input_value, **fields) -> dict:
    result = {
        "model": api.model,
        "input": input_value,
        "max_output_tokens": output_budget(api, 2048),
        "temperature": 0.0,
        "tools": [TEST_TOOL],
    }
    result.update(fields)
    return result


@case
def responses_text(api: Api) -> None:
    """A plain Responses request returns the standard response envelope."""
    response = post_responses(api, {
        "model": api.model,
        "input": "Say hello in one short sentence.",
        "max_output_tokens": output_budget(api, 64),
        "temperature": 0.0,
    })
    expect(response.status == 200, f"HTTP {response.status}: {response.error_message()}")
    expect(response.body.get("object") == "response", f"object is {response.body.get('object')!r}")
    expect(response.body.get("status") == "completed", f"status is {response.body.get('status')!r}")
    messages = [item for item in response.body.get("output", []) if item.get("type") == "message"]
    expect(bool(messages), f"no message in output: {response.body.get('output')!r}")
    text = "".join(
        part.get("text", "")
        for item in messages
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
    expect(text.strip() != "", "output text is empty")
    usage = response.body.get("usage") or {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        expect(isinstance(usage.get(key), int), f"usage.{key} missing: {usage}")


@case
def responses_required_function(api: Api) -> None:
    """tool_choice required maps to a Responses function_call output item."""
    response = post_responses(api, body(api, TOOL_PROMPT, tool_choice="required"))
    if response.status != 200:
        flm_api.raise_for_capability(response.error_message())
        raise AssertionError(f"HTTP {response.status}: {response.error_message()}")
    expect_response_tool_call(response.body)


@case
def responses_named_function(api: Api) -> None:
    """A named Responses function choice maps to the matching chat function choice."""
    response = post_responses(api, body(
        api, TOOL_PROMPT, tool_choice={"type": "function", "name": "test_tool"}))
    if response.status != 200:
        flm_api.raise_for_capability(response.error_message())
        raise AssertionError(f"HTTP {response.status}: {response.error_message()}")
    expect_response_tool_call(response.body)


@case
def responses_stream_required_function(api: Api) -> None:
    """Streaming emits Responses lifecycle and function-call events with a completed item."""
    status, events, error = stream_responses(
        api, body(api, TOOL_PROMPT, tool_choice="required"))
    if status != 200 or error is not None:
        message = error or f"HTTP {status}"
        flm_api.raise_for_capability(message)
        raise AssertionError(message)

    event_types = [event.get("type") for event in events]
    expect("response.created" in event_types, f"missing response.created: {event_types}")
    expect("response.output_item.added" in event_types, f"missing output_item.added: {event_types}")
    expect("response.function_call_arguments.done" in event_types,
           f"missing function_call_arguments.done: {event_types}")
    completed = [event for event in events if event.get("type") == "response.completed"]
    expect(len(completed) == 1, f"expected one response.completed: {event_types}")
    expect_response_tool_call(completed[0]["response"])


@case
def responses_tool_round_trip(api: Api) -> None:
    """A stateless follow-up can include a prior function call and its output."""
    first = post_responses(api, body(api, TOOL_PROMPT, tool_choice="required"))
    if first.status != 200:
        flm_api.raise_for_capability(first.error_message())
        raise AssertionError(f"HTTP {first.status}: {first.error_message()}")
    call = expect_response_tool_call(first.body)
    history = [
        {"role": "user", "content": TOOL_PROMPT},
        {
            "type": "function_call",
            "call_id": call["call_id"],
            "name": call["name"],
            "arguments": call["arguments"],
        },
        {
            "type": "function_call_output",
            "call_id": call["call_id"],
            "output": "The test result is 23.",
        },
        {"role": "user", "content": "Report the test result."},
    ]
    response = post_responses(api, body(
        api, history, tool_choice="none", max_output_tokens=output_budget(api, 128)))
    expect(response.status == 200, f"HTTP {response.status}: {response.error_message()}")
    text = response.body.get("output_text", "")
    if not text:
        text = "".join(
            part.get("text", "")
            for item in response.body.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
    expect("23" in text, f"follow-up did not use the function result: {text!r}")


@case
def responses_reject_stateful_reference(api: Api) -> None:
    """Stateful response references are rejected rather than silently ignored."""
    response = post_responses(api, {
        "model": api.model,
        "input": "Say hello.",
        "previous_response_id": "resp_previous",
    })
    expect(response.status == 400, f"expected HTTP 400, got {response.status}: {response.body}")
    expect("previous_response_id" in response.error_message(),
           f"error did not explain unsupported state: {response.error_message()!r}")


CASES = flm_api.collect(globals())

if __name__ == "__main__":
    flm_api.main(CASES, __doc__.splitlines()[0])
