#include "openai_responses.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using openai_responses::Json;

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

Json tool() {
    return {
        {"type", "function"},
        {"name", "test_tool"},
        {"description", "A test tool."},
        {"parameters", {
            {"type", "object"},
            {"properties", {{"value", {{"type", "string"}}}}},
            {"required", Json::array({"value"})},
            {"additionalProperties", false}
        }}
    };
}

void test_request_conversion_and_history() {
    Json request = {
        {"model", "gpt-oss:20b"},
        {"instructions", "Be concise."},
        {"input", Json::array({
            {
                {"role", "user"},
                {"content", Json::array({
                    {{"type", "input_text"}, {"text", "Call the tool."}}
                })}
            },
            {
                {"type", "function_call"},
                {"call_id", "call_previous"},
                {"name", "test_tool"},
                {"arguments", "{\"value\":\"hello\"}"}
            },
            {
                {"type", "function_call_output"},
                {"call_id", "call_previous"},
                {"output", "{\"result\":\"done\"}"}
            },
            {{"role", "user"}, {"content", "Now summarize the result."}}
        })},
        {"tools", Json::array({tool()})},
        {"tool_choice", {{"type", "function"}, {"name", "test_tool"}}},
        {"max_output_tokens", 256},
        {"reasoning", {{"effort", "low"}}},
        {"stream", true}
    };

    Json chat;
    std::string error;
    std::string param;
    check(openai_responses::to_chat_request(request, "fallback", chat, error, param),
          "valid Responses request should convert");
    check(chat.value("model", "") == "gpt-oss:20b", "model should be preserved");
    check(chat.value("stream", false), "stream should be preserved");
    check(chat.value("max_completion_tokens", 0) == 256, "max output tokens should map");
    check(chat.value("reasoning_effort", "") == "low", "reasoning effort should map");
    check(chat.at("tool_choice").at("function").at("name") == "test_tool",
          "named Responses tool choice should map to Chat Completions");
    check(chat.at("messages").size() == 5, "instructions and function round-trip should map to messages");
    check(chat.at("messages")[0].at("role") == "system", "instructions should become system context");
    check(chat.at("messages")[1].at("role") == "user", "user input should be preserved");
    check(chat.at("messages")[2].at("role") == "assistant", "function call should become assistant tool_calls");
    check(chat.at("messages")[2].at("tool_calls")[0].at("id") == "call_previous",
          "function call ID should be preserved");
    check(chat.at("messages")[3].at("role") == "tool", "function output should become a tool message");
    check(chat.at("messages")[4].at("role") == "user", "follow-up user input should be preserved");
}

void test_rejects_stateful_requests() {
    Json request = {
        {"model", "gpt-oss:20b"},
        {"input", "Continue."},
        {"previous_response_id", "resp_previous"}
    };
    Json chat;
    std::string error;
    std::string param;
    check(!openai_responses::to_chat_request(request, "", chat, error, param),
          "previous_response_id should be rejected");
    check(param == "previous_response_id", "stateful error should name the field");
}

void test_nonstream_response_conversion() {
    const Json chat = {
        {"model", "gpt-oss:20b"},
        {"created", 123},
        {"usage", {
            {"prompt_tokens", 10},
            {"completion_tokens", 4},
            {"total_tokens", 14},
            {"prompt_tokens_details", {{"cached_tokens", 2}}}
        }},
        {"choices", Json::array({
            {
                {"finish_reason", "tool_calls"},
                {"message", {
                    {"role", "assistant"},
                    {"content", nullptr},
                    {"tool_calls", Json::array({
                        {
                            {"id", "call_test"},
                            {"type", "function"},
                            {"function", {
                                {"name", "test_tool"},
                                {"arguments", "{\"value\":\"hello\"}"}
                            }}
                        }
                    })}
                }}
            }
        })}
    };
    const Json response = openai_responses::from_chat_response(chat, Json::object());
    check(response.value("object", "") == "response", "Responses object should have the right type");
    check(response.value("status", "") == "completed", "tool call should complete the response");
    check(response.at("output").size() == 1, "response should contain one function call");
    check(response.at("output")[0].value("type", "") == "function_call",
          "chat tool call should become a Responses function_call");
    check(response.at("output")[0].value("call_id", "") == "call_test",
          "function call ID should be preserved");
    check(response.at("usage").value("input_tokens", 0) == 10, "input usage should be mapped");
    check(response.at("usage").value("output_tokens", 0) == 4, "output usage should be mapped");
}

std::string event_type(const Json& event) {
    return event.value("type", "");
}

void test_stream_conversion() {
    std::vector<Json> events;
    std::vector<bool> finals;
    openai_responses::StreamEncoder encoder(
        "gpt-oss:20b", Json{{"tools", Json::array({tool()})}},
        [&](const Json& data, bool is_final) {
            check(data.is_string(), "Responses stream event should be an SSE string");
            const std::string raw = data.get<std::string>();
            const auto marker = raw.find("data: ");
            check(marker != std::string::npos, "SSE event should have a data line");
            if (marker != std::string::npos) {
                const auto start = marker + 6;
                const auto end = raw.find("\n\n", start);
                events.push_back(Json::parse(raw.substr(start, end - start)));
            }
            finals.push_back(is_final);
        });

    const auto chunk = [](Json body) {
        return Json("data: " + body.dump() + "\n\n");
    };
    const Json tool_start = {
        {"index", 0},
        {"id", "call_test"},
        {"type", "function"},
        {"function", {{"name", "test_tool"}, {"arguments", ""}}}
    };
    const Json start_chunk = {
        {"choices", Json::array({
            {
                {"delta", {{"tool_calls", Json::array({tool_start})}}},
                {"finish_reason", nullptr}
            }
        })}
    };
    encoder.on_chat_chunk(chunk(start_chunk), false);

    const Json argument_delta = {
        {"choices", Json::array({
            {
                {"delta", {{"tool_calls", Json::array({
                    {{"index", 0}, {"function", {{"arguments", "{\"value\":\"hello\"}"}}}}
                })}}},
                {"finish_reason", nullptr}
            }
        })}
    };
    encoder.on_chat_chunk(chunk(argument_delta), false);

    const Json finish_chunk = {
        {"choices", Json::array({
            {{"delta", {{"content", nullptr}}}, {"finish_reason", "tool_calls"}}
        })}
    };
    encoder.on_chat_chunk(chunk(finish_chunk), false);
    encoder.on_chat_chunk(Json("data: [DONE]\n\n"), true);

    bool saw_call = false;
    bool saw_completed = false;
    for (const Json& event : events) {
        if (event_type(event) == "response.output_item.done" &&
            event.at("item").value("type", "") == "function_call") {
            saw_call = event.at("item").value("name", "") == "test_tool" &&
                       event.at("item").value("arguments", "") == "{\"value\":\"hello\"}";
        }
        if (event_type(event) == "response.completed") {
            saw_completed = true;
            check(event.at("response").at("output").size() == 1,
                  "completed response should contain the function call");
        }
    }
    check(saw_call, "stream should finish with the complete function call");
    check(saw_completed, "stream should finish with response.completed");
    check(!finals.empty() && finals.back(), "last event should close the HTTP stream");
}

} // namespace

int main() {
    test_request_conversion_and_history();
    test_rejects_stateful_requests();
    test_nonstream_response_conversion();
    test_stream_conversion();
    if (failures) {
        std::cerr << failures << " test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "All OpenAI Responses adapter tests passed\n";
    return EXIT_SUCCESS;
}
