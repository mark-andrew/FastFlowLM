/// \file test_anthropic_messages.cpp
/// \brief Anthropic Messages API adapters, exercised without an NPU.
#include "anthropic_messages.hpp"

#include <cstdio>
#include <string>
#include <vector>

using anthropic_messages::Json;

namespace {

int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            failures++;                                                             \
        }                                                                           \
    } while (0)

Json tool() {
    return {
        {"name", "get_weather"},
        {"description", "Get the weather for a city."},
        {"input_schema", {
            {"type", "object"},
            {"properties", {{"city", {{"type", "string"}}}}},
            {"required", Json::array({"city"})}
        }}
    };
}

void test_request_conversion_and_tool_round_trip() {
    std::printf("Anthropic messages, tools, and tool results map to chat history\n");
    Json system_block = {{"type", "text"}, {"text", "Be concise."},
                         {"cache_control", {{"type", "ephemeral"}}}};
    Json assistant_content = Json::array({
        Json{{"type", "text"}, {"text", "I'll check."}},
        Json{{"type", "tool_use"}, {"id", "toolu_123"}, {"name", "get_weather"},
             {"input", {{"city", "York"}}}}
    });
    Json result_block = {{"type", "text"}, {"text", "Sunny, 18 C."}};
    Json tool_result = {{"type", "tool_result"}, {"tool_use_id", "toolu_123"},
                        {"content", Json::array({result_block})}};
    Json request = {
        {"model", "qwen3.6-moe:35b-a3b"},
        {"max_tokens", 256},
        {"system", Json::array({system_block})},
        {"tools", Json::array({tool()})},
        {"tool_choice", {{"type", "tool"}, {"name", "get_weather"},
                         {"disable_parallel_tool_use", true}}},
        {"messages", Json::array({
            Json{{"role", "user"}, {"content", "Weather in York?"}},
            Json{{"role", "assistant"}, {"content", assistant_content}},
            Json{{"role", "user"}, {"content", Json::array({tool_result})}},
            Json{{"role", "system"}, {"content", Json::array({
                Json{{"type", "text"}, {"text", "Follow the active workflow instruction."}}
            })}}
        })}
    };

    Json chat;
    std::string error;
    std::string param;
    CHECK(anthropic_messages::to_chat_request(request, "fallback", chat, error, param));
    CHECK(chat["model"] == "qwen3.6-moe:35b-a3b");
    CHECK(chat["max_tokens"] == 256);
    CHECK(chat["messages"].size() == 4);
    CHECK(chat["messages"][0]["role"] == "system");
    CHECK(chat["messages"][0]["content"] == "Be concise.\nFollow the active workflow instruction.");
    CHECK(chat["messages"][1]["role"] == "user");
    CHECK(chat["messages"][2]["role"] == "assistant");
    CHECK(chat["messages"][2]["tool_calls"][0]["id"] == "toolu_123");
    CHECK(chat["messages"][2]["content"] == "I'll check.");
    CHECK(chat["messages"][3]["role"] == "tool");
    CHECK(chat["messages"][3]["tool_call_id"] == "toolu_123");
    CHECK(chat["messages"][3]["content"] == "Sunny, 18 C.");
    CHECK(chat["tools"][0]["function"]["parameters"]["required"][0] == "city");
    CHECK(chat["tool_choice"]["function"]["name"] == "get_weather");
    CHECK(chat["parallel_tool_calls"] == false);
}

void test_image_and_request_validation() {
    std::printf("Anthropic inline images map to OpenAI data URLs; URL images are refused\n");
    Json request = {
        {"max_tokens", 32},
        {"messages", Json::array({
            {{"role", "user"}, {"content", Json::array({
                {{"type", "text"}, {"text", "What is in this picture?"}},
                {{"type", "image"}, {"source", {
                    {"type", "base64"}, {"media_type", "image/png"}, {"data", "AQID"}
                }}}
            })}}
        })}
    };
    Json chat;
    std::string error;
    std::string param;
    CHECK(anthropic_messages::to_chat_request(request, "fallback", chat, error, param));
    CHECK(chat["messages"][0]["content"][1]["image_url"]["url"] == "data:image/png;base64,AQID");

    request["messages"][0]["content"][1]["source"]["type"] = "url";
    CHECK(!anthropic_messages::to_chat_request(request, "fallback", chat, error, param));
    CHECK(param == "messages");
}

void test_nonstream_response_conversion() {
    std::printf("Chat tool calls become Anthropic tool_use response blocks\n");
    const Json response = {
        {"model", "qwen3.6-moe:35b-a3b"},
        {"usage", {{"prompt_tokens", 50}, {"completion_tokens", 12},
                   {"total_tokens", 62}, {"prompt_tokens_details", {{"cached_tokens", 20}}}}},
        {"choices", Json::array({Json{
            {"finish_reason", "tool_calls"},
            {"message", {
                {"role", "assistant"},
                {"content", "Checking."},
                {"tool_calls", Json::array({Json{
                    {"id", "call_123"}, {"type", "function"},
                    {"function", {{"name", "get_weather"}, {"arguments", "{\"city\":\"York\"}"}}}
                }})}
            }}
        }})}
    };
    const Json message = anthropic_messages::from_chat_response(response, Json::object());
    CHECK(message["type"] == "message");
    CHECK(message["stop_reason"] == "tool_use");
    CHECK(message["content"].size() == 2);
    CHECK(message["content"][0]["type"] == "text");
    CHECK(message["content"][1]["type"] == "tool_use");
    CHECK(message["content"][1]["name"] == "get_weather");
    CHECK(message["content"][1]["input"]["city"] == "York");
    CHECK(message["usage"]["input_tokens"] == 50);
    CHECK(message["usage"]["cache_read_input_tokens"] == 20);
}

void test_stream_events() {
    std::printf("Anthropic streaming emits text and tool_use events\n");
    std::vector<std::string> events;
    std::vector<bool> finals;
    anthropic_messages::StreamEncoder encoder("qwen3.6-moe:35b-a3b",
        [&](const Json& data, bool final) {
            events.push_back(data.is_string() ? data.get<std::string>() : data.dump());
            finals.push_back(final);
        });

    encoder.on_chat_chunk(Json("data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"index\":0}]}\n\n"), false);
    encoder.on_chat_chunk(Json("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_x\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\"}}]},\"index\":0}]}\n\n"), false);
    encoder.on_chat_chunk(Json("data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"York\\\"}\"}}]},\"finish_reason\":\"tool_calls\",\"index\":0}]}\n\n"), false);
    encoder.on_chat_chunk(Json("data: [DONE]\n\n"), true);

    const std::string joined = [&] {
        std::string all;
        for (const auto& event : events) all += event;
        return all;
    }();
    CHECK(joined.find("event: message_start") != std::string::npos);
    CHECK(joined.find("event: content_block_delta") != std::string::npos);
    CHECK(joined.find("\"type\":\"text_delta\"") != std::string::npos);
    CHECK(joined.find("\"type\":\"tool_use\"") != std::string::npos);
    CHECK(joined.find("\"type\":\"input_json_delta\"") != std::string::npos);
    CHECK(joined.find("\"stop_reason\":\"tool_use\"") != std::string::npos);
    CHECK(joined.find("event: message_stop") != std::string::npos);
    CHECK(!finals.empty() && finals.back());
}

} // namespace

int main() {
    test_request_conversion_and_tool_round_trip();
    test_image_and_request_validation();
    test_nonstream_response_conversion();
    test_stream_events();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all passed\n");
    return 0;
}
