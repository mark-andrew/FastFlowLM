/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file anthropic_messages.hpp
 * \brief Anthropic Messages API adapters for FastFlowLM's chat-completions engine
 */
#pragma once

#include <functional>
#include <map>
#include <string>

#include <nlohmann/json.hpp>

namespace anthropic_messages {

using Json = nlohmann::ordered_json;
using StreamCallback = std::function<void(const Json&, bool)>;

/// Convert the supported stateless subset of an Anthropic Messages request to Chat Completions.
/// Tool results in user messages are expanded into OpenAI-style tool messages.
bool to_chat_request(const Json& request, const std::string& fallback_model,
                    Json& chat_request, std::string& error, std::string& param);

/// Convert a Chat Completions response to an Anthropic message or error response.
Json from_chat_response(const Json& response, const Json& request);

/// Translate Chat Completions SSE chunks into Anthropic Messages events.
class StreamEncoder {
public:
    StreamEncoder(std::string model, StreamCallback callback);
    void on_chat_chunk(const Json& chunk, bool is_final);
    void on_chat_error(const Json& error);
    bool started() const { return started_; }

private:
    struct ToolState {
        int block_index = -1;
        std::string id;
        std::string name;
        std::string arguments;
        bool started = false;
    };

    void start();
    void finish();
    void fail(const Json& error);
    void emit(const std::string& event, Json data, bool is_final = false);
    ToolState& ensure_tool(const Json& delta);
    void close_blocks();
    Json error_object(const Json& error) const;

    std::string model_;
    StreamCallback callback_;
    std::string message_id_;
    std::string finish_reason_;
    int next_block_index_ = 0;
    int text_block_index_ = -1;
    bool started_ = false;
    bool finished_ = false;
    int input_tokens_ = 0;
    int output_tokens_ = 0;
    std::string text_;
    Json content_ = Json::array();
    std::map<int, ToolState> tools_;
};

Json make_error(const std::string& message, const std::string& type = "invalid_request_error",
                const std::string& param = "");

} // namespace anthropic_messages
