/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file openai_responses.hpp
 * \brief OpenAI Responses API adapters for FastFlowLM's chat-completions engine
 */
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace openai_responses {

using Json = nlohmann::ordered_json;
using StreamCallback = std::function<void(const Json&, bool)>;

/// Convert the stateless subset of a Responses request into a Chat Completions request.
/// Returns false and describes the invalid/unsupported field when conversion is impossible.
bool to_chat_request(const Json& request, const std::string& fallback_model,
                    Json& chat_request, std::string& error, std::string& param);

/// Convert a non-streaming Chat Completions response to a Responses response.
Json from_chat_response(const Json& response, const Json& request);

/// Translate Chat Completions SSE chunks to Responses API events.
class StreamEncoder {
public:
    StreamEncoder(std::string model, Json request, StreamCallback callback);
    void on_chat_chunk(const Json& chunk, bool is_final);

private:
    struct ToolState {
        int output_index = -1;
        std::string item_id;
        std::string call_id;
        std::string name;
        std::string arguments;
    };

    void start();
    void finish();
    void fail(const Json& error);
    void emit(const std::string& event, Json data, bool is_final = false);
    void ensure_message();
    void finish_message(bool complete);
    ToolState* ensure_tool(const Json& delta);
    void finish_tools(bool complete);
    Json response_object(const std::string& status) const;
    Json usage_;

    std::string model_;
    Json request_;
    StreamCallback callback_;
    std::string response_id_;
    long long created_at_ = 0;
    int sequence_number_ = 0;
    bool started_ = false;
    bool finished_ = false;
    bool failed_ = false;
    Json failure_;
    std::string finish_reason_;
    Json output_ = Json::array();
    std::map<int, ToolState> tools_;
    int message_output_index_ = -1;
    std::string message_id_;
    std::string message_text_;
};

} // namespace openai_responses
