/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file anthropic_messages.cpp
 * \brief Stateless Anthropic Messages API adapters for FastFlowLM chat completions
 */
#include "anthropic_messages.hpp"

#include <algorithm>
#include <limits>
#include <random>
#include <utility>
#include <vector>

namespace anthropic_messages {

namespace {

std::string make_id(const char* prefix) {
    static const char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    thread_local std::mt19937 generator{std::random_device{}()};
    std::uniform_int_distribution<size_t> pick(0, sizeof(alphabet) - 2);
    std::string id(prefix);
    for (int i = 0; i < 24; ++i) id += alphabet[pick(generator)];
    return id;
}

bool fail(std::string& error, std::string& param, std::string message, std::string field) {
    error = std::move(message);
    param = std::move(field);
    return false;
}

bool has_value(const Json& object, const char* key) {
    auto it = object.find(key);
    return it != object.end() && !it->is_null();
}

bool append_text_block(const Json& block, Json& parts, std::string& error, std::string& param) {
    if (!block.contains("text") || !block["text"].is_string()) {
        return fail(error, param, "text blocks must contain a string 'text'", "messages");
    }
    parts.push_back({{"type", "text"}, {"text", block["text"]}});
    return true;
}

bool append_image_block(const Json& block, Json& parts, std::string& error, std::string& param) {
    if (!block.contains("source") || !block["source"].is_object()) {
        return fail(error, param, "image blocks must contain a 'source' object", "messages");
    }
    const Json& source = block["source"];
    if (source.value("type", "") != "base64") {
        return fail(error, param, "only inline base64 image sources are supported", "messages");
    }
    if (!source.contains("media_type") || !source["media_type"].is_string() ||
        !source.contains("data") || !source["data"].is_string()) {
        return fail(error, param, "base64 image sources require string 'media_type' and 'data'", "messages");
    }
    const std::string media_type = source["media_type"].get<std::string>();
    const std::string data = source["data"].get<std::string>();
    if (media_type != "image/png" && media_type != "image/jpeg" && media_type != "image/webp") {
        return fail(error, param, "unsupported image media_type '" + media_type + "'", "messages");
    }
    parts.push_back({
        {"type", "image_url"},
        {"image_url", {{"url", "data:" + media_type + ";base64," + data}}}
    });
    return true;
}

bool tool_result_to_text(const Json& content, std::string& text, std::string& error, std::string& param) {
    if (content.is_null()) {
        text.clear();
        return true;
    }
    if (content.is_string()) {
        text = content.get<std::string>();
        return true;
    }
    if (!content.is_array()) {
        return fail(error, param, "tool_result content must be a string or an array of text blocks", "messages");
    }
    text.clear();
    for (const Json& block : content) {
        if (!block.is_object() || block.value("type", "") != "text" ||
            !block.contains("text") || !block["text"].is_string()) {
            return fail(error, param, "only text blocks are supported in tool_result content", "messages");
        }
        text += block["text"].get<std::string>();
    }
    return true;
}

Json anthropic_error(const Json& error) {
    Json detail = error.is_object() && error.contains("error") && error["error"].is_object()
        ? error["error"]
        : error.is_object() ? error : Json{{"message", error.dump()}};
    std::string type = detail.value("type", "api_error");
    if (type == "invalid_request_error") type = "invalid_request_error";
    else if (type == "authentication_error" || type == "permission_error" ||
             type == "not_found_error" || type == "rate_limit_error" ||
             type == "api_error" || type == "overloaded_error") {
        // Keep known Anthropic error types.
    } else {
        type = "api_error";
    }
    Json body = {
        {"type", "error"},
        {"error", {
            {"type", type},
            {"message", detail.value("message", "FastFlowLM request failed")}
        }}
    };
    if (detail.contains("code") && detail["code"].is_number_integer()) {
        body["error"]["code"] = detail["code"];
    }
    return body;
}

std::string anthropic_stop_reason(const std::string& finish_reason) {
    if (finish_reason == "tool_calls" || finish_reason == "function_call") return "tool_use";
    if (finish_reason == "length") return "max_tokens";
    if (finish_reason == "content_filter") return "refusal";
    return "end_turn";
}

bool parse_tool_arguments(const Json& call, Json& input, std::string& error) {
    if (!call.is_object() || !call.contains("function") || !call["function"].is_object()) {
        error = "Chat Completions returned a malformed tool call";
        return false;
    }
    const Json& function = call["function"];
    const std::string arguments = function.value("arguments", std::string("{}"));
    try {
        input = Json::parse(arguments);
    } catch (const std::exception&) {
        error = "Chat Completions returned invalid JSON tool arguments";
        return false;
    }
    if (!input.is_object()) {
        error = "Anthropic tool_use input must be a JSON object";
        return false;
    }
    return true;
}

} // namespace

Json make_error(const std::string& message, const std::string& type, const std::string& param) {
    const int code = type == "invalid_request_error" ? 400 : 500;
    Json error = {{"type", type}, {"message", message}, {"code", code}};
    if (!param.empty()) error["param"] = param;
    return {{"type", "error"}, {"error", std::move(error)}};
}

bool to_chat_request(const Json& request, const std::string& fallback_model,
                     Json& chat_request, std::string& error, std::string& param) {
    error.clear();
    param.clear();
    if (!request.is_object()) {
        return fail(error, param, "request body must be a JSON object", "");
    }
    if (!has_value(request, "messages") || !request["messages"].is_array()) {
        return fail(error, param, "'messages' must be an array", "messages");
    }
    if (!has_value(request, "max_tokens") || !request["max_tokens"].is_number_integer() ||
        request["max_tokens"].get<long long>() <= 0 ||
        request["max_tokens"].get<long long>() > std::numeric_limits<int>::max()) {
        return fail(error, param, "'max_tokens' must be a positive integer", "max_tokens");
    }
    if (has_value(request, "model") && !request["model"].is_string()) {
        return fail(error, param, "'model' must be a string", "model");
    }
    if (has_value(request, "stream") && !request["stream"].is_boolean()) {
        return fail(error, param, "'stream' must be a boolean", "stream");
    }

    const std::string model = has_value(request, "model")
        ? request["model"].get<std::string>() : fallback_model;
    chat_request = {
        {"model", model},
        {"messages", Json::array()},
        {"max_tokens", request["max_tokens"]}
    };
    if (has_value(request, "stream")) chat_request["stream"] = request["stream"];
    for (const char* field : {"temperature", "top_p", "top_k"}) {
        if (has_value(request, field)) chat_request[field] = request[field];
    }
    if (has_value(request, "stop_sequences")) {
        if (!request["stop_sequences"].is_array()) {
            return fail(error, param, "'stop_sequences' must be an array", "stop_sequences");
        }
        chat_request["stop"] = request["stop_sequences"];
    }

    if (has_value(request, "thinking")) {
        const Json& thinking = request["thinking"];
        if (!thinking.is_object() || !thinking.contains("type") || !thinking["type"].is_string()) {
            return fail(error, param, "'thinking' must contain a string 'type'", "thinking");
        }
        const std::string type = thinking["type"].get<std::string>();
        if (type == "enabled" || type == "adaptive") chat_request["think"] = true;
        else if (type == "disabled") chat_request["think"] = false;
        else return fail(error, param, "thinking.type must be 'enabled', 'adaptive', or 'disabled'", "thinking.type");
    }

    if (has_value(request, "system")) {
        const Json& system = request["system"];
        std::string system_text;
        if (system.is_string()) {
            system_text = system.get<std::string>();
        } else if (system.is_array()) {
            for (const Json& block : system) {
                if (!block.is_object() || block.value("type", "") != "text" ||
                    !block.contains("text") || !block["text"].is_string()) {
                    return fail(error, param, "system must be a string or an array of text blocks", "system");
                }
                if (!system_text.empty()) system_text += "\n";
                system_text += block["text"].get<std::string>();
            }
        } else {
            return fail(error, param, "system must be a string or an array of text blocks", "system");
        }
        if (!system_text.empty()) {
            chat_request["messages"].push_back({{"role", "system"}, {"content", system_text}});
        }
    }

    Json tools = Json::array();
    if (has_value(request, "tools")) {
        if (!request["tools"].is_array()) {
            return fail(error, param, "'tools' must be an array", "tools");
        }
        for (const Json& tool : request["tools"]) {
            if (!tool.is_object() || !tool.contains("name") || !tool["name"].is_string() ||
                !tool.contains("input_schema") || !tool["input_schema"].is_object()) {
                return fail(error, param, "each tool requires a string 'name' and object 'input_schema'", "tools");
            }
            Json function = {
                {"name", tool["name"]},
                {"parameters", tool["input_schema"]}
            };
            if (has_value(tool, "description")) function["description"] = tool["description"];
            tools.push_back({{"type", "function"}, {"function", std::move(function)}});
        }
    }
    if (!tools.empty()) chat_request["tools"] = std::move(tools);

    if (has_value(request, "tool_choice")) {
        const Json& choice = request["tool_choice"];
        if (!choice.is_object() || !choice.contains("type") || !choice["type"].is_string()) {
            return fail(error, param, "'tool_choice' must be an object with a string 'type'", "tool_choice");
        }
        const std::string type = choice["type"].get<std::string>();
        if (type == "auto") {
            chat_request["tool_choice"] = "auto";
        } else if (type == "any") {
            chat_request["tool_choice"] = "required";
        } else if (type == "tool") {
            if (!choice.contains("name") || !choice["name"].is_string()) {
                return fail(error, param, "tool_choice.type 'tool' requires a string 'name'", "tool_choice.name");
            }
            chat_request["tool_choice"] = {
                {"type", "function"},
                {"function", {{"name", choice["name"]}}}
            };
        } else {
            return fail(error, param, "tool_choice.type must be 'auto', 'any', or 'tool'", "tool_choice.type");
        }
        if (has_value(choice, "disable_parallel_tool_use")) {
            if (!choice["disable_parallel_tool_use"].is_boolean()) {
                return fail(error, param, "'disable_parallel_tool_use' must be a boolean", "tool_choice.disable_parallel_tool_use");
            }
            chat_request["parallel_tool_calls"] = !choice["disable_parallel_tool_use"].get<bool>();
        }
    } else {
        chat_request["tool_choice"] = "auto";
    }

    std::vector<std::string> inline_system_texts;
    for (const Json& message : request["messages"]) {
        if (!message.is_object() || !message.contains("role") || !message["role"].is_string()) {
            return fail(error, param, "each message requires a string 'role'", "messages");
        }
        const std::string role = message["role"].get<std::string>();
        if (role != "user" && role != "assistant" && role != "system") {
            return fail(error, param, "message role must be 'user', 'assistant', or 'system'", "messages.role");
        }
        if (!message.contains("content")) {
            return fail(error, param, "each message requires 'content'", "messages.content");
        }
        const Json& content = message["content"];
        if (role == "system") {
            std::string system_text;
            if (content.is_string()) {
                system_text = content.get<std::string>();
            } else if (content.is_array()) {
                for (const Json& block : content) {
                    if (!block.is_object() || block.value("type", "") != "text" ||
                        !block.contains("text") || !block["text"].is_string()) {
                        return fail(error, param, "system messages only support text blocks", "messages.content");
                    }
                    if (!system_text.empty()) system_text += "\n";
                    system_text += block["text"].get<std::string>();
                }
            } else {
                return fail(error, param, "system message content must be a string or text blocks", "messages.content");
            }
            if (!system_text.empty()) inline_system_texts.push_back(std::move(system_text));
            continue;
        }
        if (content.is_string()) {
            chat_request["messages"].push_back({{"role", role}, {"content", content}});
            continue;
        }
        if (!content.is_array()) {
            return fail(error, param, "message content must be a string or an array of blocks", "messages.content");
        }

        if (role == "assistant") {
            std::string text;
            Json calls = Json::array();
            for (const Json& block : content) {
                if (!block.is_object() || !block.contains("type") || !block["type"].is_string()) {
                    return fail(error, param, "each content block requires a string 'type'", "messages.content");
                }
                const std::string type = block["type"].get<std::string>();
                if (type == "text") {
                    if (!block.contains("text") || !block["text"].is_string()) {
                        return fail(error, param, "text blocks must contain a string 'text'", "messages.content");
                    }
                    text += block["text"].get<std::string>();
                } else if (type == "thinking" || type == "redacted_thinking") {
                    // Reasoning is not part of the user-visible chat transcript sent to the engine.
                    continue;
                } else if (type == "tool_use") {
                    if (!block.contains("id") || !block["id"].is_string() ||
                        !block.contains("name") || !block["name"].is_string()) {
                        return fail(error, param, "tool_use blocks require string 'id' and 'name'", "messages.content");
                    }
                    Json input = block.value("input", Json::object());
                    if (!input.is_object()) {
                        return fail(error, param, "tool_use input must be a JSON object", "messages.content");
                    }
                    calls.push_back({
                        {"id", block["id"]},
                        {"type", "function"},
                        {"function", {
                            {"name", block["name"]},
                            {"arguments", input.dump()}
                        }}
                    });
                } else {
                    return fail(error, param, "unsupported assistant content block type '" + type + "'", "messages.content");
                }
            }
            Json converted = {{"role", "assistant"}, {"content", text.empty() ? Json(nullptr) : Json(text)}};
            if (!calls.empty()) converted["tool_calls"] = std::move(calls);
            chat_request["messages"].push_back(std::move(converted));
            continue;
        }

        Json user_parts = Json::array();
        auto flush_user = [&]() {
            if (user_parts.empty()) return;
            chat_request["messages"].push_back({{"role", "user"}, {"content", user_parts}});
            user_parts = Json::array();
        };

        for (const Json& block : content) {
            if (!block.is_object() || !block.contains("type") || !block["type"].is_string()) {
                return fail(error, param, "each content block requires a string 'type'", "messages.content");
            }
            const std::string type = block["type"].get<std::string>();
            if (type == "text") {
                if (!append_text_block(block, user_parts, error, param)) return false;
            } else if (type == "image") {
                if (!append_image_block(block, user_parts, error, param)) return false;
            } else if (type == "tool_result") {
                if (!block.contains("tool_use_id") || !block["tool_use_id"].is_string()) {
                    return fail(error, param, "tool_result requires a string 'tool_use_id'", "messages.content");
                }
                flush_user();
                std::string result;
                if (!tool_result_to_text(block.value("content", Json()), result, error, param)) return false;
                chat_request["messages"].push_back({
                    {"role", "tool"},
                    {"tool_call_id", block["tool_use_id"]},
                    {"content", result}
                });
            } else {
                return fail(error, param, "unsupported user content block type '" + type + "'", "messages.content");
            }
        }
        flush_user();
        if (content.empty()) chat_request["messages"].push_back({{"role", "user"}, {"content", ""}});
    }
    if (request["messages"].empty()) {
        return fail(error, param, "'messages' must contain at least one message", "messages");
    }
    if (!inline_system_texts.empty()) {
        std::string extra_system;
        for (const std::string& text : inline_system_texts) {
            if (!extra_system.empty()) extra_system += "\n";
            extra_system += text;
        }
        Json& messages = chat_request["messages"];
        if (!messages.empty() && messages[0].value("role", "") == "system") {
            Json& system_content = messages[0]["content"];
            if (system_content.is_string() && !system_content.get_ref<const std::string&>().empty()) {
                system_content = system_content.get<std::string>() + "\n" + extra_system;
            } else {
                system_content = std::move(extra_system);
            }
        } else {
            messages.insert(messages.begin(), Json{{"role", "system"}, {"content", std::move(extra_system)}});
        }
    }
    if ((chat_request.value("tool_choice", Json("auto")) == "required") && !chat_request.contains("tools")) {
        return fail(error, param, "tool_choice type 'any' requires at least one tool", "tool_choice");
    }

    if (chat_request.value("stream", false)) {
        chat_request["stream_options"] = {{"include_usage", true}};
    }
    return true;
}

Json from_chat_response(const Json& response, const Json& request) {
    if (!response.is_object()) return make_error("Chat Completions backend returned an invalid response", "api_error");
    if (response.contains("error")) return anthropic_error(response["error"]);
    if (!response.contains("choices") || !response["choices"].is_array() || response["choices"].empty()) {
        return make_error("Chat Completions backend returned no choices", "api_error");
    }

    const Json& choice = response["choices"][0];
    const Json message = choice.value("message", Json::object());
    Json content = Json::array();
    if (message.contains("content") && message["content"].is_string()) {
        const std::string text = message["content"].get<std::string>();
        if (!text.empty()) content.push_back({{"type", "text"}, {"text", text}});
    }
    if (message.contains("tool_calls") && message["tool_calls"].is_array()) {
        for (const Json& call : message["tool_calls"]) {
            Json input;
            std::string error;
            if (!parse_tool_arguments(call, input, error)) return make_error(error, "api_error");
            const Json& function = call["function"];
            content.push_back({
                {"type", "tool_use"},
                {"id", make_id("toolu_")},
                {"name", function.value("name", "")},
                {"input", std::move(input)}
            });
        }
    }

    const Json usage = response.value("usage", Json::object());
    const int cached = usage.contains("prompt_tokens_details") &&
                       usage["prompt_tokens_details"].is_object()
        ? usage["prompt_tokens_details"].value("cached_tokens", 0) : 0;
    Json anthropic_usage = {
        {"input_tokens", usage.value("prompt_tokens", 0)},
        {"output_tokens", usage.value("completion_tokens", 0)}
    };
    if (cached > 0) {
        anthropic_usage["cache_read_input_tokens"] = cached;
        anthropic_usage["cache_creation_input_tokens"] = 0;
    }
    const std::string model = response.value("model",
        request.value("model", std::string("unknown")));
    return {
        {"id", make_id("msg_")},
        {"type", "message"},
        {"role", "assistant"},
        {"model", model},
        {"content", std::move(content)},
        {"stop_reason", anthropic_stop_reason(choice.value("finish_reason", "stop"))},
        {"stop_sequence", nullptr},
        {"usage", std::move(anthropic_usage)}
    };
}

StreamEncoder::StreamEncoder(std::string model, StreamCallback callback)
    : model_(std::move(model)), callback_(std::move(callback)), message_id_(make_id("msg_")) {}

void StreamEncoder::emit(const std::string& event, Json data, bool is_final) {
    data["type"] = event;
    callback_(Json("event: " + event + "\ndata: " +
                   data.dump(-1, ' ', false, Json::error_handler_t::replace) + "\n\n"),
              is_final);
}

void StreamEncoder::start() {
    if (started_) return;
    started_ = true;
    emit("message_start", {
        {"message", {
            {"id", message_id_},
            {"type", "message"},
            {"role", "assistant"},
            {"model", model_},
            {"content", Json::array()},
            {"stop_reason", nullptr},
            {"stop_sequence", nullptr},
            {"usage", {{"input_tokens", 0}, {"output_tokens", 1}}}
        }}
    });
}

StreamEncoder::ToolState& StreamEncoder::ensure_tool(const Json& delta) {
    const int index = delta.value("index", 0);
    ToolState& state = tools_[index];
    const Json function = delta.value("function", Json::object());
    if (function.is_object()) {
        if (function.contains("name") && function["name"].is_string()) {
            state.name = function["name"].get<std::string>();
        }
        if (function.contains("arguments") && function["arguments"].is_string()) {
            state.arguments += function["arguments"].get<std::string>();
        }
    }
    if (!state.started && !state.name.empty()) {
        start();
        state.started = true;
        state.block_index = next_block_index_++;
        state.id = make_id("toolu_");
        content_.push_back({
            {"type", "tool_use"},
            {"id", state.id},
            {"name", state.name},
            {"input", Json::object()}
        });
        emit("content_block_start", {
            {"index", state.block_index},
            {"content_block", {
                {"type", "tool_use"},
                {"id", state.id},
                {"name", state.name},
                {"input", Json::object()}
            }}
        });
        if (!state.arguments.empty()) {
            emit("content_block_delta", {
                {"index", state.block_index},
                {"delta", {{"type", "input_json_delta"}, {"partial_json", state.arguments}}}
            });
        }
    } else if (state.started && function.is_object() &&
               function.contains("arguments") && function["arguments"].is_string()) {
        const std::string& part = function["arguments"].get_ref<const std::string&>();
        if (!part.empty()) {
            emit("content_block_delta", {
                {"index", state.block_index},
                {"delta", {{"type", "input_json_delta"}, {"partial_json", part}}}
            });
        }
    }
    return state;
}

void StreamEncoder::close_blocks() {
    for (int index = 0; index < next_block_index_; ++index) {
        if (text_block_index_ == index) {
            content_[index]["text"] = text_;
            emit("content_block_stop", {{"index", index}});
            continue;
        }
        for (auto& [_, tool] : tools_) {
            if (tool.started && tool.block_index == index) {
                Json input;
                try {
                    input = Json::parse(tool.arguments.empty() ? "{}" : tool.arguments);
                } catch (const std::exception&) {
                    input = Json::object();
                }
                if (!input.is_object()) input = Json::object();
                content_[index]["input"] = std::move(input);
                emit("content_block_stop", {{"index", index}});
                break;
            }
        }
    }
}

void StreamEncoder::finish() {
    if (finished_) return;
    start();
    close_blocks();
    const std::string stop_reason = anthropic_stop_reason(finish_reason_);
    emit("message_delta", {
        {"delta", {{"stop_reason", stop_reason}, {"stop_sequence", nullptr}}},
        {"usage", {{"output_tokens", output_tokens_}}}
    });
    emit("message_stop", Json::object(), true);
    finished_ = true;
}

void StreamEncoder::fail(const Json& error) {
    if (finished_) return;
    const Json mapped = anthropic_error(error);
    emit("error", {{"error", mapped.value("error", Json::object())}}, true);
    finished_ = true;
}

void StreamEncoder::on_chat_error(const Json& error) {
    if (error.is_object() && error.contains("error")) fail(error["error"]);
    else fail(error);
}

void StreamEncoder::on_chat_chunk(const Json& data, bool is_final) {
    if (finished_) return;
    if (data.is_object()) {
        if (data.contains("error")) fail(data["error"]);
        else if (is_final) finish();
        return;
    }
    if (!data.is_string()) {
        if (is_final) finish();
        return;
    }
    const std::string raw = data.get<std::string>();
    const size_t marker = raw.find("data:");
    if (marker == std::string::npos) {
        if (is_final) finish();
        return;
    }
    size_t begin = marker + 5;
    while (begin < raw.size() && (raw[begin] == ' ' || raw[begin] == '\t')) ++begin;
    const size_t end = raw.find_first_of("\r\n", begin);
    const std::string payload = raw.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (payload == "[DONE]") {
        finish();
        return;
    }

    Json chunk;
    try {
        chunk = Json::parse(payload);
    } catch (const std::exception& e) {
        fail({{"message", std::string("Invalid Chat Completions stream chunk: ") + e.what()},
              {"type", "api_error"}});
        return;
    }
    if (chunk.contains("error")) {
        fail(chunk["error"]);
        return;
    }
    if (chunk.contains("usage") && chunk["usage"].is_object()) {
        output_tokens_ = chunk["usage"].value("completion_tokens", output_tokens_);
    }
    if (chunk.contains("choices") && chunk["choices"].is_array() && !chunk["choices"].empty()) {
        const Json& choice = chunk["choices"][0];
        if (choice.contains("finish_reason") && choice["finish_reason"].is_string()) {
            finish_reason_ = choice["finish_reason"].get<std::string>();
        }
        const Json delta = choice.value("delta", Json::object());
        if (delta.contains("content") && delta["content"].is_string()) {
            const std::string text = delta["content"].get<std::string>();
            if (!text.empty()) {
                start();
                if (text_block_index_ < 0) {
                    text_block_index_ = next_block_index_++;
                    content_.push_back({{"type", "text"}, {"text", ""}});
                    emit("content_block_start", {
                        {"index", text_block_index_},
                        {"content_block", {{"type", "text"}, {"text", ""}}}
                    });
                }
                text_ += text;
                emit("content_block_delta", {
                    {"index", text_block_index_},
                    {"delta", {{"type", "text_delta"}, {"text", text}}}
                });
            }
        }
        if (delta.contains("tool_calls") && delta["tool_calls"].is_array()) {
            for (const Json& call : delta["tool_calls"]) {
                if (call.is_object()) ensure_tool(call);
            }
        }
    }
    if (is_final) finish();
}

} // namespace anthropic_messages
