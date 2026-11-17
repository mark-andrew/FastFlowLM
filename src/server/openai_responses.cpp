/*!
 *  Copyright (c) 2026 Advanced Micro Devices, Inc.
 * \file openai_responses.cpp
 * \brief Stateless OpenAI Responses API adapters for FastFlowLM chat completions
 */
#include "openai_responses.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <utility>

namespace openai_responses {

namespace {

std::string make_id(const char* prefix) {
    static const char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    thread_local std::mt19937 generator{std::random_device{}()};
    std::uniform_int_distribution<size_t> pick(0, sizeof(alphabet) - 2);

    std::string id(prefix);
    for (int i = 0; i < 24; ++i) {
        id += alphabet[pick(generator)];
    }
    return id;
}

long long unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
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

bool response_content_to_chat(const Json& content, Json& converted,
                              std::string& error, std::string& param) {
    if (content.is_string()) {
        converted = content;
        return true;
    }
    if (!content.is_array()) {
        return fail(error, param, "message content must be a string or an array of content blocks", "input");
    }

    Json parts = Json::array();
    for (const Json& block : content) {
        if (!block.is_object() || !block.contains("type") || !block["type"].is_string()) {
            return fail(error, param, "each content block must have a string 'type'", "input");
        }
        const std::string type = block["type"].get<std::string>();
        if (type == "input_text" || type == "output_text" || type == "text") {
            if (!block.contains("text") || !block["text"].is_string()) {
                return fail(error, param, "text content blocks must contain a string 'text'", "input");
            }
            parts.push_back({{"type", "text"}, {"text", block["text"]}});
        } else if (type == "input_image") {
            if (!block.contains("image_url") || !block["image_url"].is_string()) {
                return fail(error, param,
                            "Responses input_image currently requires an inline image_url data URL",
                            "input");
            }
            parts.push_back({
                {"type", "image_url"},
                {"image_url", {{"url", block["image_url"]}}}
            });
        } else {
            return fail(error, param, "unsupported Responses input content block type '" + type + "'", "input");
        }
    }
    converted = std::move(parts);
    return true;
}

std::string response_output_to_string(const Json& output) {
    if (output.is_string()) return output.get<std::string>();
    if (output.is_array()) {
        std::string text;
        for (const Json& block : output) {
            if (block.is_object() && block.value("type", "") == "output_text" &&
                block.contains("text") && block["text"].is_string()) {
                text += block["text"].get<std::string>();
            } else if (block.is_object() && block.value("type", "") == "text" &&
                       block.contains("text") && block["text"].is_string()) {
                text += block["text"].get<std::string>();
            }
        }
        if (!text.empty()) return text;
    }
    return output.dump();
}

Json responses_usage(const Json& chat_usage) {
    const int input_tokens = chat_usage.value("prompt_tokens", 0);
    const int output_tokens = chat_usage.value("completion_tokens", 0);
    Json input_details = {{"cached_tokens", 0}};
    if (chat_usage.contains("prompt_tokens_details") &&
        chat_usage["prompt_tokens_details"].is_object()) {
        input_details["cached_tokens"] =
            chat_usage["prompt_tokens_details"].value("cached_tokens", 0);
    }
    return {
        {"input_tokens", input_tokens},
        {"input_tokens_details", input_details},
        {"output_tokens", output_tokens},
        {"total_tokens", chat_usage.value("total_tokens", input_tokens + output_tokens)}
    };
}

Json response_base(const std::string& id, const std::string& model, long long created_at,
                   const Json& request, const Json& output, const Json& usage,
                   const std::string& status, const std::string& finish_reason) {
    Json response = {
        {"id", id},
        {"object", "response"},
        {"created_at", created_at},
        {"model", model},
        {"status", status},
        {"output", output},
        {"usage", usage},
        {"error", nullptr},
        {"incomplete_details", nullptr},
        {"previous_response_id", nullptr},
        {"store", false},
        {"background", false},
        {"parallel_tool_calls", has_value(request, "parallel_tool_calls")
                                    ? request["parallel_tool_calls"] : Json(true)},
        {"tool_choice", has_value(request, "tool_choice")
                            ? request["tool_choice"] : Json("auto")},
        {"tools", has_value(request, "tools") ? request["tools"] : Json::array()},
        {"text", has_value(request, "text")
                     ? request["text"] : Json{{"format", {{"type", "text"}}}}}
    };
    if (has_value(request, "instructions")) response["instructions"] = request["instructions"];
    if (has_value(request, "metadata")) response["metadata"] = request["metadata"];
    if (has_value(request, "max_output_tokens")) response["max_output_tokens"] = request["max_output_tokens"];
    if (has_value(request, "reasoning")) response["reasoning"] = request["reasoning"];
    if (finish_reason == "length") {
        response["status"] = "incomplete";
        response["incomplete_details"] = {{"reason", "max_output_tokens"}};
    }
    return response;
}

} // namespace

bool to_chat_request(const Json& request, const std::string& fallback_model,
                     Json& chat_request, std::string& error, std::string& param) {
    error.clear();
    param.clear();
    if (!request.is_object()) {
        return fail(error, param, "request body must be a JSON object", "");
    }
    if (!has_value(request, "input")) {
        return fail(error, param, "'input' is required", "input");
    }
    if (has_value(request, "model") && !request["model"].is_string()) {
        return fail(error, param, "'model' must be a string", "model");
    }
    if (has_value(request, "stream") && !request["stream"].is_boolean()) {
        return fail(error, param, "'stream' must be a boolean", "stream");
    }
    if (has_value(request, "parallel_tool_calls") &&
        !request["parallel_tool_calls"].is_boolean()) {
        return fail(error, param, "'parallel_tool_calls' must be a boolean", "parallel_tool_calls");
    }
    if (has_value(request, "previous_response_id")) {
        return fail(error, param,
                    "previous_response_id is not supported; send the full input history",
                    "previous_response_id");
    }
    if (has_value(request, "conversation")) {
        return fail(error, param,
                    "stateful conversations are not supported; send the full input history",
                    "conversation");
    }
    if (has_value(request, "background") && request["background"] != false) {
        return fail(error, param, "background responses are not supported", "background");
    }

    const Json& input = request["input"];
    Json messages = Json::array();
    std::string instructions;
    if (has_value(request, "instructions")) {
        if (request["instructions"].is_string()) {
            instructions = request["instructions"].get<std::string>();
        } else if (request["instructions"].is_array()) {
            Json instruction_blocks;
            if (!response_content_to_chat(request["instructions"], instruction_blocks, error, param)) {
                param = "instructions";
                return false;
            }
            for (const Json& block : instruction_blocks) {
                if (block.contains("text") && block["text"].is_string()) {
                    if (!instructions.empty()) instructions += "\n";
                    instructions += block["text"].get<std::string>();
                }
            }
        } else {
            return fail(error, param, "'instructions' must be a string or an array of text blocks", "instructions");
        }
    }

    Json conversation = Json::array();
    Json pending_calls = Json::array();
    auto flush_calls = [&]() {
        if (!pending_calls.empty()) {
            conversation.push_back({
                {"role", "assistant"},
                {"content", nullptr},
                {"tool_calls", pending_calls}
            });
            pending_calls = Json::array();
        }
    };

    auto append_message = [&](const Json& item, const std::string& role) -> bool {
        if (!item.contains("content")) {
            return fail(error, param, "message is missing 'content'", "input");
        }
        Json content;
        if (!response_content_to_chat(item["content"], content, error, param)) return false;
        conversation.push_back({{"role", role}, {"content", std::move(content)}});
        return true;
    };

    if (input.is_string()) {
        conversation.push_back({{"role", "user"}, {"content", input}});
    } else if (input.is_array()) {
        for (const Json& item : input) {
            if (!item.is_object()) {
                return fail(error, param, "input array items must be objects", "input");
            }
            if (has_value(item, "type") && !item["type"].is_string()) {
                return fail(error, param, "input item type must be a string", "input");
            }
            const std::string type = item.value("type", "");
            if (type == "reasoning") {
                // Reasoning items are not needed to rebuild FLM's native chat history.
                continue;
            }
            if (type == "function_call") {
                if (!item.contains("name") || !item["name"].is_string() ||
                    !item.contains("call_id") || !item["call_id"].is_string()) {
                    return fail(error, param, "function_call items require string name and call_id", "input");
                }
                std::string arguments;
                if (item.contains("arguments") && item["arguments"].is_string()) {
                    arguments = item["arguments"].get<std::string>();
                } else if (item.contains("arguments")) {
                    arguments = item["arguments"].dump();
                } else {
                    arguments = "{}";
                }
                pending_calls.push_back({
                    {"id", item["call_id"]},
                    {"type", "function"},
                    {"function", {
                        {"name", item["name"]},
                        {"arguments", arguments}
                    }}
                });
                continue;
            }
            if (type == "function_call_output") {
                flush_calls();
                if (!item.contains("call_id") || !item["call_id"].is_string() ||
                    !item.contains("output")) {
                    return fail(error, param, "function_call_output items require call_id and output", "input");
                }
                conversation.push_back({
                    {"role", "tool"},
                    {"tool_call_id", item["call_id"]},
                    {"content", response_output_to_string(item["output"])}
                });
                continue;
            }

            flush_calls();
            const bool is_message = type.empty() || type == "message";
            if (!is_message) {
                return fail(error, param, "unsupported Responses input item type '" + type + "'", "input");
            }
            if (has_value(item, "role") && !item["role"].is_string()) {
                return fail(error, param, "message role must be a string", "input");
            }
            const std::string role = item.value("role", "");
            if (role == "system" || role == "developer") {
                if (!item.contains("content")) {
                    return fail(error, param, "instruction message is missing 'content'", "input");
                }
                Json content;
                if (!response_content_to_chat(item["content"], content, error, param)) return false;
                std::string text;
                if (content.is_string()) {
                    text = content.get<std::string>();
                } else {
                    for (const Json& block : content) {
                        if (block.contains("text") && block["text"].is_string()) {
                            if (!text.empty()) text += "\n";
                            text += block["text"].get<std::string>();
                        }
                    }
                }
                if (!instructions.empty() && !text.empty()) instructions += "\n";
                instructions += text;
            } else if (role == "user" || role == "assistant") {
                if (!append_message(item, role)) return false;
            } else {
                return fail(error, param, "message role must be system, developer, user or assistant", "input");
            }
        }
        flush_calls();
    } else {
        return fail(error, param, "'input' must be a string or an array", "input");
    }

    messages = Json::array();
    if (!instructions.empty()) {
        messages.push_back({{"role", "system"}, {"content", instructions}});
    }
    for (auto& message : conversation) messages.push_back(std::move(message));

    chat_request = {
        {"messages", std::move(messages)},
        {"stream", has_value(request, "stream") ? request["stream"] : Json(false)}
    };
    if (has_value(request, "model")) {
        chat_request["model"] = request["model"];
    } else if (!fallback_model.empty()) {
        chat_request["model"] = fallback_model;
    }

    if (has_value(request, "tools")) {
        if (!request["tools"].is_array()) {
            return fail(error, param, "'tools' must be an array", "tools");
        }
        Json tools = Json::array();
        for (const Json& tool : request["tools"]) {
            if (!tool.is_object() || !tool.contains("type") || !tool["type"].is_string() ||
                tool["type"] != "function") {
                const std::string type = tool.is_object() && tool.contains("type") && tool["type"].is_string()
                                             ? tool["type"].get<std::string>() : "invalid";
                return fail(error, param,
                            "unsupported Responses tool type '" + type + "'; only function tools are supported",
                            "tools");
            }
            if (!tool.contains("name") || !tool["name"].is_string()) {
                return fail(error, param, "function tools require a string name", "tools");
            }
            if (has_value(tool, "parameters") && !tool["parameters"].is_object()) {
                return fail(error, param, "function tool parameters must be a JSON schema object", "tools");
            }
            if (has_value(tool, "strict") && !tool["strict"].is_boolean()) {
                return fail(error, param, "function tool strict must be a boolean", "tools");
            }
            Json function = {
                {"name", tool["name"]},
                {"parameters", tool.value("parameters", Json::object())}
            };
            if (has_value(tool, "description")) function["description"] = tool["description"];
            if (has_value(tool, "strict")) function["strict"] = tool["strict"];
            tools.push_back({{"type", "function"}, {"function", function}});
        }
        chat_request["tools"] = std::move(tools);
    }

    if (has_value(request, "tool_choice")) {
        const Json& choice = request["tool_choice"];
        if (choice.is_object() && choice.contains("type") && choice["type"].is_string() &&
            choice["type"] == "function") {
            if (!choice.contains("name") || !choice["name"].is_string()) {
                return fail(error, param, "function tool_choice requires a string name", "tool_choice");
            }
            chat_request["tool_choice"] = {
                {"type", "function"},
                {"function", {{"name", choice["name"]}}}
            };
        } else {
            chat_request["tool_choice"] = choice;
        }
    }

    if (has_value(request, "max_output_tokens")) {
        if (!request["max_output_tokens"].is_number_integer() ||
            request["max_output_tokens"].get<long long>() <= 0 ||
            request["max_output_tokens"].get<long long>() > std::numeric_limits<int>::max()) {
            return fail(error, param, "'max_output_tokens' must be a positive integer", "max_output_tokens");
        }
        chat_request["max_completion_tokens"] = request["max_output_tokens"];
    }
    if (has_value(request, "reasoning")) {
        if (!request["reasoning"].is_object()) {
            return fail(error, param, "'reasoning' must be an object", "reasoning");
        }
        if (has_value(request["reasoning"], "effort")) {
            if (!request["reasoning"]["effort"].is_string()) {
                return fail(error, param, "'reasoning.effort' must be a string", "reasoning.effort");
            }
            chat_request["reasoning_effort"] = request["reasoning"]["effort"];
        }
    }
    for (const char* field : {"parallel_tool_calls", "temperature", "top_p", "presence_penalty",
                              "frequency_penalty", "stop", "seed", "backend", "think",
                              "ignore_eos", "min_tokens", "repeat_last_n", "options"}) {
        if (has_value(request, field)) chat_request[field] = request[field];
    }
    if (has_value(request, "text") && !request["text"].is_object()) {
        return fail(error, param, "'text' must be an object", "text");
    }
    if (has_value(request, "text") && has_value(request["text"], "format")) {
        const Json& format = request["text"]["format"];
        if (!format.is_object()) {
            return fail(error, param, "'text.format' must be an object", "text.format");
        }
        if (has_value(format, "type") && !format["type"].is_string()) {
            return fail(error, param, "'text.format.type' must be a string", "text.format.type");
        }
        const std::string type = format.value("type", "");
        if (type == "json_object") {
            chat_request["response_format"] = {{"type", "json_object"}};
        } else if (type == "json_schema") {
            if (!format.contains("schema") || !format["schema"].is_object()) {
                return fail(error, param, "text.format json_schema requires an object schema", "text.format");
            }
            Json schema = {
                {"name", format.value("name", "response")},
                {"schema", format["schema"]}
            };
            if (has_value(format, "strict")) schema["strict"] = format["strict"];
            chat_request["response_format"] = {
                {"type", "json_schema"},
                {"json_schema", std::move(schema)}
            };
        } else if (!type.empty() && type != "text") {
            return fail(error, param, "unsupported Responses text format '" + type + "'", "text.format");
        }
    }

    // Responses streams carry usage in the final event; request it from the
    // shared Chat Completions engine so the adapter can preserve those counts.
    if (chat_request.value("stream", false)) {
        chat_request["stream_options"] = {{"include_usage", true}};
    }
    return true;
}

Json from_chat_response(const Json& chat_response, const Json& request) {
    if (!chat_response.is_object() || chat_response.contains("error")) return chat_response;
    const Json& choices = chat_response.at("choices");
    if (!choices.is_array() || choices.empty()) {
        return {
            {"error", {
                {"message", "Chat Completions backend returned no choices"},
                {"type", "server_error"},
                {"code", 500}
            }}
        };
    }

    const Json& choice = choices[0];
    const Json message = choice.value("message", Json::object());
    const std::string finish_reason = choice.value("finish_reason", "stop");
    const bool incomplete = finish_reason == "length";
    const std::string response_id = make_id("resp_");
    const long long created_at = unix_seconds();
    Json output = Json::array();
    std::string output_text;

    if (message.contains("content") && message["content"].is_string()) {
        output_text = message["content"].get<std::string>();
        if (!output_text.empty()) {
            output.push_back({
                {"id", make_id("msg_")},
                {"type", "message"},
                {"status", incomplete ? "incomplete" : "completed"},
                {"role", "assistant"},
                {"content", Json::array({
                    {
                        {"type", "output_text"},
                        {"text", output_text},
                        {"annotations", Json::array()},
                        {"logprobs", Json::array()}
                    }
                })}
            });
        }
    }

    if (message.contains("tool_calls") && message["tool_calls"].is_array()) {
        for (const Json& call : message["tool_calls"]) {
            if (!call.is_object() || !call.contains("function") || !call["function"].is_object()) continue;
            const Json& function = call["function"];
            const std::string call_id = call.value("id", make_id("call_"));
            output.push_back({
                {"id", make_id("fc_")},
                {"type", "function_call"},
                {"status", incomplete ? "incomplete" : "completed"},
                {"call_id", call_id},
                {"name", function.value("name", "")},
                {"arguments", function.value("arguments", std::string("{}"))}
            });
        }
    }

    Json response = response_base(response_id, chat_response.value("model", ""),
                                  chat_response.value("created", created_at),
                                  request, output,
                                  responses_usage(chat_response.value("usage", Json::object())),
                                  incomplete ? "incomplete" : "completed", finish_reason);
    response["completed_at"] = unix_seconds();
    return response;
}

StreamEncoder::StreamEncoder(std::string model, Json request, StreamCallback callback)
    : model_(std::move(model)), request_(std::move(request)), callback_(std::move(callback)),
      response_id_(make_id("resp_")), created_at_(unix_seconds()) {}

Json StreamEncoder::response_object(const std::string& status) const {
    Json response = response_base(response_id_, model_, created_at_, request_, output_, usage_, status, finish_reason_);
    if (status == "completed" || status == "incomplete" || status == "failed") {
        response["completed_at"] = unix_seconds();
    }
    if (status == "failed") {
        response["error"] = failure_;
    }
    return response;
}

void StreamEncoder::emit(const std::string& event, Json data, bool is_final) {
    data["type"] = event;
    data["sequence_number"] = sequence_number_++;
    callback_(Json("event: " + event + "\ndata: " + data.dump(-1, ' ', false, Json::error_handler_t::replace) + "\n\n"),
              is_final);
}

void StreamEncoder::start() {
    if (started_) return;
    started_ = true;
    Json response = response_object("in_progress");
    emit("response.created", {{"response", response}});
    emit("response.in_progress", {{"response", response}});
}

void StreamEncoder::ensure_message() {
    if (message_output_index_ >= 0) return;
    start();
    message_output_index_ = static_cast<int>(output_.size());
    message_id_ = make_id("msg_");
    Json item = {
        {"id", message_id_},
        {"type", "message"},
        {"status", "in_progress"},
        {"role", "assistant"},
        {"content", Json::array()}
    };
    output_.push_back(item);
    emit("response.output_item.added", {
        {"output_index", message_output_index_},
        {"item", item}
    });
    Json part = {
        {"type", "output_text"},
        {"text", ""},
        {"annotations", Json::array()},
        {"logprobs", Json::array()}
    };
    output_[message_output_index_]["content"].push_back(part);
    emit("response.content_part.added", {
        {"item_id", message_id_},
        {"output_index", message_output_index_},
        {"content_index", 0},
        {"part", part}
    });
}

void StreamEncoder::finish_message(bool complete) {
    if (message_output_index_ < 0) return;
    Json& item = output_[message_output_index_];
    item["status"] = complete ? "completed" : "incomplete";
    item["content"][0]["text"] = message_text_;
    emit("response.output_text.done", {
        {"item_id", message_id_},
        {"output_index", message_output_index_},
        {"content_index", 0},
        {"text", message_text_}
    });
    emit("response.content_part.done", {
        {"item_id", message_id_},
        {"output_index", message_output_index_},
        {"content_index", 0},
        {"part", item["content"][0]}
    });
    emit("response.output_item.done", {
        {"output_index", message_output_index_},
        {"item", item}
    });
}

StreamEncoder::ToolState* StreamEncoder::ensure_tool(const Json& delta) {
    const int index = delta.value("index", 0);
    auto it = tools_.find(index);
    const Json function = delta.value("function", Json::object());
    if (it == tools_.end()) {
        if (!function.is_object() || !function.contains("name") || !function["name"].is_string()) {
            return nullptr;
        }
        start();
        ToolState state;
        state.output_index = static_cast<int>(output_.size());
        state.item_id = make_id("fc_");
        state.call_id = delta.value("id", make_id("call_"));
        state.name = function["name"].get<std::string>();
        Json item = {
            {"id", state.item_id},
            {"type", "function_call"},
            {"status", "in_progress"},
            {"call_id", state.call_id},
            {"name", state.name},
            {"arguments", ""}
        };
        output_.push_back(item);
        it = tools_.emplace(index, std::move(state)).first;
        emit("response.output_item.added", {
            {"output_index", it->second.output_index},
            {"item", item}
        });
    }
    if (function.is_object()) {
        if (function.contains("name") && function["name"].is_string()) {
            it->second.name = function["name"].get<std::string>();
            output_[it->second.output_index]["name"] = it->second.name;
        }
        if (function.contains("arguments") && function["arguments"].is_string()) {
            const std::string part = function["arguments"].get<std::string>();
            if (!part.empty()) {
                it->second.arguments += part;
                output_[it->second.output_index]["arguments"] = it->second.arguments;
                emit("response.function_call_arguments.delta", {
                    {"item_id", it->second.item_id},
                    {"output_index", it->second.output_index},
                    {"delta", part}
                });
            }
        }
    }
    if (delta.contains("id") && delta["id"].is_string()) {
        it->second.call_id = delta["id"].get<std::string>();
        output_[it->second.output_index]["call_id"] = it->second.call_id;
    }
    return &it->second;
}

void StreamEncoder::finish_tools(bool complete) {
    for (auto& [_, tool] : tools_) {
        Json& item = output_[tool.output_index];
        item["status"] = complete ? "completed" : "incomplete";
        item["name"] = tool.name;
        item["call_id"] = tool.call_id;
        item["arguments"] = tool.arguments;
        emit("response.function_call_arguments.done", {
            {"item_id", tool.item_id},
            {"output_index", tool.output_index},
            {"arguments", tool.arguments}
        });
        emit("response.output_item.done", {
            {"output_index", tool.output_index},
            {"item", item}
        });
    }
}

void StreamEncoder::fail(const Json& error) {
    if (finished_) return;
    failure_ = error.is_object() && error.contains("error") ? error["error"] : error;
    if (!failure_.is_object()) failure_ = {{"message", failure_.dump()}, {"type", "server_error"}};
    if (!failure_.contains("message")) failure_["message"] = "Responses request failed";
    if (!failure_.contains("type")) failure_["type"] = "server_error";
    start();
    emit("response.failed", {{"response", response_object("failed")}}, true);
    finished_ = true;
}

void StreamEncoder::finish() {
    if (finished_) return;
    if (failed_) {
        fail(failure_);
        return;
    }
    start();
    const bool incomplete = finish_reason_ == "length";
    finish_message(!incomplete);
    finish_tools(!incomplete);
    const std::string status = incomplete ? "incomplete" : "completed";
    Json response = response_object(status);
    response["completed_at"] = unix_seconds();
    if (incomplete) {
        emit("response.incomplete", {{"response", response}}, true);
    } else {
        emit("response.completed", {{"response", response}}, true);
    }
    finished_ = true;
}

void StreamEncoder::on_chat_chunk(const Json& data, bool is_final) {
    if (finished_) return;
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
    size_t end = raw.find_first_of("\r\n", begin);
    std::string payload = raw.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (payload == "[DONE]") {
        finish();
        return;
    }
    Json chunk;
    try {
        chunk = Json::parse(payload);
    } catch (const std::exception& e) {
        fail({{"error", {{"message", std::string("Invalid Chat Completions stream chunk: ") + e.what()},
                         {"type", "server_error"}}}});
        return;
    }
    if (chunk.contains("error")) {
        failed_ = true;
        failure_ = chunk["error"];
        if (is_final) fail(failure_);
        return;
    }
    if (chunk.contains("usage") && chunk["usage"].is_object()) {
        usage_ = responses_usage(chunk["usage"]);
    }
    if (chunk.contains("choices") && chunk["choices"].is_array() && !chunk["choices"].empty()) {
        const Json& choice = chunk["choices"][0];
        if (choice.contains("finish_reason") && choice["finish_reason"].is_string()) {
            finish_reason_ = choice["finish_reason"].get<std::string>();
        }
        if (choice.contains("delta") && choice["delta"].is_object()) {
            const Json& delta = choice["delta"];
            if (delta.contains("content") && delta["content"].is_string()) {
                const std::string text = delta["content"].get<std::string>();
                if (!text.empty()) {
                    ensure_message();
                    message_text_ += text;
                    output_[message_output_index_]["content"][0]["text"] = message_text_;
                    emit("response.output_text.delta", {
                        {"item_id", message_id_},
                        {"output_index", message_output_index_},
                        {"content_index", 0},
                        {"delta", text}
                    });
                }
            }
            if (delta.contains("tool_calls") && delta["tool_calls"].is_array()) {
                for (const Json& call : delta["tool_calls"]) ensure_tool(call);
            }
        }
    }
    if (is_final) finish();
}

} // namespace openai_responses
