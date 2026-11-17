cmake_minimum_required(VERSION 3.22)
project(anthropic_messages_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_anthropic_messages
    test_anthropic_messages.cpp
    "${FLM_SOURCE_DIR}/server/anthropic_messages.cpp")
target_include_directories(test_anthropic_messages PRIVATE
    "${FLM_SOURCE_DIR}/include"
    "${FLM_SOURCE_DIR}/server")

enable_testing()
add_test(NAME test_anthropic_messages COMMAND test_anthropic_messages)
