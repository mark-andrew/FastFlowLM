cmake_minimum_required(VERSION 3.22)
project(openai_responses_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(FLM_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")

add_executable(test_openai_responses
    test_openai_responses.cpp
    "${FLM_SOURCE_DIR}/server/openai_responses.cpp")
target_include_directories(test_openai_responses PRIVATE
    "${FLM_SOURCE_DIR}/include"
    "${FLM_SOURCE_DIR}/server")

enable_testing()
add_test(NAME test_openai_responses COMMAND test_openai_responses)
