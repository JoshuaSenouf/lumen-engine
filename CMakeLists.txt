cmake_minimum_required(VERSION 3.0)
project(LumenEngine)

option(GLFW_BUILD_DOCS OFF)
option(GLFW_BUILD_EXAMPLES OFF)
option(GLFW_BUILD_TESTS OFF)
add_subdirectory(api/glfw)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -std=c++11")
    if(NOT WIN32)
        set(GLAD_LIBRARIES dl)
    endif()
endif()

include_directories(src/renderer/
                    src/resources/
                    src/opencl/
                    api/glad/include/
                    api/glfw/include/
                    api/glm/
                    api/stb/
                    api/imgui/)

file(GLOB API_SOURCES api/glad/src/glad.c
                      api/imgui/*.cpp)

file(GLOB PROJECT_HEADERS src/renderer/*.h
                          src/resources/*.h
                          src/opencl/*.h)

file(GLOB PROJECT_SOURCES src/*.cpp
                          src/renderer/*.cpp
                          src/resources/*.cpp
                          src/opencl/*.cpp
                          src/opencl/*.cl)

file(GLOB PROJECT_SHADERS resources/shaders/*.glsl
                          resources/shaders/*.frag
                          resources/shaders/*.vert)

file(GLOB PROJECT_CONFIGS CMakeLists.txt
                          Readme.md
                         .gitattributes
                         .gitignore
                         .gitmodules)

source_group("Headers" FILES ${PROJECT_HEADERS})
source_group("Shaders" FILES ${PROJECT_SHADERS})
source_group("Sources" FILES ${PROJECT_SOURCES})
source_group("API" FILES ${API_SOURCES})

find_package(OpenCL REQUIRED)

add_definitions(-DGLFW_INCLUDE_NONE
                -DPROJECT_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\")
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS}
                               ${PROJECT_SHADERS} ${PROJECT_CONFIGS}
                               ${API_SOURCES})
target_link_libraries(${PROJECT_NAME} glfw
                      ${GLFW_LIBRARIES} ${GLAD_LIBRARIES} OpenCL::OpenCL)
set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
