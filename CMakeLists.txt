cmake_minimum_required(VERSION 3.0)
project(LumenEngine)

option(GLFW_BUILD_DOCS OFF)
option(GLFW_BUILD_EXAMPLES OFF)
option(GLFW_BUILD_TESTS OFF)

add_subdirectory(api/glfw)

if(NOT WIN32)
	set(GLAD_LIBRARIES dl)
endif()

include_directories(src/renderer/
                    src/resources/
                    src/cuda/
                    api/glad/include/
                    api/glfw/include/
                    api/glm/
                    api/stb/
                    api/imgui/
                    api/cuda/)

file(GLOB API_SOURCES api/glad/src/glad.c
                      api/imgui/*.cpp)

file(GLOB PROJECT_HEADERS src/renderer/*.h
                          src/resources/*.h
                          src/cuda/*.h)

file(GLOB PROJECT_SOURCES src/*.cpp
                          src/renderer/*.cpp
                          src/resources/*.cpp
                          src/cuda/*.cpp
                          src/cuda/*.cu)

file(GLOB PROJECT_SHADERS resources/shaders/*.glsl
                          resources/shaders/*.frag
                          resources/shaders/*.vert)

file(GLOB PROJECT_CONFIGS CMakeLists.txt
                          README.md
                         .gitattributes
                         .gitignore
                         .gitmodules)

source_group("Headers" FILES ${PROJECT_HEADERS})
source_group("Shaders" FILES ${PROJECT_SHADERS})
source_group("Sources" FILES ${PROJECT_SOURCES})
source_group("API" FILES ${API_SOURCES})


# CUDA specific options
find_package(CUDA REQUIRED)

set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CMAKE_CXX_STANDARD 11)

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    --std=c++11 -gencode arch=compute_52,code=sm_52 --compiler-options -fno-strict-aliasing --ptxas-options=-v -maxrregcount 20 --use_fast_math)

cuda_add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${PROJECT_HEADERS}
                    ${PROJECT_SHADERS} ${PROJECT_CONFIGS}
                    ${API_SOURCES})


target_link_libraries(${PROJECT_NAME} glfw
                      ${GLFW_LIBRARIES} ${GLAD_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
