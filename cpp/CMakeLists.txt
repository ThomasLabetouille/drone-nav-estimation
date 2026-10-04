cmake_minimum_required(VERSION 3.16)
project(navcpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# No -ffast-math: the port must reproduce the Python reference to rounding.
set(WARNINGS -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror)

add_library(navcpp src/eskf.cpp)
target_include_directories(navcpp PUBLIC include)
target_link_libraries(navcpp PUBLIC Eigen3::Eigen)
target_compile_options(navcpp PRIVATE ${WARNINGS})

add_executable(nav_replay src/replay_main.cpp)
target_link_libraries(nav_replay PRIVATE navcpp)
target_compile_options(nav_replay PRIVATE ${WARNINGS})

# The test target compiles the filter itself, with Eigen's heap guard on and
# assertions enabled even in Release.
add_executable(test_eskf tests/test_eskf.cpp src/eskf.cpp)
target_include_directories(test_eskf PRIVATE include)
target_link_libraries(test_eskf PRIVATE Eigen3::Eigen)
target_compile_definitions(test_eskf PRIVATE EIGEN_RUNTIME_NO_MALLOC)
target_compile_options(test_eskf PRIVATE ${WARNINGS} -UNDEBUG)

enable_testing()
add_test(NAME test_eskf COMMAND test_eskf)
