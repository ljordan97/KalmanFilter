
# This will load the CMAKE_TOOLCHAIN_FILE
cmake_policy(SET CMP0048 NEW)
project(AXV VERSION 0.0.0)

cmake_minimum_required(VERSION 3.22.0)

set(PROJECT_CMAKE_PATH ${CMAKE_SOURCE_DIR}/devOps/cmake)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_CMAKE_PATH})

set(INSTALL_BASE ${CMAKE_SOURCE_DIR}/install)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET_NAME kalman)

project(kalman)

#target_sources(${TARGET_NAME} PRIVATE
	#	src/main.cpp)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
add_executable(kalman src/main.cpp)
target_link_libraries(kalman Eigen3::Eigen)

install(TARGETS
	kalman)	


