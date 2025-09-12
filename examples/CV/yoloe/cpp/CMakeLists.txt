cmake_minimum_required(VERSION 3.10)


project(yoloe_demo)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE RELEASE)


file(GLOB SRCS "*.cpp")


find_package(OpenCV 4 REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()


#set(ONNXRUNTIME_INCLUDE_DIR "/usr/include")
#set(ONNXRUNTIME_LIB_DIR "/usr/lib")

set(ONNXRUNTIME_INCLUDE_DIR "/home/bianbu/spacemit-ort.riscv64.2.0.0+beta0/include")
set(ONNXRUNTIME_LIB_DIR "/home/bianbu/spacemit-ort.riscv64.2.0.0+beta0/lib")


include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIB_DIR})


add_executable(${PROJECT_NAME} ${SRCS})
target_include_directories(${PROJECT_NAME} PRIVATE  ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME}  onnxruntime spacemit_ep spacemit_mpp)


add_definitions(-D__fp16=_Float16)
