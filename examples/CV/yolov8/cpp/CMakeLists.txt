cmake_minimum_required(VERSION 3.10)


project(yolov8_demo)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE RELEASE)


file(GLOB SRCS "*.cc")


find_package(OpenCV 4 REQUIRED)

# 查找 OpenCL
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    message(STATUS "OpenCL found: ${OpenCL_VERSION}")
    message(STATUS "  Includes: ${OpenCL_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${OpenCL_LIBRARIES}")
    
    # 添加 OpenCL 源文件
    set(OPENCL_SOURCES
        ../../../../third-party/opencl/yolo-process/Remap.cc
        ../../../../third-party/opencl/yolo-process/utils.cc
    )
    
    # 添加 OpenCL 编译定义
    add_definitions(-DUSE_OPENCL)
else()
    message(STATUS "OpenCL not found - building without OpenCL support")
    set(OPENCL_SOURCES "")
endif()

# 设置 ONNXRUNTIME 路径（可以根据实际情况调整）
set(ONNXRUNTIME_INCLUDE_DIR "/usr/include")
set(ONNXRUNTIME_LIB_DIR "/usr/lib")

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SRCS} ${OPENCL_SOURCES})

# 设置包含目录
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
if(OpenCL_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_include_directories(${PROJECT_NAME} PRIVATE ../../../../third-party/opencl/yolo-process/)
endif()

# 设置链接库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} onnxruntime spacemit_ep spacemit_mpp)
if(OpenCL_FOUND)
    target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES} OpenCL::OpenCL gbm)
endif()


add_definitions(-D__fp16=_Float16)
