# 设置 CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(yolov8_seg_demo)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g3 -fPIC -march=rv64gcv -mabi=lp64d -std=c++20")


# 指定 OpenCV 的安装目录
set(OpenCV_INSTALL_DIR "/home/user/opencvs/install/")  # 替换为你的 OpenCV 构建目录
set(OpenCV_DIR "${OpenCV_INSTALL_DIR}/lib/cmake/opencv4")
include("/home/user/opencvs/install/lib/cmake/opencv4/OpenCVConfig.cmake")

find_package(OpenCV REQUIRED
    COMPONENTS core imgproc highgui
NO_DEFAULT_PATH)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "  Includes: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "  Libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found! Check OpenCV_DIR or install OpenCV.")
endif()


#find_package(Eigen3 3.4 REQUIRED)

# 手动添加 OpenCV 的头文件目录
include_directories(
    ${OPENCV_INSTALL_DIR}/include/opencv4
)
include_directories(/home/user/library/include/eigen3/)
include_directories(/home/user/code/spacemit-hpc-sdk-main/eigen-rvv/OpenBLAS/include)


# 添加其他需要的库
set(LIBS ${OPENCV_LIBS} onnxruntime spacemit_ep spacemit_mpp)

# 生成可执行文件
add_executable(${PROJECT_NAME} main.cc yolov8-seg.cc)

# 链接库 
#target_link_libraries(${PROJECT_NAME} ${LIBS} ${OpenCV_LIBS} Eigen3::Eigen)
target_link_libraries(${PROJECT_NAME} ${LIBS} ${OpenCV_LIBS} /home/user/code/spacemit-hpc-sdk-main/eigen-rvv/OpenBLAS/lib/libopenblas_x60p-r0.3.27.dev.so)

add_definitions(-D__fp16=_Float16)
