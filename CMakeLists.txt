cmake_minimum_required(VERSION 3.14)
project(DAG_edits_demo LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wconversion -Wundef")
set(CMAKE_CXX_FLAGS_DEBUG "-march=native -g")
set(CMAKE_CXX_FLAGS_RELEASE "-march=native -g -O3 -ffast-math")

# Ignore some warnings that add a lot of spam and aren't that critical:
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-parentheses")

find_package(CUDA REQUIRED)
find_package(GLEW REQUIRED)

include_directories("${CUDA_INCLUDE_DIRS}")

set(CMAKE_CUDA_HOST_COMPILER gcc-8)
#XXX add -g to debug
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -gencode arch=compute_61,code=sm_61)

# Add --expt-relaxed-constexpr, allowing us to call constexpr functions in
# CUDA device code (without warnings)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Add -Xcompiler to mirror some host compiler flags.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"-march=native -ffast-math\"")

include_directories(src)
include_directories(src/dags)
include_directories(src/dags/hash_dag)
include_directories(src/dags/single_dag)
include_directories(src/gmath)

include_directories(third_party/fontstash/include)

add_executable(DAG_edits_demo
        src/dags/hash_dag/hash_dag.h
        src/dags/hash_dag/hash_dag_colors.h
        src/dags/hash_dag/hash_dag_editors.h
        src/dags/hash_dag/hash_dag_factory.cpp
        src/dags/hash_dag/hash_dag_factory.h
        src/dags/hash_dag/hash_dag_globals.h
        src/dags/hash_dag/hash_table.cpp
        src/dags/hash_dag/hash_table.h
		src/dags/basic_dag/basic_dag.cpp
        src/dags/basic_dag/basic_dag.h
        src/dags/base_dag.h
        src/dags/dag_utils.h
        src/gmath/Matrix3x3.h
        src/gmath/Quaternion.h
        src/gmath/Vector2.h
        src/gmath/Vector3.h
		src/array.h
		src/camera_view.h
        src/color_utils.h
		src/cuda_error_check.h
        src/cuda_gl_buffer.h
        src/cuda_math.h
        src/dag_tracer.cu
        src/dag_tracer.h
        src/engine.cpp
        src/engine.h
        src/main.cpp
        src/path.h
        src/replay.cpp
        src/replay.h
        src/shader.cpp
        src/shader.h
        src/stats.cpp
        src/stats.h
        src/tracer.cu
        src/tracer.h
        src/typedefs.h
        src/utils.h
		src/FastNoise.cpp
		src/FastNoise.h
		src/hacky_profiler.hpp
		src/hacky_profiler.cpp
		src/hacky_profiler.inl
		src/dags/hash_dag/hash_dag_edits.h
		src/dags/hash_dag/hash_dag_utils.h
        src/memory.h
        src/memory.cpp
        src/video.h
        src/video.cpp
		src/script_definitions.h
		src/glfont.h
		src/glfont.cpp
		src/fontstash.cpp)

target_link_libraries(DAG_edits_demo ${CUDA_LIBRARIES} ${GLEW_LIBRARIES} -lglfw -lGL)
