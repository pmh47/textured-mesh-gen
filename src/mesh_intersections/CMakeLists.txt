cmake_minimum_required(VERSION 3.14)
project(mesh_intersections)

# You may need to patch tensorflow's eigen following https://bitbucket.org/eigen/eigen/commits/88fc23324517/
# You may need to patch tensorflow's absl following https://github.com/tensorflow/tensorflow/issues/31568#issuecomment-547198495 (causes a crash at runtime, missing a basic_string_view symbol)

set(CMAKE_CXX_STANDARD 17)  # this breaks tensorflow binary compatibility; see https://github.com/tensorflow/tensorflow/issues/31568#issuecomment-547198495 to fix

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(LIBIGL REQUIRED QUIET)

find_path(GUROBI_INCLUDE "gurobi_c++.h" HINTS "${GUROBI_ROOT}/include")
find_library(GUROBI_CXX_LIB "gurobi_g++5.2" HINTS "${GUROBI_ROOT}/lib")
find_library(GUROBI_LIB "gurobi81" HINTS "${GUROBI_ROOT}/lib")

execute_process(COMMAND python -c "import tensorflow; print(' '.join(tensorflow.sysconfig.get_compile_flags()))" OUTPUT_VARIABLE TF_DEFAULT_COMPILE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(TF_DEFAULT_COMPILE_FLAGS UNIX_COMMAND "${TF_DEFAULT_COMPILE_FLAGS}")
set(TF_COMPILE_FLAGS "${TF_DEFAULT_COMPILE_FLAGS}" CACHE STRING "TensorFlow compile options")

execute_process(COMMAND python -c "import tensorflow; print(' '.join(tensorflow.sysconfig.get_link_flags()))" OUTPUT_VARIABLE TF_DEFAULT_LINK_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(TF_DEFAULT_LINK_FLAGS UNIX_COMMAND "${TF_DEFAULT_LINK_FLAGS}")
set(TF_LINK_FLAGS "${TF_DEFAULT_LINK_FLAGS}" CACHE STRING "TensorFlow link options")

add_library(push_ops SHARED push_offsets_op.cpp visualise_mesh_op.cpp mesh_intersections.cpp visualisation.cpp mesh_intersections.h)
add_executable(test_app test_app.cpp mesh_intersections.cpp visualisation.cpp mesh_intersections.h)

# Write the app and library to the source, not build, folder
set_target_properties(test_app PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(push_ops PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(test_app PRIVATE ${GUROBI_INCLUDE})
target_compile_options(test_app PRIVATE ${TF_COMPILE_FLAGS} -march=native)  # tensorflow header path needed for Eigen
target_link_libraries(test_app igl::core igl::cgal igl::opengl_glfw)
target_link_libraries(test_app ${GUROBI_CXX_LIB} ${GUROBI_LIB})

target_include_directories(push_ops PRIVATE ${GUROBI_INCLUDE})
target_compile_options(push_ops PRIVATE ${TF_COMPILE_FLAGS} -march=native)
target_link_libraries(push_ops ${TF_LINK_FLAGS})
target_link_libraries(push_ops igl::core igl::cgal igl::opengl_glfw)
target_link_libraries(push_ops ${GUROBI_CXX_LIB} ${GUROBI_LIB})

