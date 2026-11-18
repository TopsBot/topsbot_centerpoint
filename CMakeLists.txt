cmake_minimum_required(VERSION 3.8)
project(topsbot_centerpoint)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(topsbot_ai_infer REQUIRED)

# ---- ta-runtime include / lib paths (respects CMAKE_SYSROOT for cross-compile) ----
if(CMAKE_SYSROOT)
  set(TOPSBOT_CENTERPOINT_SYSTEM_ROOT "${CMAKE_SYSROOT}/usr/local")
else()
  set(TOPSBOT_CENTERPOINT_SYSTEM_ROOT "/usr/local")
endif()
set(TOPSBOT_CENTERPOINT_SYSTEM_INCLUDE "${TOPSBOT_CENTERPOINT_SYSTEM_ROOT}/include")
set(TOPSBOT_CENTERPOINT_SYSTEM_LIB     "${TOPSBOT_CENTERPOINT_SYSTEM_ROOT}/lib")

# ---- centerpoint core library (inference logic, no ROS) ----
# RuntimeModel links against the ta-runtime SDK; preprocessing/postprocessing are
# vendor copies of centerpoint-pointpillars/cpp and are SDK-free. Built SHARED so
# the ta-runtime/viplite dependencies resolve inside the .so rather than on every
# consumer's link line.
add_library(topsbot_centerpoint_core SHARED
  src/centerpoint.cpp
  src/centerpoint_preprocess.cpp
  src/centerpoint_postprocess.cpp
  src/rotated_nms.cpp
  src/runtime_model.cpp
  src/centerpoint_worker.cpp
)
target_include_directories(topsbot_centerpoint_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${TOPSBOT_CENTERPOINT_SYSTEM_INCLUDE}
)
target_link_directories(topsbot_centerpoint_core PUBLIC
  ${TOPSBOT_CENTERPOINT_SYSTEM_LIB}
)
target_link_libraries(topsbot_centerpoint_core
  ta_runtime
  VIPhal
  NBGlinker
  dl
  pthread
  m
)

find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tb_det_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# ---- centerpoint backend plugin (for ai_infer_node integration) ----
add_library(topsbot_centerpoint_backend SHARED
  src/centerpoint_backend.cpp
)
target_include_directories(topsbot_centerpoint_backend PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(topsbot_centerpoint_backend
  topsbot_centerpoint_core
)
target_link_libraries(topsbot_centerpoint_backend
  topsbot_centerpoint_core
  ${OpenCV_LIBS}
)
ament_target_dependencies(topsbot_centerpoint_backend
  rclcpp
  pluginlib
  topsbot_ai_infer
)

# ---- centerpoint inference/replay node (file/manifest -> /detections + BEV) ----
add_executable(centerpoint_infer
  src/centerpoint_infer_node.cpp
)
target_include_directories(centerpoint_infer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(centerpoint_infer
  topsbot_centerpoint_core
  ${OpenCV_LIBS}
)
ament_target_dependencies(centerpoint_infer
  rclcpp
  sensor_msgs
  geometry_msgs
  tb_det_msgs
)

# ---- centerpoint offline single-image CLI (mobilenet_file_cli counterpart) ----
add_executable(centerpoint_file_cli
  src/centerpoint_file_cli.cpp
)
target_include_directories(centerpoint_file_cli PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(centerpoint_file_cli
  topsbot_centerpoint_core
  ${OpenCV_LIBS}
)

# ---- centerpoint single-frame streaming node (independent frame processing) ----
add_executable(centerpoint_single_frame_stream
  src/centerpoint_single_frame_stream.cpp
)
target_include_directories(centerpoint_single_frame_stream PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(centerpoint_single_frame_stream
  topsbot_centerpoint_core
  ${OpenCV_LIBS}
)
ament_target_dependencies(centerpoint_single_frame_stream
  rclcpp
  sensor_msgs
  tb_det_msgs
)

install(TARGETS topsbot_centerpoint_core topsbot_centerpoint_backend centerpoint_infer
    centerpoint_file_cli centerpoint_single_frame_stream
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/download_models.sh
  scripts/ensure_models.sh
  scripts/model_manager.py
  scripts/download_dataset.sh
  scripts/ensure_dataset.sh
  scripts/dataset_manager.py
  DESTINATION share/${PROJECT_NAME}/scripts
)
install(FILES README.md centerpoint_plugins.xml
  DESTINATION share/${PROJECT_NAME}
)

pluginlib_export_plugin_description_file(topsbot_ai_infer centerpoint_plugins.xml)

ament_export_include_directories(include)
ament_export_libraries(topsbot_centerpoint_core)
ament_export_dependencies(
  rclcpp
  pluginlib
  topsbot_ai_infer
)
ament_package()