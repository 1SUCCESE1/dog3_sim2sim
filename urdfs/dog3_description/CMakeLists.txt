cmake_minimum_required(VERSION 3.5)
project(dog3_description)
set(INSTALL_DIRS
  meshes
  mujoco
  urdf
  xacro
)

if($ENV{ROS_VERSION} EQUAL 2)
  find_package(ament_cmake REQUIRED)
  install(DIRECTORY ${INSTALL_DIRS} DESTINATION share/${PROJECT_NAME})
  ament_package()
endif()
