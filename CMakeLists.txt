cmake_minimum_required(VERSION 3.0.0)

project(SpaceColonization)

add_executable(spacecol example.cpp)

target_include_directories(spacecol PRIVATE "nanoflann/include")