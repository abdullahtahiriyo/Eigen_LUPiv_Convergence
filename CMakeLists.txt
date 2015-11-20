project(Eigen_LUPiv_Convergence)

cmake_minimum_required(VERSION 2.6.2)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Eigen3)
    IF(NOT EIGEN3_FOUND)
        MESSAGE("====================\n"
                "Eigen3 was not found\n"
                "====================\n")
    ENDIF(NOT EIGEN3_FOUND)
    IF(EIGEN3_FOUND)
        MESSAGE("====================\n"
                "Eigen3 was found in:" ${EIGEN3_INCLUDE_DIR} "\n"  
                "====================\n")   
    ENDIF(EIGEN3_FOUND)

include_directories(${EIGEN3_INCLUDE_DIR})

set(SRCS Util.h Geo.h Geo.cpp Constraints.h Constraints.cpp SubSystem.h SubSystem.cpp main.cpp)
add_executable(myproject ${SRCS})
