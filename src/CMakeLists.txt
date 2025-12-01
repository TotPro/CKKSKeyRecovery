cmake_minimum_required(VERSION 3.10)
project(HEAttack CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------- Paths --------
set(HEAAN_DIR "${CMAKE_SOURCE_DIR}/HEAAN/HEAAN")
set(SEAL_DIR "$ENV{HOME}/workspace/seal_install/seal-3.5.4")
set(HELIB_DIR "$ENV{HOME}/workspace/helib_install/helib-1.1.0/helib_pack")

# -------- Include Paths --------
include_directories(
    ${HEAAN_DIR}/src
    ${SEAL_DIR}/include/SEAL-3.5
    ${HELIB_DIR}/include
)

# -------- Library Paths --------
link_directories(
    ${HEAAN_DIR}/lib
    ${SEAL_DIR}/lib
    ${HELIB_DIR}/lib
    /usr/local/lib
    /opt/homebrew/opt/libomp/lib
)

# -------- Sources --------
set(COMMON_SRCS eval.cpp ntl_utils.cpp)

# -------- Executables --------
add_executable(attack attack.cpp ${COMMON_SRCS})
target_link_libraries(attack HEAAN ntl gmp m pthread)

add_executable(seal_attack seal_attack.cpp seal_utils.cpp eval.cpp)
target_link_libraries(seal_attack seal-3.5 ntl gmp omp z pthread)

add_executable(helib_attack helib_attack.cpp helib_utils.cpp ntl_utils.cpp eval.cpp)
target_link_libraries(helib_attack helib ntl gmp pthread)

# -------- RPATH Settings --------
set_target_properties(attack helib_attack  seal_attack PROPERTIES
    BUILD_RPATH "${HEAAN_DIR}/lib;${SEAL_DIR}/lib;/usr/local/lib;/opt/homebrew/opt/libomp/lib"
)
