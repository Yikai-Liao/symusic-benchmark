cmake_minimum_required(VERSION 3.20)
project(midifile_binding)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET 10.15)

message("Building python binding.")
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# show python executable path
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")

add_subdirectory(../3rdparty/nanobind nanobind EXCLUDE_FROM_ALL)

nanobind_add_module(
    midifile_binding
    NB_STATIC STABLE_ABI LTO PROTECT_STACK NOSTRIP
    ./midifile.cpp
    ../3rdparty/midifile/src/MidiEvent.cpp
    ../3rdparty/midifile/src/MidiEventList.cpp
    ../3rdparty/midifile/src/MidiFile.cpp
    ../3rdparty/midifile/src/MidiMessage.cpp
    ../3rdparty/midifile/src/Binasc.cpp
)
target_include_directories(midifile_binding PRIVATE ../3rdparty/midifile/include)

install(TARGETS midifile_binding LIBRARY DESTINATION ./)

include(CheckIPOSupported)
check_ipo_supported(RESULT has_lto OUTPUT lto_check_output)
if(has_lto)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(STATUS "Clang detected, use -fuse-ld=lld")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
    endif ()
    message(STATUS "Link-time optimization (LTO) enabled")
    set_property(TARGET midifile_binding PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "Link-time optimization (LTO) is not supported: \n${lto_check_output}")
endif()