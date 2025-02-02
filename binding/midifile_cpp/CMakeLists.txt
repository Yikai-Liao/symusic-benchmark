# ------------------------------------------------------------------------------
# CMake Configuration File for midifile_cpp Python Binding Module
#
# This script builds the midifile_cpp module with Python bindings using nanobind.
# It sets up the required C++ standard, enables Position Independent Code (PIC),
# and configures Link Time Optimization (LTO) if supported.
#
# Requirements:
#   - CMake 3.20 or higher
#   - Python (with Interpreter and Development.Module components)
#   - nanobind (located in ../../3rdparty/nanobind)
#   - midifile library sources (located in ../../3rdparty/midifile)
# ------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)

# Project declaration
project(midifile_cpp LANGUAGES CXX)

# ------------------------------------------------------------------------------
# Define common third-party directories
# ------------------------------------------------------------------------------
# Set the root directory for third-party libraries
set(THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty")

# Directory for nanobind
set(NANOBIND_DIR "${THIRDPARTY_DIR}/nanobind")

# Directory for the midifile library
set(MIDIFILE_DIR "${THIRDPARTY_DIR}/midifile")

# ------------------------------------------------------------------------------
# Compiler and Platform Configuration
# ------------------------------------------------------------------------------
# Use the C++20 standard and require its support
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable Position Independent Code (PIC) for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set the minimum deployment target for macOS (if applicable)
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 10.15 CACHE STRING "Minimum OS X deployment version")
endif()

# ------------------------------------------------------------------------------
# Locate Python and Display Python Executable Path
# ------------------------------------------------------------------------------
message(STATUS "Building Python binding for midifile_cpp")
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")

# ------------------------------------------------------------------------------
# Add nanobind as a Subdirectory
# ------------------------------------------------------------------------------
# Add nanobind to the project. The EXCLUDE_FROM_ALL flag prevents it from
# being built as part of the main build unless explicitly requested.
add_subdirectory(${NANOBIND_DIR} nanobind EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------------------
# Create the Python Module using nanobind
# ------------------------------------------------------------------------------
nanobind_add_module(
    midifile_cpp                # Module name
    NB_STATIC                   # Link nanobind statically
    STABLE_ABI                  # Use stable ABI if applicable
    LTO                         # Enable Link Time Optimization
    PROTECT_STACK               # Enable stack protection
    NOSTRIP                     # Do not strip symbols
    ${CMAKE_CURRENT_SOURCE_DIR}/midifile.cpp    # Python binding implementation file
    ${MIDIFILE_DIR}/src/MidiEvent.cpp
    ${MIDIFILE_DIR}/src/MidiEventList.cpp
    ${MIDIFILE_DIR}/src/MidiFile.cpp
    ${MIDIFILE_DIR}/src/MidiMessage.cpp
    ${MIDIFILE_DIR}/src/Binasc.cpp
)

# Add include directory for midifile headers
target_include_directories(midifile_cpp PRIVATE ${MIDIFILE_DIR}/include)

# ------------------------------------------------------------------------------
# Installation Configuration
# ------------------------------------------------------------------------------
# Install the built module's library to the current directory.
install(
    TARGETS midifile_cpp
    LIBRARY DESTINATION .
)

# ------------------------------------------------------------------------------
# Check and Enable Link Time Optimization (LTO) if Supported
# ------------------------------------------------------------------------------
include(CheckIPOSupported)
check_ipo_supported(RESULT has_lto OUTPUT lto_check_output)

if(has_lto)
    # For Clang, explicitly set the linker flag to use lld
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(STATUS "Clang detected, setting linker flag: -fuse-ld=lld")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
    endif()
    message(STATUS "Link Time Optimization (LTO) enabled")
    set_property(TARGET midifile_cpp PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "Link Time Optimization (LTO) is not supported:\n${lto_check_output}")
endif()