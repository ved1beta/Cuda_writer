cmake_minimum_required(VERSION 3.10)
project(FraudDetectionGNN CXX)

# ============================================================
# Build configuration
# ============================================================
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Find GTK3 via pkg-config
# ============================================================
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)

# ============================================================
# Include paths, link directories, compiler definitions
# ============================================================
include_directories(${GTK3_INCLUDE_DIRS} src)
link_directories(${GTK3_LIBRARY_DIRS})
add_definitions(${GTK3_CFLAGS_OTHER})

# ============================================================
# Source files
# ============================================================
set(SOURCES
    src/main.cpp
    src/graph.cpp
    src/neural_net.cpp
    src/gnn.cpp
    src/detector.cpp
    src/gui.cpp
)

# ============================================================
# Executable
# ============================================================
add_executable(fraud_gnn ${SOURCES})

target_link_libraries(fraud_gnn
    ${GTK3_LIBRARIES}
    m          # libm for exp(), sqrt(), etc.
)

target_compile_options(fraud_gnn PRIVATE
    -Wall
    -Wextra
    -O2
)
