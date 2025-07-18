cmake_minimum_required(VERSION 3.13)

project(nasl)

add_library(nasl INTERFACE)
target_include_directories(nasl INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include/>)
if (MSVC)
    target_compile_options(nasl INTERFACE /Zc:__cplusplus)
endif ()

install(TARGETS nasl EXPORT nasl_export_set)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/nasl DESTINATION include)
#set_target_properties(nasl PROPERTIES PUBLIC_HEADER "include/nasl/nasl.h;include/nasl/nasl_vec.h;include/nasl/nasl_mat.h")
#install(TARGETS nasl PUBLIC_HEADER DESTINATION include/nasl)

add_library(nasl::nasl ALIAS nasl)

if (PROJECT_IS_TOP_LEVEL)
    add_subdirectory(test)
endif()

include(CMakePackageConfigHelpers)
install(EXPORT nasl_export_set DESTINATION share/cmake/nasl/ NAMESPACE nasl:: FILE nasl-targets.cmake)
configure_file(cmake/nasl-config.cmake.in nasl-config.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/nasl-config.cmake" DESTINATION share/cmake/nasl)