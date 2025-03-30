project(nasl)

add_library(nasl INTERFACE)
target_include_directories(nasl INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include/>)
set_target_properties(nasl PROPERTIES PUBLIC_HEADER "include/nasl.h;include/nasl_vec.h;include/nasl_mat.h")
if (MSVC)
    target_compile_options(nasl INTERFACE /Zc:__cplusplus)
endif ()
install(TARGETS nasl EXPORT shady_export_set)
install(TARGETS nasl PUBLIC_HEADER DESTINATION include)

add_subdirectory(test)