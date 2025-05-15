project(nasl)

add_library(nasl INTERFACE)
target_include_directories(nasl INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include/>)
if (MSVC)
    target_compile_options(nasl INTERFACE /Zc:__cplusplus)
endif ()
install(TARGETS nasl EXPORT shady_export_set)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/nasl DESTINATION include)
#set_target_properties(nasl PROPERTIES PUBLIC_HEADER "include/nasl/nasl.h;include/nasl/nasl_vec.h;include/nasl/nasl_mat.h")
#install(TARGETS nasl PUBLIC_HEADER DESTINATION include/nasl)

add_subdirectory(test)