project(nasl)

include(CTest)

if (BUILD_TESTING)
    enable_language(CXX)
    add_executable(test_nasl src/test_vec.cpp)
    target_include_directories(test_nasl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
    set_property(TARGET test_nasl PROPERTY CXX_STANDARD 20)
    add_test(NAME test_nasl COMMAND test_nasl)
endif ()