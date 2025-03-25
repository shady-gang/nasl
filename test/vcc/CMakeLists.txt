if (TARGET vcc)
    get_target_property(NASL_INCLUDE nasl INTERFACE_INCLUDE_DIRECTORIES)
    spv_outputting_test(COMPILER vcc NAME nasl_vcc_swizzle_cpp FILES vec_swizzle.cpp ARGS ${VCC_TEST_ARGS} --entry-point test --execution-model Fragment --std=c++20 -I${NASL_INCLUDE})
endif()