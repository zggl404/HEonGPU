set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/examples)

add_executable(lowmem_resnet20_slot lowmem_resnet20_slot.cu)
add_executable(debug_convbn_initial debug_convbn_initial.cu)

target_link_libraries(lowmem_resnet20_slot PRIVATE heongpu CUDA::cudart OpenMP::OpenMP_CXX)

target_link_libraries(debug_convbn_initial PRIVATE heongpu CUDA::cudart OpenMP::OpenMP_CXX)
target_compile_options(lowmem_resnet20_slot PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
target_compile_options(debug_convbn_initial PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
set_target_properties(lowmem_resnet20_slot PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
    CUDA_RUNTIME_LIBRARY Static
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
set_target_properties(debug_convbn_initial PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
    CUDA_RUNTIME_LIBRARY Static
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
