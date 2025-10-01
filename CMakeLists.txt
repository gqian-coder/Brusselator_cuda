cmake_minimum_required(VERSION 3.19)
project(BrusselatorCUDA LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(ADIOS2 REQUIRED PATHS /home/qg7/indir/)
find_package(mgard REQUIRED)

find_package(CUDAToolkit REQUIRED)
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -w")

add_definitions(-DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer -Wfatal-errors -Wall -Wextra")

# Brusselator executable
add_executable(brusselator_cuda
  src/brusselator_cuda.cu
  src/kernels.cuh
  src/funInit.tpp
)

target_compile_features(brusselator_cuda PRIVATE cxx_std_17)
target_include_directories(brusselator_cuda PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(brusselator_cuda PRIVATE MPI::MPI_CXX adios2::adios2 mgard::mgard cuda)

# FitzHugh-Nagumo executable
add_executable(fitzhugh_nagumo_cuda
  src/fitzhugh_nagumo_cuda.cu
  src/kernels.cuh
  src/funInit.tpp
)

target_compile_features(fitzhugh_nagumo_cuda PRIVATE cxx_std_17)
target_include_directories(fitzhugh_nagumo_cuda PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(fitzhugh_nagumo_cuda PRIVATE MPI::MPI_CXX adios2::adios2 mgard::mgard cuda)
