#ifndef NVIDIA_DEFIFES_HPP
#define NVIDIA_DEFIFES_HPP

#include <iostream>

#define checkCudaError(call)                                  \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

#endif
