/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include <string>
#include <unistd.h>

#include <sys/mman.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_functions.h>
//#include <helper_cuda.h>

// Test the correctness of GDRCopy
#include "gpu_direct_wrapper.h"
#include "common/utils.h"
/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */

KVStore_wrapper * kvstore;
GDR_wrapper * gdr;

template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int matrixMultiply(int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB)
{
    // Allocate host memory for matrices A and B
    unsigned int size_A = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);
    float valB = 0.01f;

    std::string keyA_prefix = std::to_string(dimsA.x) + "x"\
        + std::to_string(dimsA.y) + "A";
    std::string keyB_prefix = std::to_string(dimsB.x) + "x"\
            + std::to_string(dimsB.y) + "B";
    std::string keyA, keyB;
    // generate_test_data
    int nIter = 300;
//    for (int j = 0; j < nIter; j++) {
//        keyA = keyA_prefix + std::to_string(j);
//        keyB = keyB_prefix + std::to_string(j);
//        constantInit(h_A, size_A, 1.0f*(j+1));
//        constantInit(h_B, size_B, valB*(j+1));
//        kvstore->put(keyA, (void*)h_A, mem_size_A);
//        kvstore->put(keyB, (void*)h_B, mem_size_B);
//    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    void* bar_ptr_ = NULL;
    int offset;
    GDR_ptr gdr_p_A, gdr_p_B, gdr_p_C;
    gdr_info_t gdr_info_A, gdr_info_B, gdr_info_C;
    gdr_mh_t gdr_mh_A, gdr_mh_B, gdr_mh_C;

    // Allocate host matrix C
    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C = (float *) malloc(mem_size_C);

    if (h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix C!\n");
        return 1;
        //exit(EXIT_FAILURE);
    }

    cudaError_t error;

    error = cudaMalloc((void **) &d_A, mem_size_A + MB(2));

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        return 1;
        //exit(EXIT_FAILURE);
    }

    gdr_p_A._d_ptr = (CUdeviceptr) d_A;
    assert(!gdr->pin_buffer(gdr_p_A._d_ptr, mem_size_A + MB(2), 0, 0, &gdr_mh_A));
    printf("gdr_pin_buffer A\t");
    assert(!gdr->map(gdr_mh_A, &bar_ptr_, mem_size_A + MB(2)));
    printf("GDR map A\t");
    assert(bar_ptr_);
    assert(!gdr_get_info(gdr->_gdr, gdr_mh_A, &gdr_info_A));
    printf("GDR get_info A\t");
    offset = ((unsigned long)gdr_p_A._d_ptr) - gdr_info_A.va;
    printf("offset of A: %d\n", offset);
    gdr_p_A._h_ptr =(void *) ((char*)bar_ptr_ + offset);


    error = cudaMalloc((void **) &d_B, mem_size_B + MB(2));

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        return 1;
        //exit(EXIT_FAILURE);
    }

    gdr_p_B._d_ptr = (CUdeviceptr) d_B;
    assert(!gdr->pin_buffer(gdr_p_B._d_ptr, mem_size_B + MB(2), 0, 0, &gdr_mh_B));
    printf("gdr_pin_buffer B\t");
    assert(!gdr->map(gdr_mh_B, &bar_ptr_, mem_size_B + MB(2)));
    printf("GDR map B\t");
    assert(bar_ptr_);
    assert(!gdr_get_info(gdr->_gdr, gdr_mh_B, &gdr_info_B));
    printf("GDR get_info B\t");
    offset = ((unsigned long)gdr_p_B._d_ptr) - gdr_info_B.va;
    printf("offset of B: %d\n", offset);
    gdr_p_B._h_ptr =(void *) ((char*)bar_ptr_ + offset);


    error = cudaMalloc((void **) &d_C, mem_size_C + MB(2));

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        return 1;
        //exit(EXIT_FAILURE);
    }

    gdr_p_C._d_ptr = (CUdeviceptr) d_C;
    assert(!gdr->pin_buffer(gdr_p_C._d_ptr, mem_size_C + MB(2), 0, 0, &gdr_mh_C));
    printf("gdr_pin_buffer C\t");
    assert(!gdr->map(gdr_mh_C, &bar_ptr_, mem_size_C + MB(2)));
    printf("GDR map C\t");
    assert(bar_ptr_);
    assert(!gdr_get_info(gdr->_gdr, gdr_mh_C, &gdr_info_C));
    printf("GDR get_info C\t");
    offset = ((unsigned long)gdr_p_C._d_ptr) - gdr_info_C.va;
    printf("offset of C: %d\n", offset);
    gdr_p_C._h_ptr =(void *) ((char*)bar_ptr_ + offset);

    // copy host memory to device
//    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
//
//    if (error != cudaSuccess)
//    {
//        printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
//        exit(EXIT_FAILURE);
//    }
//
//    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);
//
//    if (error != cudaSuccess)
//    {
//        printf("cudaMemcpy (d_B,h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
//        exit(EXIT_FAILURE);
//    }

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    // Performs warmup operation using matrixMul CUDA kernel
//    if (block_size == 16)
//    {
//        matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
//    }
//    else
//    {
//        matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
//    }
//
//    printf("done\n");

    cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        return 1;
        //exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        return 1;
        //exit(EXIT_FAILURE);
    }

    // Record the start event
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        return 1;
        //exit(EXIT_FAILURE);
    }

    // Execute the kernel

    for (int j = 0; j < nIter; j++)
    {
        keyA = keyA_prefix + std::to_string(j);
        keyB = keyB_prefix + std::to_string(j);
        kvstore->get_direct(keyA, gdr_p_A._h_ptr, mem_size_A);
        kvstore->get_direct(keyB, gdr_p_B._h_ptr, mem_size_B);
        if (block_size == 16)
        {
            matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
        else
        {
            matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }

    // Record the stop event
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        return 1;
        //exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        return 1;
        //exit(EXIT_FAILURE);
    }

//    float msecTotal = 0.0f;
//    error = cudaEventElapsedTime(&msecTotal, start, stop);
//
//    if (error != cudaSuccess)
//    {
//        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
//        exit(EXIT_FAILURE);
//    }
//
//    // Compute and print the performance
//    float msecPerMatrixMul = msecTotal / nIter;
//    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
//    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
//    printf(
//        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
//        gigaFlops,
//        msecPerMatrixMul,
//        flopsPerMatrixMul,
//        threads.x * threads.y);

    // Copy result from device to host
    //error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

    //if (error != cudaSuccess)
    //{
    //    printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        //exit(EXIT_FAILURE);
    //}
    // cudaDeviceSynchronize(); Failed
    // sleep(10); Failed
    msync(gdr_p_C._h_ptr, mem_size_C + MB(2), MS_SYNC);
    printf("Checking computed result for correctness: ");
    bool correct = true;

    // test relative error by the formula
    //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
    double eps = 1.e-6 ; // machine zero

    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
    {
        float * result = (float *) gdr_p_C._h_ptr;
        double abs_err = fabs(result[i] - (dimsA.x * 300.0 * valB * 300.0));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err/abs_val/dot_length ;

        if (rel_err > eps)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x*valB, eps);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    // Clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    if (correct)
    {
        return 0;//EXIT_SUCCESS;
    }
    else
    {
        printf("Not correct!\n");
        return 1;//EXIT_FAILURE;
    }
}


/**
 * Program main
 */
int main(int argc, char **argv)
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

//    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
//        checkCmdLineFlag(argc, (const char **)argv, "?"))
//    {
//        printf("Usage -device=n (n >= 0 for deviceID)\n");
//        printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
//        printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
//        printf("  Note: Outer matrix dimensions of A & B matrices must be equal.\n");
//
//        exit(EXIT_SUCCESS);
//    }

    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    int devID = 0;

//    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
//    {
//        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
//        cudaSetDevice(devID);
//    }

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        //exit(EXIT_SUCCESS);
    }

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    int block_size = 32;

    dim3 dimsA(5*2*block_size, 5*2*block_size, 1);
    dim3 dimsB(5*4*block_size, 5*2*block_size, 1);

    // width of Matrix A
//    if (checkCmdLineFlag(argc, (const char **)argv, "wA"))
//    {
//        dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
//    }

    // height of Matrix A
//    if (checkCmdLineFlag(argc, (const char **)argv, "hA"))
//    {
//        dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
//    }

    // width of Matrix B
//    if (checkCmdLineFlag(argc, (const char **)argv, "wB"))
//    {
//        dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
//    }

    // height of Matrix B
//    if (checkCmdLineFlag(argc, (const char **)argv, "hB"))
//    {
//        dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
//    }

    if (dimsA.x != dimsB.y)
    {
        printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
               dimsA.x, dimsB.y);
        return 1;
        //exit(EXIT_FAILURE);
    }

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    kvstore = new KVStore_wrapper();
    gdr = new GDR_wrapper();

    int matrix_result = matrixMultiply(argc, argv, block_size, dimsA, dimsB);

    //exit(matrix_result);
    delete kvstore;
    delete gdr;
}
