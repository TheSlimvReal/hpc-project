#define n 1000
#define TILE_SIZE 32

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

// custom type do allow double subscription (e.g. a[x][y])
typedef double arr[n];

void ref(arr *a, arr *b, arr *c)
{
    int i, j, k;
    for (i = 0; i < n; i++)
        for (k = 0; k < n; k++)
            for (j = 0; j < n; j++)
                c[i][j] += a[i][k] * b[k][j];
}

__global__ void mod(arr *a, arr *b, arr *c)
{
    int m, k;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

    double c_tmp = 0.0;
    // 64kb shared memory per multiprocessor: sizeof(double) * 32 * 32 * 2 = 16kb
    __shared__ double shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ double shared_b[TILE_SIZE][TILE_SIZE];

    // loop over the tiles of the input matrices
    for (m = 0; m < (n - 1) / TILE_SIZE + 1; m++)
    {
        // load data from matrices into shared memory
        if ((m * TILE_SIZE + col) < n) {
            shared_a[row][col] = a[globalRow][m * TILE_SIZE + col];
        } else {
            // setting to 0 in case indices are out of bounds
            shared_a[row][col] = 0.0f;
        }
        if ((m * TILE_SIZE + row) < n) 
        {
            shared_b[row][col] = b[(m * TILE_SIZE + row)][globalCol];
        } else 
        {
            shared_b[row][col] = 0.0;
        }
        // synchronize to ensure all threads have loaded their elements
        __syncthreads();

        // add results of current tile
        for (k = 0; k < TILE_SIZE; k++)
            c_tmp += shared_a[row][k] * shared_b[k][col];

        // synchronize to ensure all threads have completed the computation
        __syncthreads();
    }

    // Write the result to global memory
    if (globalRow < n && globalCol < n)
        c[globalRow][globalCol] = c_tmp;
}

int main(int argc, char **argv)
{
    int i, j;
    double maxError = 0.0;
    cudaEvent_t start_cpu, stop_cpu, start_gpu, stop_gpu;
    float elapsed_time_ms;
    size_t size = n * n * sizeof(double);

    arr *a, *b, *c, *c_ref;
    a = (arr *) malloc(size);
    b = (arr *) malloc(size);
    c = (arr *) malloc(size);
    c_ref = (arr *) malloc(size);

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c_ref[i][j] = 0.0;
        }

    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu, 0);

    arr *a_dev, *b_dev, *c_dev;
    cudaMalloc((void **) &a_dev, size);
    cudaMalloc((void **) &b_dev, size);
    cudaMalloc((void **) &c_dev, size);
    cudaMemcpy(a_dev, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_dev, c, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((n-1) / threads.x + 1, (n-1) / threads.y + 1);
    mod<<<blocks, threads>>>(a_dev, b_dev, c_dev);

    cudaMemcpy(c, c_dev, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&elapsed_time_ms, start_gpu, stop_gpu);
    printf("Time GPU: %f ms.\n", elapsed_time_ms);

    cudaEventRecord(start_cpu, 0);

    ref(a, b, c_ref);

    cudaEventRecord(stop_cpu, 0);
    cudaEventSynchronize(stop_cpu);
    cudaEventElapsedTime(&elapsed_time_ms, start_cpu, stop_cpu);
    printf("Time CPU: %f ms.\n", elapsed_time_ms);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (fabs(c[i][j] - c_ref[i][j]) > maxError)
            {
                maxError = fabs(c[i][j] - c_ref[i][j]);
            }
        }
    }
    // Check and see if our maxError is greater than an error bound
    if (maxError > 0.0005f)
        printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
    else
        printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);

    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(c_dev);
    free(a);
    free(b);
    free(c);
    free(c_ref);
    return 0;
}
