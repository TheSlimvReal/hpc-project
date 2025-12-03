#define n 1000

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

// custom type do allow double subscription (e.g. a[x][y])
typedef double my_arr[n];

void ref(my_arr *a, my_arr *b, my_arr *c)
{
    int i, j, k;
    for (i = 0; i < n; ++i)
        for (k = 0; k < n; k++)
            for (j = 0; j < n; ++j)
                c[i][j] += a[i][k] * b[k][j];
}

__global__ void mod(my_arr *a, my_arr *b, my_arr *c)
{
    int i, j, k;
    for (i = 0; i < n; ++i)
        for (j = 0; j < n; ++j)
            for (k = 0; k < n; k++)
                c[i][j] += a[i][k] * b[k][j];
}

int main(int argc, char **argv)
{
    int i, j;
    double maxError = 0.0;
    cudaEvent_t start_cpu, stop_cpu, start_gpu, stop_gpu;
    float elapsed_time_ms;
    size_t arr_size = n * n * sizeof(double);

    my_arr *a, *b, *c, *c_ref;
    a = (my_arr *)malloc(arr_size);
    b = (my_arr *)malloc(arr_size);
    c = (my_arr *)malloc(arr_size);
    c_ref = (my_arr *)malloc(arr_size);

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
            c_ref[i][j] = 0.0;
        }

    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_cpu, 0);

    ref(a, b, c_ref);

    cudaEventRecord(stop_cpu, 0);
    cudaEventSynchronize(stop_cpu);
    cudaEventElapsedTime(&elapsed_time_ms, start_cpu, stop_cpu);
    printf("Time CPU: %f ms.\n", elapsed_time_ms);

    cudaEventRecord(start_gpu, 0);

    my_arr *a_dev, *b_dev, *c_dev;
    cudaMalloc((void **)&a_dev, arr_size);
    cudaMalloc((void **)&b_dev, arr_size);
    cudaMalloc((void **)&c_dev, arr_size);
    cudaMemcpy(a_dev, a, arr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b, arr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_dev, c, arr_size, cudaMemcpyHostToDevice);

    mod<<<1, 1>>>(a_dev, b_dev, c_dev);

    cudaMemcpy(c, c_dev, arr_size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_gpu, 0);
    cudaEventSynchronize(stop_gpu);
    cudaEventElapsedTime(&elapsed_time_ms, start_gpu, stop_gpu);
    printf("Time GPU: %f ms.\n", elapsed_time_ms);

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
