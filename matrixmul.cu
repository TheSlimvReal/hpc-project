#define n 5000

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

void ref(double (*a)[n], double (*b)[n], double (*c)[n])
{
    int i, j, k;   
    for (i = 0; i < n; ++i)
        for (k = 0; k < n; k++)
            for (j = 0; j < n; ++j)
                c[i][j] += a[i][k] * b[k][j];
}

void mod(double (*a)[n], double (*b)[n], double (*c)[n])
{
    int i, j, k;   
    for (i = 0; i < n; ++i)
        for (k = 0; k < n; k++)
            for (j = 0; j < n; ++j)
                c[i][j] += a[i][k] * b[k][j];
}

int main(int argc, char **argv)
{
    int i, j;
    double maxError = 0.0;
    cudaEvent_t start_cpu, stop_cpu, start_gpu, stop_gpu; 
    float elapsed_time_ms;


    double (*a)[n] = (double (*)[n]) malloc(sizeof(double[n][n]));
    double (*b)[n] = (double (*)[n]) malloc(sizeof(double[n][n]));
    double (*c)[n] = (double (*)[n]) malloc(sizeof(double[n][n]));
    double (*c_ref)[n] = (double (*)[n]) malloc(sizeof(double[n][n]));

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
            c_ref[i][j] = 0.0;
        }

    cudaEventCreate( &start_cpu);
    cudaEventCreate( &stop_cpu ); 
    cudaEventCreate( &start_gpu);
    cudaEventCreate( &stop_gpu );

    cudaEventRecord( start_cpu, 0 );

    ref(a, b, c_ref);

    cudaEventRecord( stop_cpu, 0 );
    cudaEventSynchronize( stop_cpu );
    cudaEventElapsedTime( &elapsed_time_ms, start_cpu, stop_cpu );
    printf("Time CPU: %f ms.\n", elapsed_time_ms);

    cudaEventRecord( start_gpu, 0 );

    mod(a, b, c);

    cudaEventRecord( stop_gpu, 0 ); 
    cudaEventSynchronize( stop_gpu );
    cudaEventElapsedTime( &elapsed_time_ms, start_gpu, stop_gpu );
    printf("Time GPU: %f ms.\n", elapsed_time_ms);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (fabs(c[i][j]-c_ref[i][j]) > maxError) { maxError = fabs(c[i][j]-c_ref[i][j]); }
        }
    }
    // Check and see if our maxError is greater than an error bound
    if (maxError > 0.0005f)
        printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
    else
        printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);

    free(a);
    free(b);
    free(c);
    free(c_ref);
    return 0;
}
