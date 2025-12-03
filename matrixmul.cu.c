#define n 1000

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

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
    int i, j, k;
    double maxError = 0.0;

    double (*a)[n] = malloc(sizeof(double[n][n]));
    double (*b)[n] = malloc(sizeof(double[n][n]));
    double (*c)[n] = malloc(sizeof(double[n][n]));
    double (*c_ref)[n] = malloc(sizeof(double[n][n]));

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
            c_ref[i][j] = 0.0;
        }
    ref(a, b, c_ref);

    double start_time = omp_get_wtime();

    mod(a, b, c);

    double run_time = omp_get_wtime() - start_time;

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

    printf("Matrixmul computation in %f seconds\n", run_time);



    free(a);
    free(b);
    free(c);
    free(c_ref);
    return 0;
}
