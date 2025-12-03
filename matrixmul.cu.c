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

int main(int argc, char **argv)
{
    int i, j, k;

    double (*a)[n] = malloc(sizeof(double[n][n]));
    double (*b)[n] = malloc(sizeof(double[n][n]));
    double (*c)[n] = malloc(sizeof(double[n][n]));

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
        {
            a[i][j] = 2.0;
            b[i][j] = 3.0;
            c[i][j] = 0.0;
        }

    double start_time = omp_get_wtime();

    ref(a, b, c);

    double run_time = omp_get_wtime() - start_time;
    
    printf("%d\n\n", n);
    for (int i = 0; i < 100; i++)
    {
        for (int j = 0; j < 100; j++)
        {
            printf("%.0f ", c[i][j]);
        }
        printf("\n");
    }

    printf("Matrixmul computation in %f seconds\n", run_time);



    free(a);
    free(b);
    free(c);
    return 0;
}
