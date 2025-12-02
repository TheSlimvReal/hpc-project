#define n 1024

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char **argv)
{
   int i, j, k;

   double(*a) = (double*) _mm_malloc(sizeof(double[n * n]), 64);
   double(*b) = (double*) _mm_malloc(sizeof(double[n * n]), 64);
   double(*c) = (double*) _mm_malloc(sizeof(double[n * n]), 64);

   for (i = 0; i < n; i++)
      for (j = 0; j < n; j++)
      {
         a[i * n + j] = 2.0;
         b[i * n + j] = 3.0;
         c[i * n + j] = 0.0;
      }
  
   double start_time = omp_get_wtime();

   for (i = 0; i < n; ++i)
      for (k = 0; k < n; k++)
         for (j = 0; j < n; ++j)
            __assume_aligned(c, 64);
            __assume_aligned(a, 64);
            __assume_aligned(b, 64);
            c[i * n + j] += a[i * n + k] * b[k * n + j];

   double run_time = omp_get_wtime() - start_time;

   printf("Matrixmul computation in %f seconds\n", run_time);

   FILE *f = fopen("mat-res.txt", "w");
   if (!f)
   {
      perror("fopen");
      return 1;
   }

   fprintf(f, "%d\n\n", n);
   for (int i = 0; i < 1000; i++)
   {
      for (int j = 0; j < 1000; j++)
      {
         fprintf(f, "%.0f ", c[i*n+j]);
      }
      fprintf(f, "\n");
   }

   fclose(f);

   free(a);
   free(b);
   free(c);
   return 0;
}
