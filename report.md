# Exam

> icx -O3 -qopenmp ./matrixmul.c 

N=5000 Seq first = 51.188849

> icx -O3 -march=alderlake -qopenmp ./matrixmul.c

See vectorization report (minimal vec speedup)

42.836823 seconds
> icx -O3 -xHost -qopenmp ./matrixmul.c

4.830035 seconds

N=10.000 - 45.670735 seconds
great vec speedup (see report)

inline + reordering:
```C
  double ( *a ) = malloc(sizeof(double[n*n]));
  double ( *b ) = malloc(sizeof(double[ n* n]));
  double ( *c ) = malloc(sizeof(double[ n*n]));

  for (i=0; i<n; i++)
     for (j=0; j<n; j++) {
        a[i*n + j] = 2.0;
        b[i *n + j] = 3.0;
        c[i *n + j] = 0.0;
     }
double start_time = omp_get_wtime();

   for (j=0; j<n; ++j)
   for (i=0; i<n; ++i)
    for (k=0; k<n; k++)
           c[i *n +j] += a[i*n+k]*b[k*n +j];
```
8.115145 seconds


parallel

```C
     #pragma omp parallel for 
    for (i=0; i<n; ++i)
     for (k=0; k<n; k++)
        for (j=0; j<n; ++j)
           c[i][j] += a[i][k]*b[k][j];
```

| seriale | 42 |
| 2 | 20.4 |
| 4 | 10.5 |
| 8 | 7.32 |
| 16 | 7.36 |
| 20 | 7.22 |
| 40 | 7.46 |
