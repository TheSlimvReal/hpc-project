# Tricks

Compile on mac:
> /opt/homebrew/Cellar/llvm/21.1.6/bin/clang -fopenmp ./matrixmul.c          

# TODO Lab

 - [ ] Test blocks
 - [ ] Test linear
 - [ ] Test aligned + report
 - [ ] Test linear aligned + report
 - [ ] Test parallel for
 - [ ] Test parallel for + default(none)...
 - [ ] Compare firstprivate(a,b,c) with shared(a,b,c)
 - [ ] Redo test on varying cores with best approach
 - [ ] Run best setup with different sizes for problem. 

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

inline + reordering (reordering optimal see lecture slide SPC p73):
TODO just inline speedup!
```C
   double(*a) = malloc(sizeof(double[n * n]));
   double(*b) = malloc(sizeof(double[n * n]));
   double(*c) = malloc(sizeof(double[n * n]));

   for (i = 0; i < n; i++)
      for (j = 0; j < n; j++)
      {
         a[i * n + j] = 2.0;
         b[i * n + j] = 3.0;
         c[i * n + j] = 0.0;
      }
  
   double start_time = omp_get_wtime();

   for (j = 0; j < n; ++j)
      for (i = 0; i < n; ++i)
         for (k = 0; k < n; k++)
            c[i * n + j] += a[i * n + k] * b[k * n + j];
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

| serial | 42 |
| 2 | 20.4 |
| 4 | 10.5 |
| 8 | 7.32 |
| 16 | 7.36 |
| 20 | 7.22 |
| 40 | 7.46 |

big improvement for 

```C
#pragma omp parallel for default(none) private(i, j, k) shared(a, b, c)
```
TODO times:
---
Even faster

```C
#pragma omp parallel for default(none) private(i, j, k) firstprivate(a, b) shared(c)
```


## CUDA

First iteration approach similar to CUDA homework (C reference implementation)
```C
my_arr *a_dev, *b_dev, *c_dev;
cudaMalloc((void **)&a_dev, arr_size);
cudaMalloc((void **)&b_dev, arr_size);
cudaMalloc((void **)&c_dev, arr_size);
cudaMemcpy(a_dev, a, arr_size, cudaMemcpyHostToDevice);
cudaMemcpy(b_dev, b, arr_size, cudaMemcpyHostToDevice);
cudaMemcpy(c_dev, c, arr_size, cudaMemcpyHostToDevice);

mod<<<1, 1>>>(a_dev, b_dev, c_dev);

cudaMemcpy(c, c_dev, arr_size, cudaMemcpyDeviceToHost);

//...

__global__ void mod(my_arr *a, my_arr *b, my_arr *c)
{
    int i, j, k;
    for (i = 0; i < n; ++i)
        for (j = 0; j < n; ++j)
            for (k = 0; k < n; k++)
                c[i][j] += a[i][k] * b[k][j];
}
```

Compiled with 

> nvcc -arch=sm_75 ./matrixmul.cu 

Result:

```
N=1000
Time GPU: 201653.984375 ms.
Time CPU: 3239.883057 ms.
```

Second iteration 

```C
    dim3 threads(32, 32);
    dim3 blocks((n-1) / threads.x + 1, (n-1) / threads.y + 1);
    mod<<<blocks, threads>>>(a_dev, b_dev, c_dev);

    //...

__global__ void mod(my_arr *a, my_arr *b, my_arr *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n | j >= n ) return;
    for (int k = 0; k < n; k++)
        c[i][j] += a[i][k] * b[k][j];
}
```
nvprof
```
/content# nvprof ./a.out 
==1611== NVPROF is profiling process 1611, command: ./a.out
Time GPU: 3994.812744 ms.
==1611== Profiling application: ./a.out
==1611== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   90.90%  3.62948s         1  3.62948s  3.62948s  3.62948s  mod(double[7000]*, double[7000]*, double[7000]*)
                    6.90%  275.68ms         3  91.895ms  91.301ms  92.514ms  [CUDA memcpy HtoD]
                    2.20%  87.719ms         1  87.719ms  87.719ms  87.719ms  [CUDA memcpy DtoH]
```
register
```
/content# nvcc -arch=sm_75 -Xptxas="-v" ./matrixmul.cu 
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z3modPA7000_dS0_S0_' for 'sm_75'
ptxas info    : Function properties for _Z3modPA7000_dS0_S0_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, 376 bytes cmem[0]
```

```
N=1000
Time GPU: 192.292511 ms.
Time CPU: 4307.398438 ms.

N=5000
Time GPU: 9942.131836 ms.
Time CPU: 443797.250000 ms.
```
size n=5000
| dim| time|
|---|---|
| 32x32 | 9942 |
| 1024x1 | 72698 |
| 1x1024| 1857 |
| 1x512| 1696 |
| 1x256| 1627 |

n=7000

|dim |time|
|---|---|
|1x32| 6694|
|1x64| 4081|
|1x128| 3982|
|1x256| 4323|
|1x512 | 4364|
|1x1024| 4665|
|2x128| 4011|
|2x256| 4348|
|4x256| 6917|
|2x512| 4386|