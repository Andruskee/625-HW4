//
// mutliply.cu - example matrix multiplication in CUDA
//
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

//
// gpu_matrix_multiply
//
// grid and block configuration:
//
//   dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
//   dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
//

__global__ void gpu_matrix_multiply(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int h = 0; h < n; h++) 
        {
            sum += a[row * n + h] * b[h * k + col];
        }
        c[row * k + col] = sum;
    }
} 

//
// cpu_matrix_multiply - to verify results
//
void cpu_matrix_multiply(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
        for (int j = 0; j < k; ++j) {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) 
                tmp += h_a[i * n + h] * h_b[h * k + j];
            h_result[i * k + j] = tmp;
        }
}

//
// main
//
int main(int argc, char const *argv[])
{
    int m, n, k;

    srand(123);

    if (argc > 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    } else {
        m = 400;
        n = 500;
        k = 600;
    }

    // allocate memory in host RAM, h_cc is used to store CPU result
    int *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, sizeof(int)*m*n);
    cudaMallocHost((void **) &h_b, sizeof(int)*n*k);
    cudaMallocHost((void **) &h_c, sizeof(int)*m*k);
    cudaMallocHost((void **) &h_cc, sizeof(int)*m*k);

    // randomly init A
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_a[i * n + j] = rand() % 512;
        }
    }

    // randomly init B
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            h_b[i * k + j] = rand() % 512;
        }
    }

    float gpu_elapsed_time_ms, cpu_elapsed_time_ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // record start execution time of GPU version
    cudaEventRecord(start, 0);

    // allocate memory on CUDA device 
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **) &d_a, sizeof(int)*m*n);
    cudaMalloc((void **) &d_b, sizeof(int)*n*k);
    cudaMalloc((void **) &d_c, sizeof(int)*m*k);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   
    // execute kernel 
    gpu_matrix_multiply<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m, n, k);    

    // copy results from device to host 
    cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // record end time working on CUDA
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapsed
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Elapsed time for A %dx%d times B %dx%d on GPU: %f ms.\n\n", m, n, n, k, gpu_elapsed_time_ms);

    // compute using CPU
    cudaEventRecord(start, 0);

    cpu_matrix_multiply(h_a, h_b, h_cc, m, n, k);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
    printf("Elapsed time on CPU: %f ms.\n\n", cpu_elapsed_time_ms);

    // validate results computed by GPU
    int all_ok = 1;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            //printf("[%d][%d]:%d == [%d][%d]:%d, ", i, j, h_cc[i*k + j], i, j, h_c[i*k + j]);
            if(h_cc[i*k + j] != h_c[i*k + j])
            {
                all_ok = 0;
            }
        }
        //printf("\n");
    }

    // roughly compute speedup
    if(all_ok)
    {
        printf("Identical results on GPU and CPU. Speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
    }
    else
    {
        printf("Results don't match.\n");
    }

    // free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    return 0;
}
