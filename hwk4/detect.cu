//
// detect.cu - apply mask to all interior points to detect edges, allow a threshold to
//           - be specfied as well to filter out small values
//           - set edge output to black color = (0,0,0) 
//
#include <stdio.h>
#include <stdlib.h>
#include "bmp.h"

#define BLOCK_SIZE 16

extern __global__ void gpu_edge_detect(int *a, int *b, int m, int n);

byte bmp_hdr[BMP_HDR_SIZE];

//
// detect_naive - the naive baseline version
//
void detect_naive(int *a, int *b, int width, int height, int threshold)
{
  int x, y, z;
  int tmp;
  int outputColor; // set outputColor to 255 for white or 0 for black
  volatile int  mask[3][3] = {{-1, 0, -1},
                              { 0, 4,  0},
                              {-1, 0, -1}};

  for (y = 1; y < width-1; y++)
    for (x = 1; x < height-1; x++)
    {
      outputColor = 255; // in any computed color > threshold, set color to black (an edge)
      for (z = 0; z < 3; z++)
      {
        tmp = mask[0][0]*a[(x-1)*width*3+(y-1)*3+z] +
              mask[1][0]*a[x*width*3+(y-1)*3+z] +
              mask[2][0]*a[(x+1)*width*3+(y-1)*3+z] +
              mask[0][1]*a[(x-1)*width*3+y*3+z] +
              mask[1][1]*a[x*width*3+y*3+z] +
              mask[2][1]*a[(x+1)*width*3+y*3+z] +
              mask[0][2]*a[(x-1)*width*3+(y+1)*3+z] +
              mask[1][2]*a[x*width*3+(y+1)*3+z] +
              mask[2][2]*a[(x+1)*width*3+(y+1)*3+z];
        if (tmp>threshold)
              outputColor = 0;
      }
      for (z = 0; z < 3; z++)
        b[x*width*3+y*3+z] = outputColor;
    }
  return;
}

int main(int argc, char *argv[])
{
  int width = 800, height = 800, size;
  int threshold = 40;
  float gpu_elapsed_time_ms, cpu_elapsed_time_ms;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory in host RAM, h_c is used to store CPU result
  int *h_a, *h_b, *h_c;
  
  cudaMallocHost((void **) &h_a, sizeof(int)*MAX_ROW*MAX_COL*3 + 1000);
  cudaMallocHost((void **) &h_b, sizeof(int)*MAX_ROW*MAX_COL*3 + 1000);
  cudaMallocHost((void **) &h_c, sizeof(int)*MAX_ROW*MAX_COL*3 + 1000);

  if (argc<3)
  {
    fprintf(stderr, "Usage: detect <input> <base output> [optimized output]\n");
    exit(1);
  }

  read_bmp(argv[1], bmp_hdr, h_a, &size, &width, &height);

  // compute using CPU
  cudaEventRecord(start, 0);

  detect_naive(h_a, h_b, width, height, threshold);

  printf("Detect naive completed\n");

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
  printf("Elapsed time on CPU: %f ms.\n\n", cpu_elapsed_time_ms);

  write_bmp(argv[2], bmp_hdr, h_b, width, height);

  if (argc > 3)
  {
    // record start execution time of GPU version
    cudaEventRecord(start, 0);

    // allocate memory on CUDA device 
    int *d_a, *d_c;

    // copy matrix A from host to device memory

    // execute kernel 

    // copy results from device to host 

    // record end time working on CUDA
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaFree(d_a);
    cudaFree(d_c);

    // compute time elapsed
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Elapsed time for edge detection on GPU: %f ms.\n\n", gpu_elapsed_time_ms);
    write_bmp(argv[3], bmp_hdr, h_c, width, height);
    // Compute speedup
    printf("Speedup = %f\n", cpu_elapsed_time_ms / gpu_elapsed_time_ms);
  }

  // validate results computed by GPU
  int all_ok = 1;
  for (int i = 0; i < width; ++i) {
      for (int j = 0; j < height; ++j) {
          if (h_b[i*MAX_COL + j] != h_c[i*MAX_COL + j]) {
              all_ok = 0;
          }
      }
  }

  // roughly compute speedup
  if (all_ok) {
      printf("Identical results on GPU and CPU.\n");
  }
  else {
      printf("Results don't match.\n");
  }

  // free memory
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);
  return 0;
}

