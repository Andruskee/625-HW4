//
// detectOPT.cu - optimized kernel - apply mask to all interior points
//                to detect edges
//

#include "bmp.h"
#define BLOCK_SIZE 16

//
//   a = array in, b = array out
//   m = width, n = height
//

__global__ void gpu_edge_detect(int *a, int *b, int m, int n)
{ 
// TO DO

} 
