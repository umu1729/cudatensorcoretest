#include <stdio.h>
#include <mma.h>
using namespace nvcuda;


#define WARP_SIZE 32



__global__ void cuda_tensor(float *a, float *b, float *c){

    __shared__ __align__(4) nv_bfloat16 s_a[16*16];
    __shared__ __align__(4) nv_bfloat16 s_b[16*16];
    __shared__ __align__(4) float s_c[16*16];
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    #pragma unroll
    for (int i=0; i<16*16/WARP_SIZE; i++){
        int idx = i*WARP_SIZE + tid;
        s_a[idx] = __float2bfloat16(a[idx]);
        s_b[idx] = __float2bfloat16(b[idx]);
    }
    __syncthreads();
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fA;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> fB;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fC;
    wmma::fill_fragment(fC, 0.0f);
    
    wmma::load_matrix_sync(fA, s_a, 16);
    wmma::load_matrix_sync(fB, s_b, 16);
    wmma::mma_sync(fC, fA, fB, fC);
    wmma::store_matrix_sync(s_c, fC, 16, wmma::mem_row_major);

    #pragma unroll
    for (int i=0; i<16*16/WARP_SIZE; i++){
        int idx = i*WARP_SIZE + tid;
        c[idx] = s_c[idx];
    }
    __syncthreads();
}

int main() {

    float c_a[16*16];
    float c_b[16*16];
    float c_c[16*16];
    
    for (int i=0; i<16*16; i++) c_a[i] = i;
    for (int i=0; i<16*16; i++) c_b[i] = i*(i%3);
    for (int i=0; i<16*16; i++) c_c[i] = 0;

    float *d_a, *d_b, *d_c;
    cudaMalloc( (void**)&d_a, 16*16*sizeof(float) );
    cudaMalloc( (void**)&d_b, 16*16*sizeof(float) );
    cudaMalloc( (void**)&d_c, 16*16*sizeof(float) );
    cudaMemcpy(d_a, c_a, 16*16*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, c_b, 16*16*sizeof(float), cudaMemcpyHostToDevice);
    cuda_tensor<<<1, WARP_SIZE>>>(d_a, d_b, d_c); 
    cudaDeviceSynchronize();
    cudaMemcpy(c_c, d_c, 16*16*sizeof(float), cudaMemcpyDeviceToHost);
    
    
    float chk[16*16];
    for (int i=0; i<16*16; i++) chk[i]=0;
    for (int i=0; i<16; i++) for (int j=0; j<16; j++) for (int k=0; k<16; k++)
        chk[i*16+j] += c_a[i*16+k] * c_b[k*16+j];
    
    float eps = 1e-5;
    bool flag = true;
    printf("host tensor-core diff\n");
    for (int i=0; i<16*16; i++) {
        printf("%f %f %f \n", chk[i], c_c[i], chk[i]- c_c[i]);
        flag &= (chk[i]- c_c[i]) < eps;
        flag &= -eps < (chk[i]- c_c[i]);
    }
    printf("\x1b[31m");
    if (flag) printf("******** OK! ********\n"); else printf ("******** WRONG... ********\n");
    printf("\x1b[0m");
    
    return 0;
}