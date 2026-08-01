// head2head.c — our SMMLA GEMM vs llama.cpp's PRODUCTION Q8_0 kernel, on Neoverse-N2.
//
// Links the real libggml-cpu.so and calls ggml_vec_dot_q8_0_q8_0 — the exact function
// llama.cpp uses for a Q8_0 matmul row. We compare it, over the same M x K x B work, to
// our hand-written 4x4-tiled SMMLA kernel. Bit-layout matches ggml's block_q8_0 (fp16 scale),
// so this is an honest, apples-to-apples comparison against production code.
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define QK 32
typedef struct { _Float16 d; int8_t qs[QK]; } blk;   // == ggml block_q8_0 (34 bytes, fp16 d)

// the real production kernel from libggml-cpu.so:
extern void ggml_vec_dot_q8_0_q8_0(int n, float *s, size_t bs,
                                   const void *vx, size_t bx,
                                   const void *vy, size_t by, int nrc);

#define K 4096
#define M 4096
#define NB (K/QK)
#define B 8

static blk *W, *X;
static float *Yg, *Yo;
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

// ---- llama.cpp production path: one ggml_vec_dot per output ----
static void ggml_path(void){
    for(int r=0;r<M;r++) for(int c=0;c<B;c++)
        ggml_vec_dot_q8_0_q8_0(K, &Yg[(size_t)r*B+c], 0, W+(size_t)r*NB, 0, X+(size_t)c*NB, 0, 1);
}

// ---- our 4x4-tiled SMMLA GEMM ----
static void ours(void){
    for(int r=0;r<M;r+=4){
        const blk *w0=W+(size_t)(r+0)*NB,*w1=W+(size_t)(r+1)*NB,*w2=W+(size_t)(r+2)*NB,*w3=W+(size_t)(r+3)*NB;
        for(int c=0;c<B;c+=4){
            const blk *x0=X+(size_t)(c+0)*NB,*x1=X+(size_t)(c+1)*NB,*x2=X+(size_t)(c+2)*NB,*x3=X+(size_t)(c+3)*NB;
            float y[16]={0};
            for(int b=0;b<NB;b++){
                int32x4_t t00=vdupq_n_s32(0),t01=vdupq_n_s32(0),t10=vdupq_n_s32(0),t11=vdupq_n_s32(0);
                for(int k=0;k<QK;k+=8){
                    int8x16_t wa=vcombine_s8(vld1_s8(w0[b].qs+k),vld1_s8(w1[b].qs+k));
                    int8x16_t wb=vcombine_s8(vld1_s8(w2[b].qs+k),vld1_s8(w3[b].qs+k));
                    int8x16_t xa=vcombine_s8(vld1_s8(x0[b].qs+k),vld1_s8(x1[b].qs+k));
                    int8x16_t xb=vcombine_s8(vld1_s8(x2[b].qs+k),vld1_s8(x3[b].qs+k));
                    t00=vmmlaq_s32(t00,wa,xa); t01=vmmlaq_s32(t01,wa,xb);
                    t10=vmmlaq_s32(t10,wb,xa); t11=vmmlaq_s32(t11,wb,xb);
                }
                float dw[4]={(float)w0[b].d,(float)w1[b].d,(float)w2[b].d,(float)w3[b].d};
                float dx[4]={(float)x0[b].d,(float)x1[b].d,(float)x2[b].d,(float)x3[b].d};
                int32_t v[16]={vgetq_lane_s32(t00,0),vgetq_lane_s32(t00,1),vgetq_lane_s32(t01,0),vgetq_lane_s32(t01,1),
                               vgetq_lane_s32(t00,2),vgetq_lane_s32(t00,3),vgetq_lane_s32(t01,2),vgetq_lane_s32(t01,3),
                               vgetq_lane_s32(t10,0),vgetq_lane_s32(t10,1),vgetq_lane_s32(t11,0),vgetq_lane_s32(t11,1),
                               vgetq_lane_s32(t10,2),vgetq_lane_s32(t10,3),vgetq_lane_s32(t11,2),vgetq_lane_s32(t11,3)};
                for(int mi=0;mi<4;mi++) for(int ci=0;ci<4;ci++) y[mi*4+ci]+=dw[mi]*dx[ci]*(float)v[mi*4+ci];
            }
            for(int mi=0;mi<4;mi++) for(int ci=0;ci<4;ci++) Yo[(size_t)(r+mi)*B+(c+ci)]=y[mi*4+ci];
        }
    }
}

static double bench(void(*f)(void),int it){ f(); double t=now_s(); for(int i=0;i<it;i++)f(); return (now_s()-t)/it; }

int main(void){
    srand(3);
    W=malloc((size_t)M*NB*sizeof(blk)); X=malloc((size_t)B*NB*sizeof(blk));
    Yg=malloc((size_t)M*B*sizeof(float)); Yo=malloc((size_t)M*B*sizeof(float));
    for(size_t i=0;i<(size_t)M*NB;i++){ W[i].d=(_Float16)0.5f; for(int j=0;j<QK;j++)W[i].qs[j]=(int8_t)((rand()&0xFF)-128);}
    for(size_t i=0;i<(size_t)B*NB;i++){ X[i].d=(_Float16)0.5f; for(int j=0;j<QK;j++)X[i].qs[j]=(int8_t)((rand()&0xFF)-128);}

    int it=15;
    double tg=bench(ggml_path,it), to=bench(ours,it);
    double e=0; for(size_t i=0;i<(size_t)M*B;i++){double d=fabs(Yg[i]-Yo[i]); if(d>e)e=d;}
    double gg=(2.0*M*K*B)/tg/1e9, go=(2.0*M*K*B)/to/1e9;
    printf("Head-to-head vs llama.cpp production Q8_0  ·  M=%d K=%d B=%d  ·  Neoverse-N2\n",M,K,B);
    printf("  %-28s %8.2f ms  %7.1f GFLOP/s\n","llama.cpp ggml_vec_dot_q8_0",tg*1e3,gg);
    printf("  %-28s %8.2f ms  %7.1f GFLOP/s   %.2fx\n","ours (SMMLA 4x4 GEMM)",to*1e3,go,tg/to);
    printf("  correctness vs production: max|err| = %.3g\n",e);
    return 0;
}
