// gemm_q8.c — quantized int8 GEMM microkernel for Arm Neoverse-N2 (Azure Cobalt 100).
//
// The lesson from the GEMV version: at batch=1, MoE inference is MEMORY-bandwidth-bound —
// you stream the whole weight matrix once per token, and fancy compute kernels don't help.
//
// The fix is the same one that powers a shared dev server: BATCHING. Serve B requests at
// once and a single weight read feeds all B — arithmetic intensity rises, the kernel becomes
// COMPUTE-bound, and now Arm's int8 matrix-multiply instruction (SMMLA / i8mm) delivers a
// large speedup. This kernel proves that crossover on the real silicon.
//
//   scalar : portable reference C (and correctness oracle)
//   neon   : Armv8.2 SDOT   (vdotq_s32)
//   i8mm   : Armv8.6 SMMLA  (vmmlaq_s32) — a 2x2 int8 matrix-multiply per instruction:
//            one instruction computes dot(w_m0,x_b0), dot(w_m0,x_b1),
//            dot(w_m1,x_b0), dot(w_m1,x_b1) — 4 outputs, reusing the weight load across the batch.
//
// Build: cc -O3 -mcpu=neoverse-n2 -o gemm_q8 gemm_q8.c -lm ; ./gemm_q8

#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define QK 32
typedef struct { float d; int8_t qs[QK]; } block_q8;

#define K  4096
#define M  4096
#define NB (K / QK)
#define MAXB 16

static block_q8 *W;             // [M*NB]
static block_q8 *X;             // [MAXB*NB]
static float    *Yref, *Yout;   // [M*MAXB]

static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

// ---------------- naive baseline (vectorization disabled — honest "plain C" floor) ----------------
#pragma GCC push_options
#pragma GCC optimize ("no-tree-vectorize")
static void gemm_naive(int B){
    for(int r=0;r<M;r++){
        const block_q8 *wr=W+(size_t)r*NB;
        for(int c=0;c<B;c++){
            const block_q8 *xc=X+(size_t)c*NB;
            float acc=0.f;
            for(int b=0;b<NB;b++){
                int32_t s=0;
                for(int j=0;j<QK;j++) s+=(int32_t)wr[b].qs[j]*(int32_t)xc[b].qs[j];
                acc+=wr[b].d*xc[b].d*(float)s;
            }
            Yout[(size_t)r*B+c]=acc;
        }
    }
}
#pragma GCC pop_options

// ---------------- scalar reference (compiler auto-vectorized) ----------------
static void gemm_scalar(int B){
    for(int r=0;r<M;r++){
        const block_q8 *wr=W+(size_t)r*NB;
        for(int c=0;c<B;c++){
            const block_q8 *xc=X+(size_t)c*NB;
            float acc=0.f;
            for(int b=0;b<NB;b++){
                int32_t s=0;
                for(int j=0;j<QK;j++) s+=(int32_t)wr[b].qs[j]*(int32_t)xc[b].qs[j];
                acc+=wr[b].d*xc[b].d*(float)s;
            }
            Yout[(size_t)r*B+c]=acc;
        }
    }
}

// ---------------- NEON SDOT ----------------
static void gemm_neon(int B){
    for(int r=0;r<M;r++){
        const block_q8 *wr=W+(size_t)r*NB;
        for(int c=0;c<B;c++){
            const block_q8 *xc=X+(size_t)c*NB;
            float acc=0.f;
            for(int b=0;b<NB;b++){
                int32x4_t s=vdupq_n_s32(0);
                s=vdotq_s32(s,vld1q_s8(wr[b].qs),   vld1q_s8(xc[b].qs));
                s=vdotq_s32(s,vld1q_s8(wr[b].qs+16),vld1q_s8(xc[b].qs+16));
                acc+=wr[b].d*xc[b].d*(float)vaddvq_s32(s);
            }
            Yout[(size_t)r*B+c]=acc;
        }
    }
}

// ---------------- i8mm SMMLA, 4x4 register-tiled (4 independent accumulators for ILP) ----------------
#ifdef __ARM_FEATURE_MATMUL_INT8
static void gemm_i8mm(int B){
    for(int r=0;r<M;r+=4){
        const block_q8 *w0=W+(size_t)(r+0)*NB,*w1=W+(size_t)(r+1)*NB,
                       *w2=W+(size_t)(r+2)*NB,*w3=W+(size_t)(r+3)*NB;
        for(int c=0;c<B;c+=4){
            const block_q8 *x0=X+(size_t)(c+0)*NB,*x1=X+(size_t)(c+1)*NB,
                           *x2=X+(size_t)(c+2)*NB,*x3=X+(size_t)(c+3)*NB;
            float y[16]={0};
            for(int b=0;b<NB;b++){
                // four 2x2 sub-tiles = the 4x4 output; 4 SMMLA chains in flight
                int32x4_t t00=vdupq_n_s32(0),t01=vdupq_n_s32(0),t10=vdupq_n_s32(0),t11=vdupq_n_s32(0);
                for(int k=0;k<QK;k+=8){
                    int8x16_t wa=vcombine_s8(vld1_s8(w0[b].qs+k),vld1_s8(w1[b].qs+k)); // rows 0,1
                    int8x16_t wb=vcombine_s8(vld1_s8(w2[b].qs+k),vld1_s8(w3[b].qs+k)); // rows 2,3
                    int8x16_t xa=vcombine_s8(vld1_s8(x0[b].qs+k),vld1_s8(x1[b].qs+k)); // cols 0,1
                    int8x16_t xb=vcombine_s8(vld1_s8(x2[b].qs+k),vld1_s8(x3[b].qs+k)); // cols 2,3
                    t00=vmmlaq_s32(t00,wa,xa); t01=vmmlaq_s32(t01,wa,xb);
                    t10=vmmlaq_s32(t10,wb,xa); t11=vmmlaq_s32(t11,wb,xb);
                }
                float dw[4]={w0[b].d,w1[b].d,w2[b].d,w3[b].d}, dx[4]={x0[b].d,x1[b].d,x2[b].d,x3[b].d};
                int32_t v[16]={ vgetq_lane_s32(t00,0),vgetq_lane_s32(t00,1),vgetq_lane_s32(t01,0),vgetq_lane_s32(t01,1),
                                vgetq_lane_s32(t00,2),vgetq_lane_s32(t00,3),vgetq_lane_s32(t01,2),vgetq_lane_s32(t01,3),
                                vgetq_lane_s32(t10,0),vgetq_lane_s32(t10,1),vgetq_lane_s32(t11,0),vgetq_lane_s32(t11,1),
                                vgetq_lane_s32(t10,2),vgetq_lane_s32(t10,3),vgetq_lane_s32(t11,2),vgetq_lane_s32(t11,3) };
                for(int mi=0;mi<4;mi++) for(int ci=0;ci<4;ci++) y[mi*4+ci]+=dw[mi]*dx[ci]*(float)v[mi*4+ci];
            }
            for(int mi=0;mi<4;mi++) for(int ci=0;ci<4;ci++) Yout[(size_t)(r+mi)*B+(c+ci)]=y[mi*4+ci];
        }
    }
}
#endif

static double bench(void(*fn)(int),int B,int it){ fn(B); double t=now_s(); for(int i=0;i<it;i++) fn(B); return (now_s()-t)/it; }
static double max_err(int B){ double e=0; for(size_t i=0;i<(size_t)M*B;i++){double d=fabs(Yout[i]-Yref[i]); if(d>e)e=d;} return e; }
static void row(const char*n,double s,int B,double base){
    double g=(2.0*M*K*B)/s/1e9;
    printf("    %-7s %8.3f ms  %7.1f GFLOP/s  %5.2fx  max|err|=%.3g\n",n,s*1e3,g,base>0?base/s:1.0,max_err(B));
}

int main(void){
    srand(7);
    W=malloc((size_t)M*NB*sizeof(block_q8));
    X=malloc((size_t)MAXB*NB*sizeof(block_q8));
    Yref=malloc((size_t)M*MAXB*sizeof(float));
    Yout=malloc((size_t)M*MAXB*sizeof(float));
    for(size_t i=0;i<(size_t)M*NB;i++){W[i].d=0.01f+(rand()%100)*1e-4f; for(int j=0;j<QK;j++)W[i].qs[j]=(int8_t)((rand()&0xFF)-128);}
    for(size_t i=0;i<(size_t)MAXB*NB;i++){X[i].d=0.02f; for(int j=0;j<QK;j++)X[i].qs[j]=(int8_t)((rand()&0xFF)-128);}

    printf("Quantized MoE GEMM  ·  M=%d K=%d  ·  Neoverse-N2  ·  speedup vs naive C\n",M,K);
    int batches[]={1,4,16};
    for(int bi=0;bi<3;bi++){
        int B=batches[bi]; int it = B==1?40:(B==4?20:8);
        printf("  batch B=%-2d  (one weight read serves %d concurrent requests)\n",B,B);
        gemm_naive(B); memcpy(Yref,Yout,(size_t)M*B*sizeof(float));   // oracle + honest floor
        double base=bench(gemm_naive,B,it); row("naive",base,B,0);
        memset(Yout,0,(size_t)M*B*sizeof(float)); row("autovec",bench(gemm_scalar,B,it),B,base);
        memset(Yout,0,(size_t)M*B*sizeof(float)); row("neon",bench(gemm_neon,B,it),B,base);
#ifdef __ARM_FEATURE_MATMUL_INT8
        if(B%4==0){ memset(Yout,0,(size_t)M*B*sizeof(float)); row("i8mm",bench(gemm_i8mm,B,it),B,base); }
#endif
    }
    return 0;
}
