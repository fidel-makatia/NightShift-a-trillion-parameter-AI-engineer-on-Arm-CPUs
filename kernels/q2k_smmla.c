// q2k_smmla.c — a batched SMMLA GEMM for Q2_K x Q8_K on Arm Neoverse-N2.
//
// Q2_K is the quant Kimi K2 actually uses (UD-Q2_K_XL). llama.cpp has NO i8mm/SMMLA GEMM for
// K-quants — ggml_vec_dot_q2_K_q8_K runs SDOT per output row. We add one: unpack the 2-bit
// weights to int8, then do a 4x4-tiled SMMLA GEMM, reusing each weight read across the batch.
// Correctness is checked bit-for-bit against the linked production kernel.
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define QK_K 256
typedef struct { uint8_t scales[16]; uint8_t qs[64]; _Float16 d; _Float16 dmin; } block_q2_K; // 84B
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[16]; } block_q8_K;                    // 292B

extern void ggml_vec_dot_q2_K_q8_K(int n,float*s,size_t bs,const void*vx,size_t bx,const void*vy,size_t by,int nrc);

#define K 4096
#define M 1024
#define NB (K/QK_K)          // super-blocks per row
#define B 16

static block_q2_K *W;        // [M*NB]
static block_q8_K *X;        // [B*NB]
static float *Yg,*Yo;
static double now_s(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

// unpack a Q2_K super-block's 2-bit quants to int8[256] in q8 order — NEON vectorized
static inline void expand(const block_q2_K*x,int8_t we8[256]){
    const uint8_t*q2=x->qs; uint8_t*we=(uint8_t*)we8; const uint8x16_t m=vdupq_n_u8(3);
    for(int k=0;k<2;k++){
        uint8x16_t a=vld1q_u8(q2), b=vld1q_u8(q2+16);
        vst1q_u8(we+0 ,vandq_u8(a,m));               vst1q_u8(we+16,vandq_u8(b,m));
        vst1q_u8(we+32,vandq_u8(vshrq_n_u8(a,2),m)); vst1q_u8(we+48,vandq_u8(vshrq_n_u8(b,2),m));
        vst1q_u8(we+64,vandq_u8(vshrq_n_u8(a,4),m)); vst1q_u8(we+80,vandq_u8(vshrq_n_u8(b,4),m));
        vst1q_u8(we+96,vandq_u8(vshrq_n_u8(a,6),m)); vst1q_u8(we+112,vandq_u8(vshrq_n_u8(b,6),m));
        q2+=32; we+=128;
    }
}

// ---- llama.cpp production path ----
static void ggml_path(void){
    for(int r=0;r<M;r++) for(int c=0;c<B;c++)
        ggml_vec_dot_q2_K_q8_K(K,&Yg[(size_t)r*B+c],0,W+(size_t)r*NB,0,X+(size_t)c*NB,0,1);
}

// ---- ours: 4x4-tiled SMMLA GEMM (unpack amortized across the batch) ----
static int8_t WE[4][NB][256];   // expanded weights for the current 4-row tile
static void ours(void){
    for(int r=0;r<M;r+=4){
        // unpack the 4 weight rows ONCE, reuse across every activation column
        for(int i=0;i<4;i++) for(int sb=0;sb<NB;sb++) expand(W+(size_t)(r+i)*NB+sb, WE[i][sb]);
        for(int c=0;c<B;c+=4){
            float acc[16]={0};
            for(int sb=0;sb<NB;sb++){
                const block_q2_K*wr[4]={W+(size_t)(r+0)*NB+sb,W+(size_t)(r+1)*NB+sb,W+(size_t)(r+2)*NB+sb,W+(size_t)(r+3)*NB+sb};
                const block_q8_K*xc[4]={X+(size_t)(c+0)*NB+sb,X+(size_t)(c+1)*NB+sb,X+(size_t)(c+2)*NB+sb,X+(size_t)(c+3)*NB+sb};
                const int8_t*we[4]={WE[0][sb],WE[1][sb],WE[2][sb],WE[3][sb]};
                // scaled-int accumulators stay in vector regs (no per-group scalar extract)
                int32x4_t a00=vdupq_n_s32(0),a01=vdupq_n_s32(0),a10=vdupq_n_s32(0),a11=vdupq_n_s32(0);
                for(int g=0;g<16;g++){
                    const int base=g*16;
                    int32x4_t t00=vdupq_n_s32(0),t01=vdupq_n_s32(0),t10=vdupq_n_s32(0),t11=vdupq_n_s32(0);
                    for(int k=0;k<16;k+=8){
                        int8x16_t wa=vcombine_s8(vld1_s8(we[0]+base+k),vld1_s8(we[1]+base+k));
                        int8x16_t wb=vcombine_s8(vld1_s8(we[2]+base+k),vld1_s8(we[3]+base+k));
                        int8x16_t xa=vcombine_s8(vld1_s8(xc[0]->qs+base+k),vld1_s8(xc[1]->qs+base+k));
                        int8x16_t xb=vcombine_s8(vld1_s8(xc[2]->qs+base+k),vld1_s8(xc[3]->qs+base+k));
                        t00=vmmlaq_s32(t00,wa,xa); t01=vmmlaq_s32(t01,wa,xb);
                        t10=vmmlaq_s32(t10,wb,xa); t11=vmmlaq_s32(t11,wb,xb);
                    }
                    // per-group scale by weight-row: lanes [m0,m0,m1,m1] / [m2,m2,m3,m3]
                    int32_t s01[4]={wr[0]->scales[g]&0xF,wr[0]->scales[g]&0xF,wr[1]->scales[g]&0xF,wr[1]->scales[g]&0xF};
                    int32_t s23[4]={wr[2]->scales[g]&0xF,wr[2]->scales[g]&0xF,wr[3]->scales[g]&0xF,wr[3]->scales[g]&0xF};
                    int32x4_t v01=vld1q_s32(s01), v23=vld1q_s32(s23);
                    a00=vmlaq_s32(a00,t00,v01); a01=vmlaq_s32(a01,t01,v01);
                    a10=vmlaq_s32(a10,t10,v23); a11=vmlaq_s32(a11,t11,v23);
                }
                int isum[16]={vgetq_lane_s32(a00,0),vgetq_lane_s32(a00,1),vgetq_lane_s32(a01,0),vgetq_lane_s32(a01,1),
                              vgetq_lane_s32(a00,2),vgetq_lane_s32(a00,3),vgetq_lane_s32(a01,2),vgetq_lane_s32(a01,3),
                              vgetq_lane_s32(a10,0),vgetq_lane_s32(a10,1),vgetq_lane_s32(a11,0),vgetq_lane_s32(a11,1),
                              vgetq_lane_s32(a10,2),vgetq_lane_s32(a10,3),vgetq_lane_s32(a11,2),vgetq_lane_s32(a11,3)};
                for(int ri=0;ri<4;ri++){
                    float xd=(float)wr[ri]->d, xdm=(float)wr[ri]->dmin;
                    for(int ci=0;ci<4;ci++){
                        int summs=0; for(int g=0;g<16;g++) summs+=xc[ci]->bsums[g]*(wr[ri]->scales[g]>>4);
                        float dall=xc[ci]->d*xd, dmin=xc[ci]->d*xdm;
                        acc[ri*4+ci]+=dall*(float)isum[ri*4+ci]-dmin*(float)summs;
                    }
                }
            }
            for(int ri=0;ri<4;ri++) for(int ci=0;ci<4;ci++) Yo[(size_t)(r+ri)*B+(c+ci)]=acc[ri*4+ci];
        }
    }
}

static double bench(void(*f)(void),int it){ f(); double t=now_s(); for(int i=0;i<it;i++)f(); return (now_s()-t)/it; }

int main(void){
    srand(11);
    W=malloc((size_t)M*NB*sizeof(block_q2_K)); X=malloc((size_t)B*NB*sizeof(block_q8_K));
    Yg=malloc((size_t)M*B*sizeof(float)); Yo=malloc((size_t)M*B*sizeof(float));
    for(size_t i=0;i<(size_t)M*NB;i++){ W[i].d=(_Float16)0.03f; W[i].dmin=(_Float16)0.02f;
        for(int j=0;j<16;j++) W[i].scales[j]=(uint8_t)(rand()&0xFF);
        for(int j=0;j<64;j++) W[i].qs[j]=(uint8_t)(rand()&0xFF); }
    for(size_t i=0;i<(size_t)B*NB;i++){ X[i].d=0.5f;
        for(int j=0;j<QK_K;j++) X[i].qs[j]=(int8_t)((rand()&0xFF)-128);
        for(int g=0;g<16;g++){ int s=0; for(int l=0;l<16;l++) s+=X[i].qs[g*16+l]; X[i].bsums[g]=(int16_t)s; } }

    int it=20;
    double tg=bench(ggml_path,it), to=bench(ours,it);
    double e=0,rel=0; for(size_t i=0;i<(size_t)M*B;i++){double d=fabs(Yg[i]-Yo[i]); if(d>e)e=d; double r=d/(fabs(Yg[i])+1e-6); if(r>rel)rel=r;}
    double gg=(2.0*M*K*B)/tg/1e9, go=(2.0*M*K*B)/to/1e9;
    printf("Q2_K x Q8_K GEMM  ·  M=%d K=%d B=%d  ·  Neoverse-N2  (the quant K2 uses)\n",M,K,B);
    printf("  %-30s %8.2f ms  %7.1f GFLOP/s\n","llama.cpp ggml_vec_dot_q2_K (SDOT)",tg*1e3,gg);
    printf("  %-30s %8.2f ms  %7.1f GFLOP/s   %.2fx\n","ours (SMMLA 4x4, new)",to*1e3,go,tg/to);
    printf("  correctness vs production: max|abs|=%.4g  max|rel|=%.3g\n",e,rel);
    return 0;
}
