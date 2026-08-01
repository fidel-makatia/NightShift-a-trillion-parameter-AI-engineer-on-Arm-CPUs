// gemv_q8.c — quantized int8 GEMV kernels for Arm Neoverse-N2 (Azure Cobalt 100).
//
// This is the operation at the heart of MoE inference: y = W · x, where W is an int8
// weight matrix (per-block fp32 scales, Q8_0-style) and x is an int8-quantized activation
// vector. llama.cpp runs this billions of times per K3 generation; it's the compute core
// of every expert's feed-forward layer.
//
// We implement four versions and benchmark them on the actual Neoverse-N2 silicon:
//   1. scalar   — portable reference C (the baseline; also the correctness oracle)
//   2. neon     — Armv8.2 SDOT   (vdotq_s32): 4-way int8 dot-product accumulate
//   3. sve2     — SVE2 SVDOT     (svdot_s32): scalable-vector int8 dot product
//   4. i8mm     — Armv8.6 SMMLA  (vmmlaq_s32): int8 2x2 matrix-multiply, 2 rows/instr
//
// All three optimized paths use instructions KleidiAI itself relies on — but here they're
// hand-written for the exact shape of an MoE expert GEMV, and we measure the speedup.
//
// Build:  see build.sh  (needs -march=armv9-a+sve2+i8mm on Neoverse-N2)
// Run:    ./gemv_q8   -> prints GFLOP/s and speedup for each kernel, plus correctness.

#include <arm_neon.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define QK 32          // block size (elements per fp32 scale) — matches ggml Q8_0
typedef struct { float d; int8_t qs[QK]; } block_q8;   // one quantized block

// --- problem size: representative of a K3 expert FFN GEMV ---
#define K  4096        // input dim  (must be multiple of QK)
#define M  8192        // output rows (experts' neurons)
#define NB (K / QK)    // blocks per row

static block_q8 *W;    // [M * NB]  weight rows
static block_q8 *X;    //     [NB]  activation vector
static float    *Yref, *Yout;

static double now_s(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------- 1. scalar reference
static void gemv_scalar(void) {
    for (int r = 0; r < M; r++) {
        float acc = 0.f;
        const block_q8 *wr = W + (size_t)r * NB;
        for (int b = 0; b < NB; b++) {
            int32_t s = 0;
            for (int j = 0; j < QK; j++) s += (int32_t)wr[b].qs[j] * (int32_t)X[b].qs[j];
            acc += wr[b].d * X[b].d * (float)s;
        }
        Yout[r] = acc;
    }
}

// ---------------------------------------------------------------- 2. NEON SDOT
static void gemv_neon(void) {
    for (int r = 0; r < M; r++) {
        float acc = 0.f;
        const block_q8 *wr = W + (size_t)r * NB;
        for (int b = 0; b < NB; b++) {
            int32x4_t s = vdupq_n_s32(0);
            // 32 int8 = 2 x 16-lane loads; SDOT accumulates 4-int8 groups into int32x4
            s = vdotq_s32(s, vld1q_s8(wr[b].qs +  0), vld1q_s8(X[b].qs +  0));
            s = vdotq_s32(s, vld1q_s8(wr[b].qs + 16), vld1q_s8(X[b].qs + 16));
            acc += wr[b].d * X[b].d * (float)vaddvq_s32(s);
        }
        Yout[r] = acc;
    }
}

// ---------------------------------------------------------------- 3. SVE2 SVDOT
#ifdef __ARM_FEATURE_SVE
static void gemv_sve(void) {
    for (int r = 0; r < M; r++) {
        float acc = 0.f;
        const block_q8 *wr = W + (size_t)r * NB;
        for (int b = 0; b < NB; b++) {
            svint32_t s = svdup_s32(0);
            for (int j = 0; j < QK; j += svcntb()) {
                svbool_t pg = svwhilelt_b8(j, QK);
                svint8_t w = svld1_s8(pg, wr[b].qs + j);
                svint8_t x = svld1_s8(pg, X[b].qs + j);
                s = svdot_s32(s, w, x);          // 4-way int8 dot-product accumulate
            }
            acc += wr[b].d * X[b].d * (float)svaddv_s32(svptrue_b32(), s);
        }
        Yout[r] = acc;
    }
}
#endif

// ---------------------------------------------------------------- 4. i8mm SMMLA (2 rows/instr)
#ifdef __ARM_FEATURE_MATMUL_INT8
// SMMLA: acc[2x2] += a[2x8] * b[2x8]^T. We pack 2 weight rows into `a` and duplicate the
// activation into both rows of `b`, so one SMMLA yields dot(w0,x) and dot(w1,x) over 8 elems.
static void gemv_i8mm(void) {
    for (int r = 0; r < M; r += 2) {
        const block_q8 *w0 = W + (size_t)(r + 0) * NB;
        const block_q8 *w1 = W + (size_t)(r + 1) * NB;
        float acc0 = 0.f, acc1 = 0.f;
        for (int b = 0; b < NB; b++) {
            int32x4_t s = vdupq_n_s32(0);        // [c00 c01 c10 c11]
            for (int c = 0; c < QK; c += 8) {
                // a = [w0[c:c+8], w1[c:c+8]] as int8x16 (row-major 2x8)
                int8x16_t a = vcombine_s8(vld1_s8(w0[b].qs + c), vld1_s8(w1[b].qs + c));
                int8x8_t  xv = vld1_s8(X[b].qs + c);
                int8x16_t bb = vcombine_s8(xv, xv);          // b = [x, x] (2x8)
                s = vmmlaq_s32(s, a, bb);                     // 2x2 int8 matmul-accumulate
            }
            // c00 = dot(w0,x), c10 = dot(w1,x) (lanes 0 and 2)
            float sc = w0[b].d * X[b].d;
            acc0 += sc * (float)vgetq_lane_s32(s, 0);
            acc1 += (w1[b].d * X[b].d) * (float)vgetq_lane_s32(s, 2);
        }
        Yout[r]     = acc0;
        Yout[r + 1] = acc1;
    }
}
#endif

// ---------------------------------------------------------------- harness
static double bench(void (*fn)(void), int iters) {
    fn();                                    // warm
    double t0 = now_s();
    for (int i = 0; i < iters; i++) fn();
    return (now_s() - t0) / iters;
}

static double max_err(void) {
    double e = 0;
    for (int r = 0; r < M; r++) { double d = fabs(Yout[r] - Yref[r]); if (d > e) e = d; }
    return e;
}

static void report(const char *name, double secs, double base) {
    double gflops = (2.0 * (double)M * (double)K) / secs / 1e9;
    printf("  %-8s  %8.3f ms   %7.1f GFLOP/s   %5.2fx   max|err|=%.3g\n",
           name, secs * 1e3, gflops, base > 0 ? base / secs : 1.0, max_err());
}

int main(void) {
    srand(1234);
    W = malloc((size_t)M * NB * sizeof(block_q8));
    X = malloc((size_t)NB * sizeof(block_q8));
    Yref = malloc((size_t)M * sizeof(float));
    Yout = malloc((size_t)M * sizeof(float));
    for (size_t i = 0; i < (size_t)M * NB; i++) {
        W[i].d = 0.01f + (rand() % 100) * 1e-4f;
        for (int j = 0; j < QK; j++) W[i].qs[j] = (int8_t)((rand() & 0xFF) - 128);
    }
    for (int b = 0; b < NB; b++) {
        X[b].d = 0.02f;
        for (int j = 0; j < QK; j++) X[b].qs[j] = (int8_t)((rand() & 0xFF) - 128);
    }

    const int iters = 30;
    printf("Quantized MoE-expert GEMV  ·  M=%d  K=%d  Q8_0-style  ·  Neoverse-N2\n", M, K);
    printf("  kernel      time         throughput      speedup   correctness\n");

    gemv_scalar(); memcpy(Yref, Yout, (size_t)M * sizeof(float));   // oracle
    double base = bench(gemv_scalar, iters);
    report("scalar", base, 0);

    memset(Yout, 0, (size_t)M * sizeof(float)); report("neon",  bench(gemv_neon, iters), base);
#ifdef __ARM_FEATURE_SVE
    memset(Yout, 0, (size_t)M * sizeof(float)); report("sve2",  bench(gemv_sve,  iters), base);
#endif
#ifdef __ARM_FEATURE_MATMUL_INT8
    memset(Yout, 0, (size_t)M * sizeof(float)); report("i8mm",  bench(gemv_i8mm, iters), base);
#endif
    return 0;
}
