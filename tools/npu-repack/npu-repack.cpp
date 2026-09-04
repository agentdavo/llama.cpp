// npu-repack - measure and emit DPU-consumable weights from a mixed-quant GGUF (e.g. GSQ-RCO).
//
// The 3720 DPU applies its requant scale per output channel (VpuDPUInvariantRegisters ppe_scale, and
// the per-channel table in the slab image of hpi_npu3720_repack.h). It has no per-K-block scale. GGUF
// k-quants and i-quants all scale along K in 32- or 256-wide groups, so no GGUF type is directly
// DPU-consumable, whatever the wmode.
//
// Two ways out:
//   (A) requantise to a DPU-native integer with per-channel scales. To get resolution back, split the
//       GEMM along K into tiles of G columns: inside a tile a per-(channel,tile) scale IS a
//       per-channel scale. Costs K/G workloads plus a partial-sum accumulation.
//   (B) dequantise on a SHAVE into CMX and feed the DPU, keeping the source bpw in DDR.
// This tool measures (A) so the choice is made on numbers.
//
// Dequantisation goes through ggml to_float, so the reference is the reference. --emit writes the
// per-channel int8 slab image; --verify-repack proves it matches hpi_repack_q8_0 byte for byte.
//
// Usage: llama-npu-repack <model.gguf> [--rows N] [--per-type N] [--tensor SUBSTR] [--all]
//                                      [--emit DIR] [--slabch N] [--verify-repack]

#include "ggml.h"
#include "gguf.h"

#include "../../ggml/src/ggml-npu/hpi-3720/hpi_npu3720_repack.h"   // the proven Q8_0 layout, for --verify-repack

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <map>
#include <string>
#include <vector>

namespace {

// bits = DPU wmode width (8 -> I8, 4 -> I4). G = K columns sharing one scale; G == 0 means the whole
// row, which is the layout hpi_npu3720_repack.h emits and the only one that is a single workload.
struct scheme {
    const char * name;
    int          bits;
    int          G;
};

const scheme SCHEMES[] = {
    { "i8/row",  8,   0 },
    { "i4/row",  4,   0 },
    { "i8/256",  8, 256 },
    { "i4/256",  4, 256 },
    { "i4/128",  4, 128 },
    { "i4/64",   4,  64 },
    { "i4/32",   4,  32 },
};
const int N_SCHEMES = (int) (sizeof(SCHEMES) / sizeof(SCHEMES[0]));

// DDR cost of a scheme in bits per weight: the integer plus one fp16 scale per group.
double scheme_bpw(const scheme & s, int64_t K) {
    const double g = s.G == 0 ? (double) K : (double) s.G;
    return (double) s.bits + 16.0 / g;
}

// Squared error of requantising one row under a scheme. Symmetric, no zero point: an asymmetric
// scheme would need a per-channel correction proportional to sum_k x[k], which the PPE bias cannot
// express because it depends on the activation.
//
// The scale is rounded to fp16 here, so the number is what the hardware would actually see.
double row_sqerr(const float * w, int64_t K, const scheme & s) {
    const int     qmax = (1 << (s.bits - 1)) - 1;   // 127 or 7
    const int64_t G    = s.G == 0 ? K : (int64_t) s.G;

    double se = 0.0;
    for (int64_t g0 = 0; g0 < K; g0 += G) {
        const int64_t n = std::min<int64_t>(G, K - g0);

        float amax = 0.0f;
        for (int64_t i = 0; i < n; i++) {
            const float a = std::fabs(w[g0 + i]);
            if (a > amax) amax = a;
        }
        if (amax == 0.0f) continue;                 // all-zero group is exact

        const float scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16(amax / (float) qmax));
        if (scale == 0.0f) continue;

        for (int64_t i = 0; i < n; i++) {
            float q = std::rint(w[g0 + i] / scale);
            if (q >  (float) qmax) q =  (float) qmax;
            if (q < -(float) qmax) q = -(float) qmax;
            const double d = (double) w[g0 + i] - (double) (q * scale);
            se += d * d;
        }
    }
    return se;
}

struct acc {
    double  se[N_SCHEMES] = { 0.0 };
    double  ref     = 0.0;   // sum of w^2 over the sampled elements
    int64_t tensors = 0;
    int64_t elems   = 0;     // file-wide elements of this type (not the sample)
    int64_t bytes   = 0;
};

// Slab layout, verbatim from hpi_npu3720_repack.h (SWZ=0):
//   [ slabch x 16B {u32 data_off, u32 0x00ffffff, f32 sw, u32 0} ][ slabch x K int8 rows ]
// data_off is relative to the slab base; the loader relocates it. Only the source of the floats
// differs from hpi_repack_q8_0: ggml to_float, so any quant type works.
void emit_slab_rows(const float * w, int64_t slabch, int64_t K, uint8_t * out) {
    const size_t table_bytes = (size_t) slabch * 16;
    int8_t *     rows        = (int8_t *) (out + table_bytes);

    for (int64_t z = 0; z < slabch; z++) {
        const float * src = w + z * K;

        float amax = 0.0f;
        for (int64_t i = 0; i < K; i++) {
            const float a = std::fabs(src[i]);
            if (a > amax) amax = a;
        }
        if (amax == 0.0f) amax = 1.0f;
        const float sw = amax / 127.0f;

        uint8_t *      e     = out + (size_t) z * 16;
        const uint32_t off   = (uint32_t) (table_bytes + (size_t) z * (size_t) K);
        const uint32_t magic = 0x00ffffffu, zero = 0;
        memcpy(e +  0, &off,   4);
        memcpy(e +  4, &magic, 4);
        memcpy(e +  8, &sw,    4);
        memcpy(e + 12, &zero,  4);

        int8_t * dst = rows + (size_t) z * (size_t) K;
        for (int64_t i = 0; i < K; i++) {
            long q = lrintf(src[i] / sw);           // round-half-to-even, as the proven path does
            if (q >  127) q =  127;
            if (q < -127) q = -127;
            dst[i] = (int8_t) q;
        }
    }
}

std::string basename_of(const char * p) {
    const char * a = strrchr(p, '/');
    const char * b = strrchr(p, '\\');
    const char * s = a > b ? a : b;
    return s ? s + 1 : p;
}

} // namespace

int main(int argc, char ** argv) {
    const char * path     = nullptr;
    const char * filter   = nullptr;
    const char * emit_dir = nullptr;
    const char * raw_dir  = nullptr;
    int64_t      max_rows = 64;    // sampled rows per tensor
    int64_t      per_type = 6;     // tensors sampled per quant type
    int64_t      slabch   = 16;    // output channels per slab
    bool         all      = false;
    bool         verify   = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--emit") == 0 && i + 1 < argc) {
            emit_dir = argv[++i];
        } else if (strcmp(argv[i], "--emit-raw") == 0 && i + 1 < argc) {
            raw_dir = argv[++i];
        } else if (strcmp(argv[i], "--slabch") == 0 && i + 1 < argc) {
            slabch = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--verify-repack") == 0) {
            verify = true;
        } else if (strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
            max_rows = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--per-type") == 0 && i + 1 < argc) {
            per_type = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--tensor") == 0 && i + 1 < argc) {
            filter = argv[++i];
        } else if (strcmp(argv[i], "--all") == 0) {
            all = true;
        } else if (argv[i][0] == '-') {
            fprintf(stderr, "unknown option %s\n", argv[i]);
            return 1;
        } else {
            path = argv[i];
        }
    }
    if (!path) {
        fprintf(stderr, "usage: llama-npu-repack <model.gguf> [--rows N] [--per-type N] [--tensor SUBSTR] [--all]\n");
        return 1;
    }

    gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(path, gp);
    if (!ctx) {
        fprintf(stderr, "failed to open %s as GGUF\n", path);
        return 1;
    }

    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "failed to reopen %s\n", path);
        gguf_free(ctx);
        return 1;
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    const size_t  data_off  = gguf_get_data_offset(ctx);

    printf("file      : %s\n", basename_of(path).c_str());
    printf("tensors   : %lld,  data offset %zu\n", (long long) n_tensors, data_off);
    printf("sampling  : %lld rows/tensor, %lld tensors/type%s\n\n",
           (long long) max_rows, (long long) per_type, all ? " (--all: every tensor)" : "");

    // -----------------------------------------------------------------------------------------
    // --verify-repack: prove the generic emitter reproduces the proven Q8_0 path byte for byte.
    // Needs a Q8_0 2-D tensor in the file; that is the only type both paths can produce.
    if (verify) {
        int rc = 1;
        for (int64_t t = 0; t < n_tensors; t++) {
            if (gguf_get_tensor_type(ctx, t) != GGML_TYPE_Q8_0) continue;
            const int64_t * ne = gguf_get_tensor_ne(ctx, t);
            if (ne[2] != 1 || ne[3] != 1 || ne[1] < slabch)     continue;

            const char *  name      = gguf_get_tensor_name(ctx, t);
            const int64_t K         = ne[0];
            const size_t  row_bytes = (size_t) (K / 32) * sizeof(hpi_block_q8_0);
            const size_t  slab_sz   = (size_t) slabch * 16 + (size_t) slabch * (size_t) K;

            std::vector<uint8_t> blocks((size_t) slabch * row_bytes);
            const size_t off = data_off + gguf_get_tensor_offset(ctx, t);
            if (fseeko64(f, (off64_t) off, SEEK_SET) != 0 ||
                fread(blocks.data(), 1, blocks.size(), f) != blocks.size()) {
                fprintf(stderr, "verify: read failed on %s\n", name);
                break;
            }

            std::vector<uint8_t> ref(slab_sz), got(slab_sz);
            if (hpi_repack_q8_0((const hpi_block_q8_0 *) blocks.data(), (int) slabch, (int) K, (int) slabch,
                                ref.data()) != 0) {
                fprintf(stderr, "verify: hpi_repack_q8_0 rejected %s\n", name);
                break;
            }

            std::vector<float> w((size_t) slabch * (size_t) K);
            const ggml_type_traits * tr = ggml_get_type_traits(GGML_TYPE_Q8_0);
            for (int64_t z = 0; z < slabch; z++) {
                tr->to_float(blocks.data() + (size_t) z * row_bytes, w.data() + z * K, K);
            }
            emit_slab_rows(w.data(), slabch, K, got.data());

            const bool same = memcmp(ref.data(), got.data(), slab_sz) == 0;
            printf("=== verify-repack ===\n");
            printf("  tensor %s  slabch=%lld K=%lld  slab=%zu B\n", name, (long long) slabch, (long long) K, slab_sz);
            printf("  generic emitter vs hpi_repack_q8_0: %s\n\n", same ? "BYTE-IDENTICAL" : "*** MISMATCH ***");
            rc = same ? 0 : 2;
            break;
        }
        if (rc == 1) printf("=== verify-repack ===\n  no Q8_0 2-D tensor in this file; nothing to verify against\n\n");
        if (rc == 2) { fclose(f); gguf_free(ctx); return 2; }
    }

    // -----------------------------------------------------------------------------------------
    // pass 1: inventory (metadata only, whole file)
    std::map<std::string, acc> by_type;
    int64_t tot_bytes = 0, tot_elems = 0;

    for (int64_t t = 0; t < n_tensors; t++) {
        const ggml_type ty = gguf_get_tensor_type(ctx, t);
        const int64_t * ne = gguf_get_tensor_ne(ctx, t);
        const int64_t   n  = ne[0] * ne[1] * ne[2] * ne[3];
        const size_t    sz = gguf_get_tensor_size(ctx, t);

        acc & a = by_type[ggml_type_name(ty)];
        a.tensors++;
        a.elems += n;
        a.bytes += (int64_t) sz;
        tot_bytes += (int64_t) sz;
        tot_elems += n;
    }

    printf("=== inventory ===\n");
    printf("%-9s %7s %14s %10s %7s %8s\n", "type", "tensors", "elements", "GB", "share", "bpw");
    for (const auto & kv : by_type) {
        const acc & a = kv.second;
        printf("%-9s %7lld %14lld %10.3f %6.1f%% %8.3f\n",
               kv.first.c_str(), (long long) a.tensors, (long long) a.elems,
               (double) a.bytes / 1e9, 100.0 * (double) a.bytes / (double) tot_bytes,
               8.0 * (double) a.bytes / (double) a.elems);
    }
    printf("%-9s %7lld %14lld %10.3f %6.1f%% %8.3f   <- whole file\n\n",
           "TOTAL", (long long) n_tensors, (long long) tot_elems,
           (double) tot_bytes / 1e9, 100.0, 8.0 * (double) tot_bytes / (double) tot_elems);

    // -----------------------------------------------------------------------------------------
    // pass 2: sampled requantisation error
    std::map<std::string, int64_t> sampled;
    std::map<std::string, acc>     err;

    printf("=== per-tensor added error (relative Frobenius, sampled) ===\n");
    printf("%-34s %-8s %6s %6s", "tensor", "type", "N", "K");
    for (int s = 0; s < N_SCHEMES; s++) printf(" %9s", SCHEMES[s].name);
    printf("\n");

    std::vector<uint8_t> raw;
    std::vector<float>   row;

    for (int64_t t = 0; t < n_tensors; t++) {
        const char *    name = gguf_get_tensor_name(ctx, t);
        const ggml_type ty   = gguf_get_tensor_type(ctx, t);
        const int64_t * ne   = gguf_get_tensor_ne(ctx, t);

        if (filter && !strstr(name, filter)) continue;
        if (ne[2] != 1 || ne[3] != 1)        continue;   // 2-D weights only
        if (ne[0] < 256 || ne[1] < 8)        continue;   // skip norms and tiny vectors

        const ggml_type_traits * tr = ggml_get_type_traits(ty);
        if (!tr->to_float) continue;

        const std::string tn = ggml_type_name(ty);
        if (!all && filter == nullptr && sampled[tn] >= per_type) continue;
        sampled[tn]++;

        ggml_quantize_init(ty);   // i-quant grids

        const int64_t K    = ne[0];                 // contraction dim (row length)
        const int64_t N    = ne[1];                 // output channels
        const int64_t blck = tr->blck_size;
        if (K % blck != 0) continue;
        const size_t row_bytes = (size_t) (K / blck) * tr->type_size;

        raw.resize(row_bytes);
        row.resize((size_t) K);

        const int64_t nr   = std::min<int64_t>(max_rows, N);
        const int64_t step = N / nr;

        acc  a;
        bool ok = true;
        for (int64_t r = 0; r < nr; r++) {
            const size_t off = data_off + gguf_get_tensor_offset(ctx, t) + (size_t) (r * step) * row_bytes;
            if (fseeko64(f, (off64_t) off, SEEK_SET) != 0 || fread(raw.data(), 1, row_bytes, f) != row_bytes) {
                fprintf(stderr, "read failed on %s row %lld\n", name, (long long) r);
                ok = false;
                break;
            }
            tr->to_float(raw.data(), row.data(), K);

            for (int64_t i = 0; i < K; i++) a.ref += (double) row[i] * (double) row[i];
            for (int s = 0; s < N_SCHEMES; s++) a.se[s] += row_sqerr(row.data(), K, SCHEMES[s]);
        }
        if (!ok || a.ref == 0.0) continue;

        printf("%-34s %-8s %6lld %6lld", name, tn.c_str(), (long long) N, (long long) K);
        for (int s = 0; s < N_SCHEMES; s++) printf(" %8.4f%%", 100.0 * std::sqrt(a.se[s] / a.ref));
        printf("\n");
        fflush(stdout);

        acc & g = err[tn];
        g.ref += a.ref;
        g.tensors++;
        for (int s = 0; s < N_SCHEMES; s++) g.se[s] += a.se[s];
    }

    // -----------------------------------------------------------------------------------------
    printf("\n=== added error by source type (aggregate over sampled tensors) ===\n");
    printf("%-9s %7s", "type", "sampled");
    for (int s = 0; s < N_SCHEMES; s++) printf(" %9s", SCHEMES[s].name);
    printf("\n");
    for (const auto & kv : err) {
        printf("%-9s %7lld", kv.first.c_str(), (long long) kv.second.tensors);
        for (int s = 0; s < N_SCHEMES; s++) {
            printf(" %8.4f%%", 100.0 * std::sqrt(kv.second.se[s] / kv.second.ref));
        }
        printf("\n");
    }

    // -----------------------------------------------------------------------------------------
    // --emit-raw: the same repack, but unpacked, for the offline blob authoring in re/emit_gsqrco_gemm.py.
    // <name>.i8 = N x K int8 rows, <name>.sw = N f32 channel scales, <name>.f32 = the ggml dequantised
    // weight (ground truth for the error the repack adds). Needs --tensor; these files are large.
    if (raw_dir) {
        std::error_code ec;
        std::filesystem::create_directories(raw_dir, ec);
        printf("\n=== emit-raw ===\n");
        if (!filter) {
            printf("  --emit-raw needs --tensor: these files are N*K*5 bytes each\n");
        }
        for (int64_t t = 0; filter && t < n_tensors; t++) {
            const char *    name = gguf_get_tensor_name(ctx, t);
            const ggml_type ty   = gguf_get_tensor_type(ctx, t);
            const int64_t * ne   = gguf_get_tensor_ne(ctx, t);
            if (!strstr(name, filter) || ne[2] != 1 || ne[3] != 1) continue;

            const ggml_type_traits * tr = ggml_get_type_traits(ty);
            if (!tr->to_float) continue;
            const int64_t K = ne[0], N = ne[1];
            if (K % tr->blck_size != 0) continue;
            ggml_quantize_init(ty);

            const size_t         row_bytes = (size_t) (K / tr->blck_size) * tr->type_size;
            std::vector<float>   w((size_t) K);
            std::vector<int8_t>  q((size_t) K);
            std::vector<uint8_t> rb(row_bytes);
            std::vector<float>   sw((size_t) N);

            const std::string base = std::string(raw_dir) + "/" + name;
            FILE * fi8 = fopen((base + ".i8").c_str(), "wb");
            FILE * f32 = fopen((base + ".f32").c_str(), "wb");
            if (!fi8 || !f32) { fprintf(stderr, "  cannot write %s.*\n", base.c_str()); if (fi8) fclose(fi8); if (f32) fclose(f32); continue; }

            bool ok = fseeko64(f, (off64_t) (data_off + gguf_get_tensor_offset(ctx, t)), SEEK_SET) == 0;
            for (int64_t n = 0; ok && n < N; n++) {
                if (fread(rb.data(), 1, row_bytes, f) != row_bytes) { ok = false; break; }
                tr->to_float(rb.data(), w.data(), K);

                float amax = 0.0f;
                for (int64_t i = 0; i < K; i++) { const float a = std::fabs(w[i]); if (a > amax) amax = a; }
                if (amax == 0.0f) amax = 1.0f;
                sw[n] = amax / 127.0f;
                for (int64_t i = 0; i < K; i++) {
                    long v = lrintf(w[i] / sw[n]);
                    if (v >  127) v =  127;
                    if (v < -127) v = -127;
                    q[i] = (int8_t) v;
                }
                if (fwrite(q.data(), 1, (size_t) K, fi8) != (size_t) K) { ok = false; break; }
                if (fwrite(w.data(), 4, (size_t) K, f32) != (size_t) K) { ok = false; break; }
            }
            fclose(fi8);
            fclose(f32);
            if (!ok) { fprintf(stderr, "  FAILED on %s\n", name); continue; }

            FILE * fs = fopen((base + ".sw").c_str(), "wb");
            if (fs) { fwrite(sw.data(), 4, (size_t) N, fs); fclose(fs); }
            printf("  %s  %s  N=%lld K=%lld -> .i8 .sw .f32\n", name, ggml_type_name(ty), (long long) N, (long long) K);
        }
    }

    // -----------------------------------------------------------------------------------------
    // --emit: write the i8/row slab images.
    if (emit_dir) {
        std::error_code ec;
        std::filesystem::create_directories(emit_dir, ec);

        // size the job first - emitting a whole 27 G-element model is 27 GB of slabs.
        int64_t plan_bytes = 0, plan_tensors = 0;
        for (int64_t t = 0; t < n_tensors; t++) {
            const char *    name = gguf_get_tensor_name(ctx, t);
            const int64_t * ne   = gguf_get_tensor_ne(ctx, t);
            if (filter && !strstr(name, filter))            continue;
            if (ne[2] != 1 || ne[3] != 1)                   continue;
            if (ne[1] % slabch != 0 || ne[0] < 256)         continue;
            if (!ggml_get_type_traits(gguf_get_tensor_type(ctx, t))->to_float) continue;
            plan_bytes += ne[1] * 16 + ne[1] * ne[0];
            plan_tensors++;
        }
        printf("\n=== emit (i8/row slabs, slabch=%lld) ===\n", (long long) slabch);
        printf("  %lld tensors -> %.3f GB in %s\n", (long long) plan_tensors, (double) plan_bytes / 1e9, emit_dir);
        if (plan_bytes > 8LL * 1000 * 1000 * 1000 && !all) {
            printf("  refusing: over 8 GB and no --all given (narrow it with --tensor, or pass --all)\n");
        } else {
            const std::string mpath = std::string(emit_dir) + "/manifest.txt";
            FILE * mf = fopen(mpath.c_str(), "w");
            if (!mf) {
                fprintf(stderr, "  cannot write %s\n", mpath.c_str());
            } else {
                fprintf(mf, "# name\tsrc_type\tN\tK\tslabch\tbytes\n");
                std::vector<float>   wbuf;
                std::vector<uint8_t> slab;

                for (int64_t t = 0; t < n_tensors; t++) {
                    const char *    name = gguf_get_tensor_name(ctx, t);
                    const ggml_type ty   = gguf_get_tensor_type(ctx, t);
                    const int64_t * ne   = gguf_get_tensor_ne(ctx, t);
                    if (filter && !strstr(name, filter))    continue;
                    if (ne[2] != 1 || ne[3] != 1)           continue;
                    if (ne[1] % slabch != 0 || ne[0] < 256) continue;

                    const ggml_type_traits * tr = ggml_get_type_traits(ty);
                    if (!tr->to_float)                      continue;
                    const int64_t K = ne[0], N = ne[1];
                    if (K % tr->blck_size != 0)             continue;
                    ggml_quantize_init(ty);

                    const size_t row_bytes = (size_t) (K / tr->blck_size) * tr->type_size;
                    const size_t slab_sz   = (size_t) slabch * 16 + (size_t) slabch * (size_t) K;
                    wbuf.resize((size_t) slabch * (size_t) K);
                    slab.resize(slab_sz);
                    raw.resize((size_t) slabch * row_bytes);

                    const std::string opath = std::string(emit_dir) + "/" + name + ".slab";
                    FILE * of = fopen(opath.c_str(), "wb");
                    if (!of) { fprintf(stderr, "  cannot write %s\n", opath.c_str()); continue; }

                    bool ok = true;
                    if (fseeko64(f, (off64_t) (data_off + gguf_get_tensor_offset(ctx, t)), SEEK_SET) != 0) ok = false;
                    for (int64_t s = 0; ok && s < N / slabch; s++) {
                        if (fread(raw.data(), 1, raw.size(), f) != raw.size()) { ok = false; break; }
                        for (int64_t z = 0; z < slabch; z++) {
                            tr->to_float(raw.data() + (size_t) z * row_bytes, wbuf.data() + z * K, K);
                        }
                        emit_slab_rows(wbuf.data(), slabch, K, slab.data());
                        if (fwrite(slab.data(), 1, slab_sz, of) != slab_sz) { ok = false; break; }
                    }
                    fclose(of);
                    if (!ok) { fprintf(stderr, "  FAILED on %s\n", name); continue; }

                    fprintf(mf, "%s\t%s\t%lld\t%lld\t%lld\t%lld\n", name, ggml_type_name(ty),
                            (long long) N, (long long) K, (long long) slabch,
                            (long long) (N * 16 + N * K));
                    printf("  %-34s %-8s %6lld x %-6lld -> %9.3f MB\n", name, ggml_type_name(ty),
                           (long long) N, (long long) K, (double) (N * 16 + N * K) / 1e6);
                    fflush(stdout);
                }
                fclose(mf);
                printf("  manifest: %s\n", mpath.c_str());
            }
        }
    }

    printf("\n=== DDR cost of each scheme (bits/weight, shown for K = 5120) ===\n");
    for (int s = 0; s < N_SCHEMES; s++) {
        printf("  %-9s %6.3f bpw   %s\n", SCHEMES[s].name, scheme_bpw(SCHEMES[s], 5120),
               SCHEMES[s].G == 0 ? "1 DPU workload" : "K/G DPU workloads + partial-sum accumulation");
    }

    fclose(f);
    gguf_free(ctx);
    return 0;
}
