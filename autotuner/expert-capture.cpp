// ExpertAtlas — expert-activation capture for MoE models on llama.cpp.
//
// Hooks llama.cpp's graph eval callback and records, for every token, which of
// the model's experts the router selected (the "ffn_moe_topk" tensor, per layer).
// Aggregates into a (layer, expert) -> count table and writes it as CSV.
//
// The point: on real workloads MoE routing is heavily skewed (~15-20% of experts
// serve ~80% of tokens). Measuring that skew per workload lets ExpertAtlas keep the
// hot experts resident in RAM and push the cold ones to NVMe — fitting a
// trillion-parameter model on a smaller, cheaper Arm VM.
//
// Build:  see build.sh (links against the existing /opt/llama.cpp build).
// Run:    ./expert-capture -m <model.gguf> -f prompts.txt -o activations.csv

#include "llama.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <fstream>

struct capture_state {
    // key = (layer << 20) | expert_id  -> activation count
    std::map<uint64_t, uint64_t> counts;
    int n_expert_used = 0;
    long long tokens_seen = 0;
};

// Called for every graph node, twice: ask=true (do you want this tensor?),
// then ask=false (here is the computed data). CPU backend => t->data is host memory.
static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<capture_state *>(user_data);

    const bool is_topk = std::strncmp(t->name, "ffn_moe_topk", 12) == 0;
    if (ask) {
        return is_topk;               // only ask for the router's selected-experts tensor
    }
    if (!is_topk || t->type != GGML_TYPE_I32 || t->data == nullptr) {
        return true;
    }

    // shape: ne[0] = n_expert_used (top-k), ne[1] = n_tokens in this batch
    const int64_t n_used   = t->ne[0];
    const int64_t n_tokens = t->ne[1];
    const int layer = ggml_get_name(t)[12] ? atoi(ggml_get_name(t) + 13) : 0; // "ffn_moe_topk-<il>"

    const int32_t * ids = static_cast<const int32_t *>(t->data);
    for (int64_t tok = 0; tok < n_tokens; ++tok) {
        for (int64_t k = 0; k < n_used; ++k) {
            const int32_t expert = ids[tok * n_used + k];
            st->counts[((uint64_t) layer << 20) | (uint32_t) expert]++;
        }
    }
    st->n_expert_used = (int) n_used;
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, prompts_path, out_path = "activations.csv";
    int n_ctx = 8192, n_threads = 64;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if      (a == "-m") model_path   = next();
        else if (a == "-f") prompts_path = next();
        else if (a == "-o") out_path     = next();
        else if (a == "-c") n_ctx        = std::stoi(next());
        else if (a == "-t") n_threads    = std::stoi(next());
    }
    if (model_path.empty() || prompts_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -f prompts.txt [-o out.csv] [-c ctx] [-t threads]\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    capture_state st;

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx            = n_ctx;
    cparams.n_batch          = n_ctx;
    cparams.n_threads        = n_threads;
    cparams.n_threads_batch  = n_threads;
    cparams.cb_eval          = eval_callback;
    cparams.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    // Feed each prompt through a single forward pass (prefill). That exercises the
    // router over real workload tokens — no generation needed for activation stats.
    std::ifstream in(prompts_path);
    std::string line;
    int prompt_idx = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<llama_token> toks(line.size() + 8);
        int n = llama_tokenize(vocab, line.c_str(), (int) line.size(),
                               toks.data(), (int) toks.size(), true, false);
        if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, line.c_str(), (int) line.size(),
                               toks.data(), (int) toks.size(), true, false); }
        toks.resize(n > 0 ? n : 0);
        if (toks.empty()) continue;
        if ((int) toks.size() > n_ctx) toks.resize(n_ctx);

        llama_batch batch = llama_batch_get_one(toks.data(), (int) toks.size());
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed on prompt %d\n", prompt_idx); }
        llama_memory_clear(llama_get_memory(ctx), true);   // reset KV between prompts
        st.tokens_seen += toks.size();
        fprintf(stderr, "\rprocessed prompt %d (%lld tokens)   ", ++prompt_idx, st.tokens_seen);
    }
    fprintf(stderr, "\n");

    std::ofstream out(out_path);
    out << "layer,expert,count\n";
    for (auto & [key, cnt] : st.counts) {
        out << (int) (key >> 20) << "," << (int) (key & 0xFFFFF) << "," << cnt << "\n";
    }
    fprintf(stderr, "wrote %s  (%zu layer/expert cells, top-k=%d, %lld tokens over %d prompts)\n",
            out_path.c_str(), st.counts.size(), st.n_expert_used, st.tokens_seen, prompt_idx);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
