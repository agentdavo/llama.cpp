#!/usr/bin/env python3
"""
make_tiny_q8_0_gguf.py — synthesize a tiny but VALID llama-arch GGUF with Q8_0 weights, no download.

Its only purpose is to give `llama-bench`/`llama-cli --device hpi-3720 -ngl 999` a real model whose
weight matmuls are Q8_0, so the ggml-npu backend actually executes them (and logs it). Weights are
random; output is garbage — this proves the offload path runs, not model quality.

Usage:  python3 make_tiny_q8_0_gguf.py [out.gguf]
Requires: numpy, and gguf-py on sys.path (this file lives inside the llama.cpp tree).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "gguf-py"))
import numpy as np
import gguf
from gguf import GGUFWriter, GGMLQuantizationType
from gguf.quants import quantize

# tiny dims; every Q8_0 tensor's row length (ne0) is a multiple of 32
N_VOCAB, N_EMBD, N_HEAD, N_HEAD_KV, N_LAYER, N_FF, N_CTX = 32, 64, 4, 4, 2, 128, 512
HEAD_DIM = N_EMBD // N_HEAD                       # 16
N_EMBD_KV = N_HEAD_KV * HEAD_DIM                  # 64
RMS_EPS = 1e-5
rng = np.random.default_rng(1234)

def rnd(*shape):
    return (rng.standard_normal(shape).astype(np.float32) * 0.02)

def q8(name, arr):                                # add a Q8_0 tensor (arr shape = [out, in], in % 32 == 0)
    assert arr.shape[-1] % 32 == 0, (name, arr.shape)
    w.add_tensor(name, quantize(np.ascontiguousarray(arr), GGMLQuantizationType.Q8_0),
                 raw_dtype=GGMLQuantizationType.Q8_0)

def f32(name, arr):
    w.add_tensor(name, np.ascontiguousarray(arr.astype(np.float32)))

out = sys.argv[1] if len(sys.argv) > 1 else "tiny-npu-q8_0.gguf"
w = GGUFWriter(out, "llama")

# --- metadata (llama arch; general.architecture is written by the GGUFWriter ctor) ---
w.add_string("general.name", "tiny-npu-test")
w.add_uint32("general.file_type", 7)             # Q8_0
w.add_uint32("llama.context_length", N_CTX)
w.add_uint32("llama.embedding_length", N_EMBD)
w.add_uint32("llama.block_count", N_LAYER)
w.add_uint32("llama.feed_forward_length", N_FF)
w.add_uint32("llama.attention.head_count", N_HEAD)
w.add_uint32("llama.attention.head_count_kv", N_HEAD_KV)
w.add_uint32("llama.rope.dimension_count", HEAD_DIM)
w.add_float32("llama.attention.layer_norm_rms_epsilon", RMS_EPS)
w.add_uint32("llama.vocab_size", N_VOCAB)

# --- minimal tokenizer (llama-bench uses random token ids, never tokenizes text) ---
tokens = ["<unk>", "<s>", "</s>"] + [f"tok{i}" for i in range(N_VOCAB - 3)]
scores = [0.0] * N_VOCAB
ttypes = [2, 3, 3] + [1] * (N_VOCAB - 3)         # UNKNOWN, CONTROL, CONTROL, then NORMAL
w.add_tokenizer_model("llama")
w.add_token_list(tokens)
w.add_token_scores(scores)
w.add_token_types(ttypes)
w.add_bos_token_id(1)
w.add_eos_token_id(2)
w.add_unk_token_id(0)
w.add_add_bos_token(True)

# --- tensors ---
q8("token_embd.weight", rnd(N_VOCAB, N_EMBD))
f32("output_norm.weight", np.ones(N_EMBD))
q8("output.weight", rnd(N_VOCAB, N_EMBD))
for i in range(N_LAYER):
    p = f"blk.{i}."
    f32(p + "attn_norm.weight", np.ones(N_EMBD))
    q8(p + "attn_q.weight",      rnd(N_EMBD,    N_EMBD))
    q8(p + "attn_k.weight",      rnd(N_EMBD_KV, N_EMBD))
    q8(p + "attn_v.weight",      rnd(N_EMBD_KV, N_EMBD))
    q8(p + "attn_output.weight", rnd(N_EMBD,    N_EMBD))
    f32(p + "ffn_norm.weight", np.ones(N_EMBD))
    q8(p + "ffn_gate.weight", rnd(N_FF,   N_EMBD))
    q8(p + "ffn_up.weight",   rnd(N_FF,   N_EMBD))
    q8(p + "ffn_down.weight", rnd(N_EMBD, N_FF))

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {out}: llama {N_LAYER}L d{N_EMBD} ff{N_FF} vocab{N_VOCAB}, weights Q8_0")
