// Qwen3 decode-step kernels for the PS4 GPU (GCN gfx701), compiled by LLVM from OpenCL C 1.2 without its
// standard headers. One token per command buffer step: embed -> layers x (norm, qkv, q/k norm + RoPE + cache,
// attention, o, add, norm, gate/up, SiLU x up, down, add) -> final norm -> logits -> sample. The next token never
// leaves the GPU.

#define HEAD_DIM 128
#define ROPE_DIM 128
#define MAX_SEQ 2048

#include "common.h"
#include "attn.h"
