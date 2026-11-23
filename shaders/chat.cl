// Everything the chat app runs on the GPU: the Qwen3.5 decoder (qwen35.cl), Parakeet speech to text
// (parakeet.cl) and Chatterbox's T3 (t3.cl) and S3Gen + HiFT
// (s3gen.cl), compiled together into assets/kernels.bin.

#include "qwen35.cl"
#include "parakeet.cl"
#include "t3.cl"
#include "s3gen.cl"
