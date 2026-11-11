// Qwen byte-level BPE tokenizer (reads assets/tokenizer.bin written by pc/export_chat.py).
#ifndef CHAT_TOKENIZER_H
#define CHAT_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t n_vocab, n_merges;
    const uint32_t* byte_id;     // [256] token id of each raw byte
    const uint8_t** raw;         // [n_vocab] raw bytes of each token
    uint16_t* raw_len;           // [n_vocab]
    uint8_t* special;            // [n_vocab] 1 for added/special tokens
    uint64_t* keys;              // merge hash: (a << 32 | b) + 1, 0 = empty
    uint32_t* rank;              // merge rank (lower merges first)
    uint32_t* merged;            // resulting token id
    uint32_t mask;
} Tokenizer;

// data must stay alive while the tokenizer is used; returns 0 on success
int tok_load(Tokenizer* t, const uint8_t* data, size_t size);
// plain text (no special tokens) -> ids; returns the number of ids written (at most max)
int tok_encode(const Tokenizer* t, const char* utf8, uint32_t* out, int max);
// raw bytes of one token (may be part of a UTF-8 character); special tokens give 0 bytes
int tok_piece(const Tokenizer* t, uint32_t id, const uint8_t** bytes);

#endif
