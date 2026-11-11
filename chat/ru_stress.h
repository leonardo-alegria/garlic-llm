// Russian stress marks for Chatterbox's text (assets/ru_stress.bin).
// As RussianTextStresser, which the official tokenizer calls for "ru", does for unambiguous word forms: U+0301
// after the stressed vowel, and е -> ё where the word has ё. Words not in the file (rare forms, homographs such as
// замок / все / стоит) stay as they are. Call it after punc_norm and BEFORE NFKD.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct RuStress {
    uint32_t n_keys, n_blocks;
    const uint8_t* index;   // n_blocks x u32 LE
    const uint8_t* data;
    size_t data_size;
} RuStress;

int rustress_load(RuStress* d, const uint8_t* blob, size_t size);   // keeps pointers into blob; 0 if fine
// Stresses UTF-8 text into out (cap bytes, NUL-terminated); cap >= 2 * strlen(in) + 1 is always enough.
size_t rustress_text(const RuStress* d, const char* in, char* out, size_t cap);
