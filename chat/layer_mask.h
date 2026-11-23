// Which transformer layers are full attention. Qwen3 (qwen35 == 0) is full attention on every layer.
// Qwen3.5 stores that in a 32-bit mask (full_mask). The shipped pattern repeats: bit i is set when
// (i + 1) is a multiple of 4 (0x88888888). A shift of 32 or more on a uint32_t is undefined, and
// Qwen3-14B has 40 layers while Qwen3.5-27B has 64, so the mask alone cannot name those layers.
// For a layer past the mask, continue the interval of the first 32 bits when they are a regular
// pattern. A mask that is not a regular interval reports layers >= 32 as not full.
#ifndef CHAT_LAYER_MASK_H
#define CHAT_LAYER_MASK_H

#include <stdint.h>

static inline int garlic_layer_full(int qwen35, uint32_t full_mask, uint32_t i) {
    if (!qwen35) return 1;
    if (i < 32) return (int)((full_mask >> i) & 1u);
    uint32_t interval = 0;
    for (uint32_t b = 0; b < 32; b++) {
        if ((full_mask >> b) & 1u) {
            interval = b + 1;
            break;
        }
    }
    if (!interval) return 0;
    for (uint32_t b = 0; b < 32; b++) {
        int expect = ((b + 1) % interval) == 0;
        int got = (int)((full_mask >> b) & 1u);
        if (expect != got) return 0;
    }
    return ((i + 1) % interval) == 0;
}

#endif
