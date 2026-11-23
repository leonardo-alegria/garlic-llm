// Host tests for chat/layer_mask.h. Layers at index 32 and above must not shift a uint32_t by >= 32.
// Build and run: gcc -std=c99 -Wall -Wextra -g -I chat tests/layer_mask_test.c -o build/layer_mask_test
#include <stdio.h>

#include "layer_mask.h"

static int fails, checks;
#define CHECK(c)                                                              \
    do {                                                                      \
        checks++;                                                             \
        if (!(c)) {                                                           \
            fails++;                                                          \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);               \
        }                                                                     \
    } while (0)

int main(void) {
    // Qwen3 (PMDL): every layer is full attention, including layer 39 of Qwen3-14B.
    for (uint32_t i = 0; i < 64; i++) CHECK(garlic_layer_full(0, 0xFFFFFFFFu, i) == 1);
    for (uint32_t i = 0; i < 64; i++) CHECK(garlic_layer_full(0, 0, i) == 1);

    // Qwen3.5: full when (i + 1) % 4 == 0. The stored mask is 0x88888888 (bits 3, 7, ..., 31).
    // Layer 35 and layer 63 are past the mask and must still be full; layer 32 must not.
    const uint32_t q35 = 0x88888888u;
    for (uint32_t i = 0; i < 64; i++) {
        int want = ((i + 1) % 4) == 0;
        CHECK(garlic_layer_full(1, q35, i) == want);
    }

    // All-full and all-DeltaNet masks extend.
    for (uint32_t i = 0; i < 64; i++) CHECK(garlic_layer_full(1, 0xFFFFFFFFu, i) == 1);
    for (uint32_t i = 0; i < 64; i++) CHECK(garlic_layer_full(1, 0, i) == 0);

    // Every 8th layer (bits 7, 15, 23, 31). Layer 39 is the next one; layer 32 is not.
    const uint32_t every8 = 0x80808080u;
    CHECK(garlic_layer_full(1, every8, 7) == 1);
    CHECK(garlic_layer_full(1, every8, 8) == 0);
    CHECK(garlic_layer_full(1, every8, 31) == 1);
    CHECK(garlic_layer_full(1, every8, 32) == 0);
    CHECK(garlic_layer_full(1, every8, 39) == 1);
    CHECK(garlic_layer_full(1, every8, 47) == 1);

    // Only layer 0 is full. That is not a repeating interval, so layers >= 32 stay not full.
    CHECK(garlic_layer_full(1, 0x1u, 0) == 1);
    CHECK(garlic_layer_full(1, 0x1u, 1) == 0);
    CHECK(garlic_layer_full(1, 0x1u, 32) == 0);
    CHECK(garlic_layer_full(1, 0x1u, 63) == 0);

    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
