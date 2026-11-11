// Speech playback on the PS4 (sceAudioOut, main port, 48 kHz): 24 kHz mono float samples are queued and played by
// a thread of their own, upsampled 2x by a half-band filter.
#pragma once
#include <stdint.h>

int aout_init(int32_t user_id);              // 0 if the port is open (idempotent)
void aout_push(const float* pcm, int n);     // append 24 kHz samples (waits while the queue is full)
// Gain for the samples pushed from now on (a voice's loudness, S3Voice.gain), with a limiter: a sample that would
// pass 0.99 (HiFT's own clamp) lowers the gain at once just enough, which then recovers in ~40 ms. At gain 1 HiFT's
// samples never pass 0.99, so they go through unchanged. Returns how many samples the limiter has touched so far.
int64_t aout_gain(float g);
void aout_clear(void);                       // drop everything queued (interrupt)
int64_t aout_played(void);                   // 24 kHz samples played since aout_init (for words as they are spoken)
int64_t aout_queued(void);                   // samples pushed since aout_init
float aout_level(void);                      // 0..1, loudness of what is playing now
