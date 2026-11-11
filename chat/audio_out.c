// See audio_out.h. One producer (the voice pipeline) and one consumer (the playback thread) share a ring of
// 24 kHz samples; the consumer always outputs (silence when the ring is empty), so the port's timing is steady.
#include "audio_out.h"

#include <math.h>
#include <orbis/AudioOut.h>
#include <orbis/libkernel.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define RATE_IN 24000
#define CHUNK 128                      // input samples per output buffer (256 at 48 kHz: 5.3 ms)
#define RING (1 << 19)                 // 21.8 s at 24 kHz (static memory comes out of the flexible budget)
#define TAPS 8                         // half-band filter: 8 taps per side

static float ring[RING];
static volatile int64_t wr, rd;       // total samples written / consumed
static volatile int clear_req, level_q;
static int port = -1, stereo;
static float hb[TAPS];
static OrbisPthread th;


static void* play_main(void* arg) {
    (void)arg;
    static float w[CHUNK + 2 * TAPS];   // the last 2 TAPS - 1 samples of the previous chunk, then this chunk
    static int16_t out[2 * CHUNK * 2];
    memset(w, 0, sizeof(w));
    for (;;) {
        if (clear_req) {
            rd = wr;
            memset(w, 0, sizeof(w));
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            clear_req = 0;
        }
        int64_t avail = wr - rd;
        int n = avail < CHUNK ? (int)avail : CHUNK;
        memmove(w, w + CHUNK, (2 * TAPS - 1) * sizeof(float));
        for (int i = 0; i < CHUNK; i++) w[2 * TAPS - 1 + i] = i < n ? ring[(rd + i) & (RING - 1)] : 0.0f;
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
        rd += n;
        // sample i of this chunk sits at w[i + 2 TAPS - 1]; output the one TAPS - 1 earlier (filter delay) and the
        // half-way point after it
        double ss = 0;
        for (int i = 0; i < CHUNK; i++) {
            const float* c = w + i + TAPS - 1;   // c[0] = x[t], c[1] = x[t + 1]
            float a = c[0], b = 0.0f;
            for (int k = 0; k < TAPS; k++) b += hb[k] * (c[-k] + c[1 + k]);
            float s[2] = {a, b};
            for (int h = 0; h < 2; h++) {
                float v = s[h] * 32767.0f;
                int16_t q = (int16_t)(v > 32767.0f ? 32767 : v < -32768.0f ? -32768 : (int)lrintf(v));
                if (stereo) out[4 * i + 2 * h] = out[4 * i + 2 * h + 1] = q;
                else out[2 * i + h] = q;
            }
            ss += (double)a * a;
        }
        float rms = (float)sqrt(ss / CHUNK);
        float db = rms > 0 ? 20.0f * log10f(rms) : -100.0f;
        float lv = (db + 50.0f) / 40.0f;
        level_q = (int)(1000 * (lv < 0 ? 0 : lv > 1 ? 1 : lv));
        sceAudioOutOutput(port, out);   // blocks until the previous buffer has started playing
    }
    return NULL;
}

int aout_init(int32_t user_id) {
    if (port >= 0) return 0;
    void say(const char* fmt, ...);
    int ri = sceAudioOutInit();   // (an error if already initialised: harmless)
    // the main port belongs to the system user (255); the console user as a fallback; mono, else stereo
    int32_t users[2] = {255, user_id};
    for (int u = 0; u < 2 && port < 0; u++)
        for (int st = 0; st < 2 && port < 0; st++) {
            port = sceAudioOutOpen(users[u], ORBIS_AUDIO_OUT_PORT_TYPE_MAIN, 0, 2 * CHUNK, 48000,
                                   st ? ORBIS_AUDIO_OUT_PARAM_FORMAT_S16_STEREO : ORBIS_AUDIO_OUT_PARAM_FORMAT_S16_MONO);
            stereo = st;
            if (port < 0) say("audio out: open (user %d, %s) -> 0x%08x", users[u], st ? "stereo" : "mono", (unsigned)port);
        }
    if (port < 0) {
        say("audio out: init 0x%08x, no port", (unsigned)ri);
        return port;
    }
    say("audio out: port %d (%s)", port, stereo ? "stereo" : "mono");
    // value half-way between two samples: sinc(k + 1/2) under a Blackman window, normalised to a gain of 1
    double sum = 0;
    for (int k = 0; k < TAPS; k++) {
        double x = k + 0.5, wn = 0.42 + 0.5 * cos(M_PI * x / (TAPS + 0.5)) + 0.08 * cos(2 * M_PI * x / (TAPS + 0.5));
        hb[k] = (float)(sin(M_PI * x) / (M_PI * x) * wn);
        sum += 2 * hb[k];
    }
    for (int k = 0; k < TAPS; k++) hb[k] = (float)(hb[k] / sum);
    int32_t vol[8] = {32768, 32768, 32768, 32768, 32768, 32768, 32768, 32768};   // SCE_AUDIO_VOLUME_0DB
    sceAudioOutSetVolume(port, 0xFF, vol);
    OrbisPthreadAttr attr;
    scePthreadAttrInit(&attr);
    scePthreadAttrSetstacksize(&attr, 256 << 10);
    if (scePthreadCreate(&th, &attr, play_main, NULL, "garlic_speak")) {
        sceAudioOutClose(port);
        port = -1;
        return -1;
    }
    scePthreadSetprio(th, 256 + 64);   // above normal: the speech must not stutter while the GPU is driven
    return 0;
}

static float gain = 1.0f, lim = 1.0f;   // the voice's gain; the limiter's (<= 1)
static int64_t limited;

int64_t aout_gain(float g) {
    gain = g;
    return limited;
}

void aout_push(const float* pcm, int n) {
    int64_t w0 = wr;
    for (int i = 0; i < n; i++) {
        while (w0 + i - rd >= RING - 1) {   // full: publish what is there and wait for the player
            wr = w0 + i;
            sceKernelUsleep(5000);
        }
        float y = pcm[i] * gain, a = fabsf(y);
        if (a * lim > 0.99f) lim = 0.99f / a, limited++;
        y *= lim;
        lim = 1.0f - (1.0f - lim) * 0.99896f;   // release: time constant 960 samples (40 ms)
        ring[(w0 + i) & (RING - 1)] = y;
        if ((i & 1023) == 1023 || i == n - 1) {
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            wr = w0 + i + 1;
        }
    }
}

void aout_clear(void) {
    if (port < 0) return;
    clear_req = 1;
    for (int i = 0; i < 40 && clear_req; i++) sceKernelUsleep(1000);   // the thread drops the rest within ~5 ms
}

int64_t aout_played(void) { return rd; }
int64_t aout_queued(void) { return wr; }
float aout_level(void) { return level_q / 1000.0f; }

