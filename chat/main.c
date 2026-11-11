// Garlic LLM: chat with Qwen3.5 running on the PS4's own GPU, on the TV, with the controller; speech to text with
// Parakeet. This file is the console side (screen, controller, microphone, the conversation with the engine);
// everything that draws is in ui.c.
//
// Chat: Cross = keyboard, hold Square = talk, Circle = stop the answer, Triangle = new conversation,
// L2 / R2 = temperature, Up / Down = scroll, OPTIONS = system keyboard (e.g. a USB keyboard), R3 / touch pad =
// think before answering on / off.
// The model weights live in /data (see model.idx); GPU code, tokenizer and fonts are in the package's assets.

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <orbis/CommonDialog.h>
#include <orbis/ImeDialog.h>
#include <orbis/Pad.h>
#include <orbis/Sysmodule.h>
#include <orbis/SystemService.h>
#include <orbis/UserService.h>
#include <orbis/VideoOut.h>
#include <orbis/libkernel.h>

#include "engine.h"
#include "tokenizer.h"
#include "pyrun.h"
#include "ui.h"

#define ASSETS "/app0/assets"
#define IM_START eng.im_start   // <|im_start|> / <|im_end|> of the loaded model (model.idx)
#define IM_END eng.im_end
#define THINK 248068u           // Qwen3.5 <think> and </think>: an empty block = answer without thinking
#define THINK_END 248069u
#define MODEL_NAME (eng.qwen35 ? (eng.hidden >= 4096 ? "Qwen3.5-9B" : "Qwen3.5-4B") : "Qwen3-4B")
#define MAX_REPLY 3000
// One log per app (so that opening the other app does not lose this one's log), and the one before it
#if defined(GARLIC_4B)
#define LOG_NAME "/data/ps4ai_log_4b"
#else
#define LOG_NAME "/data/ps4ai_log"
#endif
#define LOG_FILE LOG_NAME ".txt"
#define LOG_PREV LOG_NAME "_prev.txt"
#define VOICE_FILE "/data/voz.wav"   // used instead of the microphone when there is none (testing)
#define KEYS_FILE "/data/garlic_keys.txt"   // test aid: buttons, questions and screenshots from a file (inj_buttons)

uint32_t* back;

// Log to the kernel log and to LOG_FILE (synced every line, so it survives a crash; read it over FTP).
void say(const char* fmt, ...) {   // a line in the log, with the seconds since the app started
    char buf[420];
    int h = snprintf(buf, 32, "[garlic %8.3f] ", sceKernelGetProcessTime() / 1e6);
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf + h, sizeof(buf) - h - 2, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if (n > (int)sizeof(buf) - h - 2) n = sizeof(buf) - h - 2;
    buf[h + n] = '\n';
    buf[h + n + 1] = 0;
    sceKernelDebugOutText(0, buf);
    int f = open(LOG_FILE, O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (f >= 0) {
        write(f, buf, h + n + 1);
        fsync(f);
        close(f);
    }
}

uint8_t* read_all(const char* path, size_t* size) {
    int f = open(path, O_RDONLY);
    if (f < 0) return NULL;
    off_t n = lseek(f, 0, SEEK_END);
    lseek(f, 0, SEEK_SET);
    uint8_t* buf = malloc(n);
    size_t done = 0;
    while (buf && done < (size_t)n) {
        ssize_t r = read(f, buf + done, n - done);
        if (r <= 0) break;
        done += r;
    }
    close(f);
    *size = done;
    return buf;
}

static double now(void) { return (double)sceKernelReadTsc() / sceKernelGetTscFrequency(); }

// ---- screen ------------------------------------------------------------------------------------------------------

static int video;
static OrbisKernelEqueue flip_queue;
static uint32_t* fb[2];
static int cur, frame = 1;

static int video_init(void) {
    video = sceVideoOutOpen(ORBIS_VIDEO_USER_MAIN, ORBIS_VIDEO_OUT_BUS_MAIN, 0, 0);
    if (video < 0) return -1;
    sceKernelCreateEqueue(&flip_queue, "garlic flip");
    sceVideoOutAddFlipEvent(flip_queue, video, 0);
    size_t size = (2ull * SCREEN_W * SCREEN_H * 4 + 0x1FFFFF) & ~0x1FFFFFull;
    off_t off;
    void* mem = NULL;
    if (sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), size, 0x200000, 3, &off)) return -2;
    if (sceKernelMapDirectMemory(&mem, size, 0x33, 0, off, 0x200000)) return -3;
    fb[0] = mem;
    fb[1] = (uint32_t*)mem + SCREEN_W * SCREEN_H;
    OrbisVideoOutBufferAttribute attr;
    sceVideoOutSetBufferAttribute(&attr, 0x80000000, 1, 0, SCREEN_W, SCREEN_H, SCREEN_W);
    if (sceVideoOutRegisterBuffers(video, 0, (void**)fb, 2, &attr)) return -4;
    sceVideoOutSetFlipRate(video, 0);
    back = ui_screen_buf();   // drawn in cached memory, then copied to the write-combined frame buffer
    return back ? 0 : -5;
}

static void present(void) {
    memcpy(fb[cur], back, SCREEN_W * SCREEN_H * 4);
    sceVideoOutSubmitFlip(video, cur, ORBIS_VIDEO_OUT_FLIP_VSYNC, frame);
    for (;;) {
        OrbisVideoOutFlipStatus st;
        sceVideoOutGetFlipStatus(video, &st);
        if (st.flipArg >= frame) break;
        OrbisKernelEvent ev;
        int count;
        if (sceKernelWaitEqueue(flip_queue, &ev, 1, &count, 0)) break;
    }
    cur ^= 1;
    frame++;
}

// ---- controller --------------------------------------------------------------------------------------------------

static int g_pad = -1;

// Test aid: /data/garlic_keys.txt drives the app without a controller. One command per line: a button
// (X O SQ TRI L1 L2 L3 R1 R2 R3 OPT TOUCH UP DOWN LEFT RIGHT), held for 0.15 s or ":seconds"
// ("SQ:3" records 3 s of dictation, from /data/voz.wav without a microphone); "wait <seconds>"; "ask <text>" (sent as
// if typed, when the chat screen is idle); "shot" (the screen -> /data/garlic_shot.bmp, 960x540) or "shot full" (1920x1080, garlic_shot_full.bmp); "#" a comment.
// The file is renamed garlic_keys.done.txt when read; the next one is read when this one is done.
static char* inj_buf;
static size_t inj_len, inj_at;
static uint32_t inj_mask;
static double inj_until, inj_check;
static char inj_ask[2048];
static volatile int inj_ask_ready;

static OrbisPthreadMutex ts_mu, ts_msg_mu;   // (the chat screen's thread, below: it draws while it holds ts_msg_mu)
static volatile int ts_run;

static void screenshot(void) {   // `back` (the last frame drawn), halved -> BMP
    enum { W = 960, H = 540 };
    int locked = ts_run && scePthreadMutexLock(&ts_msg_mu) == 0;   // (not half drawn)
    static uint8_t img[54 + W * H * 3];
    uint8_t* h = img;
    memset(h, 0, 54);
    h[0] = 'B', h[1] = 'M';
    uint32_t v[] = {sizeof(img), 0, 54, 40, W, (uint32_t)-H};
    memcpy(h + 2, &v[0], 4), memcpy(h + 10, &v[2], 4), memcpy(h + 14, &v[3], 4), memcpy(h + 18, &v[4], 4);
    memcpy(h + 22, &v[5], 4);
    h[26] = 1, h[28] = 24;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const uint32_t* s = back + (2 * y) * SCREEN_W + 2 * x;
            uint32_t p[4] = {s[0], s[1], s[SCREEN_W], s[SCREEN_W + 1]};
            uint8_t* d = img + 54 + (y * W + x) * 3;
            for (int c = 0; c < 3; c++)
                d[c] = (uint8_t)((((p[0] >> 8 * c) & 255) + ((p[1] >> 8 * c) & 255) + ((p[2] >> 8 * c) & 255) +
                                  ((p[3] >> 8 * c) & 255) + 2) / 4);
        }
    if (locked) scePthreadMutexUnlock(&ts_msg_mu);
    int f = open("/data/garlic_shot.bmp", O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (f >= 0) write(f, img, sizeof(img)), close(f);
    say("keys: screenshot in /data/garlic_shot.bmp");
}

static void screenshot_full(void) {   // `back` at its full size -> BMP, written row by row (no large buffer)
    int locked = ts_run && scePthreadMutexLock(&ts_msg_mu) == 0;
    int f = open("/data/garlic_shot_full.bmp", O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (f >= 0) {
        uint8_t h[54] = {0};
        static uint8_t row[SCREEN_W * 3];
        uint32_t v[] = {54 + SCREEN_W * SCREEN_H * 3, 0, 54, 40, SCREEN_W, (uint32_t)-SCREEN_H};
        h[0] = 'B', h[1] = 'M';
        memcpy(h + 2, &v[0], 4), memcpy(h + 10, &v[2], 4), memcpy(h + 14, &v[3], 4), memcpy(h + 18, &v[4], 4);
        memcpy(h + 22, &v[5], 4);
        h[26] = 1, h[28] = 24;
        write(f, h, 54);
        for (int y = 0; y < SCREEN_H; y++) {
            const uint32_t* s = back + y * SCREEN_W;
            for (int x = 0; x < SCREEN_W; x++) row[3 * x] = s[x] & 255, row[3 * x + 1] = (s[x] >> 8) & 255, row[3 * x + 2] = (s[x] >> 16) & 255;
            write(f, row, sizeof(row));
        }
        close(f);
    }
    if (locked) scePthreadMutexUnlock(&ts_msg_mu);
    say("keys: screenshot in /data/garlic_shot_full.bmp");
}

static uint32_t inj_buttons(void) {
    double t = now();
    if (inj_until) {
        if (t < inj_until) return inj_mask;
        if (inj_mask) {   // released a moment before the next command
            inj_mask = 0, inj_until = t + 0.15;
            return 0;
        }
        inj_until = 0;
    }
    if (inj_ask_ready) return 0;   // (the main loop sends it)
    if (!inj_buf) {
        if (t < inj_check) return 0;
        inj_check = t + 0.5;
        size_t n = 0;
        char* f = (char*)read_all(KEYS_FILE, &n);
        if (!f) return 0;
        rename(KEYS_FILE, "/data/garlic_keys.done.txt");
        if (!n) {
            free(f);
            return 0;
        }
        inj_buf = f, inj_len = n, inj_at = 0;
    }
    static const struct {
        const char* name;
        uint32_t bit;
    } names[] = {{"X", ORBIS_PAD_BUTTON_CROSS},     {"O", ORBIS_PAD_BUTTON_CIRCLE},  {"SQ", ORBIS_PAD_BUTTON_SQUARE},
                 {"TRI", ORBIS_PAD_BUTTON_TRIANGLE}, {"L1", ORBIS_PAD_BUTTON_L1},     {"L2", ORBIS_PAD_BUTTON_L2},
                 {"L3", ORBIS_PAD_BUTTON_L3},        {"R1", ORBIS_PAD_BUTTON_R1},     {"R2", ORBIS_PAD_BUTTON_R2},
                 {"R3", ORBIS_PAD_BUTTON_R3},        {"OPT", ORBIS_PAD_BUTTON_OPTIONS}, {"TOUCH", ORBIS_PAD_BUTTON_TOUCH_PAD},
                 {"UP", ORBIS_PAD_BUTTON_UP},        {"DOWN", ORBIS_PAD_BUTTON_DOWN}, {"LEFT", ORBIS_PAD_BUTTON_LEFT},
                 {"RIGHT", ORBIS_PAD_BUTTON_RIGHT}};
    while (inj_at < inj_len) {
        size_t e = inj_at;
        while (e < inj_len && inj_buf[e] != '\n') e++;
        char line[2048];
        size_t l = e - inj_at < sizeof(line) - 1 ? e - inj_at : sizeof(line) - 1;
        memcpy(line, inj_buf + inj_at, l);
        inj_at = e + 1;
        while (l && (line[l - 1] == '\r' || line[l - 1] == ' ')) l--;
        line[l] = 0;
        if (!l || line[0] == '#') continue;
        say("keys: %.200s", line);
        if (!strncmp(line, "wait", 4)) {
            inj_until = t + atof(line + 4);
            return 0;
        }
        if (!strncmp(line, "ask ", 4)) {
            snprintf(inj_ask, sizeof(inj_ask), "%s", line + 4);
            inj_ask_ready = 1;
            return 0;
        }
        if (!strcmp(line, "shot full")) {
            screenshot_full();
            continue;
        }
        if (!strcmp(line, "shot")) {
            screenshot();
            continue;
        }
        char* colon = strchr(line, ':');
        double hold = colon ? atof(colon + 1) : 0.15;
        if (colon) *colon = 0;
        for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++)
            if (!strcmp(line, names[i].name)) {
                inj_mask = names[i].bit, inj_until = t + (hold > 0 ? hold : 0.15);
                return inj_mask;
            }
        say("keys: unknown command");
    }
    free(inj_buf);
    inj_buf = NULL;
    say("keys: done");
    return 0;
}

static uint32_t read_buttons(void) {
    OrbisPadData pd;
    memset(&pd, 0, sizeof(pd));
    uint32_t inj = inj_buttons();
    if (g_pad < 0 || scePadReadState(g_pad, &pd) != 0) return inj;
    uint32_t b = pd.buttons | inj;
    if (pd.leftStick.x < 50) b |= ORBIS_PAD_BUTTON_LEFT;   // the left stick moves like the D-pad
    if (pd.leftStick.x > 205) b |= ORBIS_PAD_BUTTON_RIGHT;
    if (pd.leftStick.y < 50) b |= ORBIS_PAD_BUTTON_UP;
    if (pd.leftStick.y > 205) b |= ORBIS_PAD_BUTTON_DOWN;
    return b;
}

// ---- state -------------------------------------------------------------------------------------------------------

static Engine eng;
static Tokenizer tok;
static ChatView view;
static int pos, generating, need_close, reply_tokens, user_id;
static double reply_t0;   // when the reply's first token came (wall clock: the speed as seen, screen included)
static int cut_close;   // the reply to close was stopped, not ended by the model: close it with "…"
static uint32_t next_token, seed = 12345;
static uint8_t pending[8];
static int n_pending;
static const char* toast_text;
static int in_voice_mode;   // the voice mode screen is showing (its own thread draws it, see vs_main)
static void show_voice(int state);
static void tts_new_reply(void);
static void vs_publish(int think, int listen, float level);
static void vs_stop(void);
static double toast_until;
#define THINK_T 0.7f
#define THINK_CLOSE "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n"
static int think_on, think_cap = 1024;
static int think_turn;     // the reply in progress began with reasoning...
static int thinking;       // ...and is still reasoning (think_n tokens so far, up to turn_cap)
static int think_n, turn_cap, think_cut, think_saved, think_pos0, think_answer;
static double think_t_ask, think_t0, think_t1;
static char think_text[32768];
static size_t think_len;
static uint32_t hist_ids[MAX_REPLY + 16];   // the turn as the model keeps it: an empty <think></think>, the answer
static int hist_n, hist_done;
static void start_turn(const char* user_text);

static void toast(StrId id) {
    toast_text = tr(id);
    toast_until = now() + 3.0;
    say("toast: %s", toast_text);
}

// The newest reply with a complete ```python block (R1 runs it); -1 if none.
static int code_msg(void) {
    for (int i = n_msgs - 1; i >= 0 && i >= n_msgs - 6; i--) {
        if (msgs[i].who != MSG_AI) continue;
        const char* t = msgs[i].text;
        const char* f = strstr(t, "```py");
        if (f && strstr(f + 5, "\n```")) return i;
    }
    return -1;
}

static void view_update(void) {
    view.model = MODEL_NAME;
    view.pos = pos;
    view.max_seq = (int)eng.max_seq;
    view.can_run = !generating && code_msg() >= 0;
    view.generating = generating;
    view.time = now();
    double left = toast_until - view.time;
    view.toast = left > 0 ? toast_text : NULL;
    view.toast_alpha = left > 0.4 ? 1.0f : left > 0 ? (float)(left / 0.4) : 0.0f;
    view.think_avail = eng.qwen35;
    view.think_on = think_on;
    view.think_cap = thinking ? turn_cap : think_cap;
    view.thinking = thinking;
    view.think_n = think_n;
    view.think_s = thinking ? view.time - think_t0 : 0;
    view.think_text = think_text;
    view.think_len = think_len;
    if (!generating) view.pos = pos + (hist_n - hist_done);   // (the answer still to read again counts)
}

// While a reply is generated on the chat screen (the reasoning and the answer), a thread of its own draws the screen,
// as in voice mode (vs_main). Drawn by the main loop, one token per frame, every token would also wait for a frame's
// drawing and flip, which slows the generation down; several tokens per frame would make the screen stutter.
// Meanwhile the main loop only publishes the view (show_chat) under ts_mu, the reasoning is only appended (past the
// published think_len).
// The reply's text changes meanwhile (msg_append may realloc it), so the thread draws holding ts_msg_mu, and the main
// loop adds the new text under it only when the lock is free (reply_append: trylock, else the bytes wait in ts_text
// for the next token); ts_stop adds what is left. The main loop never waits for a frame.
static OrbisPthreadMutex ts_mu, ts_msg_mu;
static OrbisPthread ts_th;
static volatile int ts_run;
static char ts_text[16384];   // reply bytes not yet in the message (the thread was drawing)
static int ts_text_n;

static void* ts_main(void* arg) {
    (void)arg;
    double prev = now(), t_stats = prev, draw_s = 0, gap_max = 0;
    int frames = 0;
    while (ts_run) {
        scePthreadMutexLock(&ts_mu);
        ChatView v = view;
        scePthreadMutexUnlock(&ts_mu);
        double t = now(), dt = t - prev;
        prev = t;
        v.time = t;
        if (v.thinking) v.think_s = t - think_t0;
        double left = toast_until - t;
        v.toast = left > 0 ? toast_text : NULL;
        v.toast_alpha = left > 0.4 ? 1.0f : left > 0 ? (float)(left / 0.4) : 0.0f;
        int scroll = v.scroll;
        scePthreadMutexLock(&ts_msg_mu);   // (the messages stay as they are while drawn)
        draw_chat(&v);
        scePthreadMutexUnlock(&ts_msg_mu);
        double t1 = now();
        present();
        scePthreadMutexLock(&ts_mu);
        if (view.scroll == scroll) view.scroll = v.scroll;   // (draw_chat keeps it in range)
        scePthreadMutexUnlock(&ts_mu);
        draw_s += t1 - t, frames++;
        if (frames > 1 && dt > gap_max) gap_max = dt;
        if (t - t_stats >= 10.0) {
            say("chat screen thread: %.1f frames/s, draw %.1f ms, longest gap %.0f ms", frames / (t - t_stats),
                1000 * draw_s / frames, 1000 * gap_max);
            t_stats = t, frames = 0, draw_s = gap_max = 0;
        }
    }
    return NULL;
}

static void ts_start(void) {
    static int init;
    if (!init) {
        scePthreadMutexInit(&ts_mu, NULL, "garlic_think_screen");
        scePthreadMutexInit(&ts_msg_mu, NULL, "garlic_think_msgs");
        init = 1;
    }
    ts_text_n = 0;
    view_update();
    ts_run = 1;
    OrbisPthreadAttr attr;
    scePthreadAttrInit(&attr);
    scePthreadAttrSetstacksize(&attr, 1 << 20);
    if (scePthreadCreate(&ts_th, &attr, ts_main, NULL, "garlic_think_screen")) {
        ts_run = 0;
        say("thinking screen: no thread, the main loop draws");
        return;
    }
    scePthreadSetprio(ts_th, 256 + 128);   // above the GPU loop (normal), as the voice screen
}

static void ts_stop(void) {
    if (!ts_run) return;
    double t0 = now();
    ts_run = 0;
    scePthreadJoin(ts_th, NULL);
    if (ts_text_n && n_msgs) msg_append(&msgs[n_msgs - 1], ts_text, ts_text_n);   // (what waited for the lock)
    ts_text_n = 0;
    say("chat screen thread stopped in %.1f ms", (now() - t0) * 1e3);
}

// New text of the reply: straight into the message, or (while the thread draws the chat screen) when it is not drawing.
static void reply_append(Msg* m, const char* s, int n) {
    if (!ts_run) {
        msg_append(m, s, n);
        return;
    }
    if (ts_text_n + n > (int)sizeof(ts_text)) {   // (only if the lock stayed busy for ~16 KB: wait for it)
        scePthreadMutexLock(&ts_msg_mu);
        msg_append(m, ts_text, ts_text_n), ts_text_n = 0;
        scePthreadMutexUnlock(&ts_msg_mu);
    }
    memcpy(ts_text + ts_text_n, s, n), ts_text_n += n;
    if (scePthreadMutexTrylock(&ts_msg_mu) == 0) {
        msg_append(m, ts_text, ts_text_n), ts_text_n = 0;
        scePthreadMutexUnlock(&ts_msg_mu);
    }
}

static void show_chat(void) {
    static int frames;
    if (ts_run) {   // (the thinking screen's thread draws: the view is only published)
        scePthreadMutexLock(&ts_mu);
        view_update();
        scePthreadMutexUnlock(&ts_mu);
        return;
    }
    view_update();
    double t0 = now();
    draw_chat(&view);
    double t1 = now();
    present();
    if (generating && ++frames % 300 == 0)   // drawing time on the console's CPU, now and then
        say("frame: draw %.1f ms, present %.1f ms", (t1 - t0) * 1000, (now() - t1) * 1000);
}

static void loading(void* user, float frac, const char* what) {
    static double last;
    double t = now();
    if (frac < 1.0f && t - last < 0.05) return;
    last = t;
    draw_loading(frac, user ? (const char*)user : what);
    present();
}

static void show_error(const char* what) {
    say("fatal: %s", what);
    vs_stop();   // (the voice screen's thread no longer draws)
    ts_stop();
    for (;;) {
        draw_error(what);
        present();
    }
}

// ---- conversation --------------------------------------------------------------------------------------------

static int encode_into(uint32_t* ids, int n, int max, const char* text) { return n + tok_encode(&tok, text, ids + n, max - n); }


// ---- Thinking: the model reasons before it answers (R3 or the touch pad, for the next questions) --------------------
// Qwen3.5's own thinking (its chat template with enable_thinking): the turn opens <think> and the model reasons, at
// temperature 0.7 with min-p 0.05, for up to think_cap tokens (1024; another number in /data/garlic_think.txt, e.g.
// 512). If it has not closed </think> by then, Qwen's closing sentence and </think> are fed and it answers (a forced
// close). Voice mode never thinks; the default is off.
// While it reasons the screen shows the reasoning's last lines, its tokens and seconds; the reply is only the answer.
// Then the reasoning leaves the conversation, as Qwen's template drops it from earlier turns: the DeltaNet state from
// before it was saved (in the dictation's work memory, which nothing uses while the chat generates) and is put back
// when the turn ends, and the answer is read again after an empty <think></think>, like every other turn of the app
// (hist_ids: when idle, a few tokens per frame; whatever is left goes before the next prompt). Without room for the
// copy the reasoning stays in the conversation.

static void think_config(void) {
    size_t sz;
    char* f = (char*)read_all("/data/garlic_think.txt", &sz);
    if (f && sz) {
        char b[16] = {0};
        memcpy(b, f, sz < 15 ? sz : 15);
        int c = atoi(b);
        if (c >= 64 && c <= 4096) think_cap = c;
    }
    free(f);
    uint64_t need = engine_state_size(&eng), have = eng.asr.ws_end ? eng.asr.ws_end - eng.asr.ws_begin : 0;
    say("thinking (R3): up to %d tokens; the state copy takes %.2f of the %.2f MiB of dictation work memory%s", think_cap,
        need / 1048576.0, have / 1048576.0, need <= have ? "" : ": NO ROOM, the reasoning will stay in the history");
}

static void think_toggle(void) {
    think_on = !think_on;
    toast(think_on ? S_THINK_ON_TOAST : S_THINK_OFF_TOAST);
}

static int think_save(void) {
    uint64_t need = engine_state_size(&eng);
    if (!eng.asr.ws_end || eng.asr.ws_end - eng.asr.ws_begin < need) return 0;
    double t0 = now();
    if (engine_state_copy(&eng, eng.asr.ws_begin, 1)) {
        say("thinking: state copy failed (%s): the reasoning stays in the history", eng.error);
        return 0;
    }
    say("thinking: DeltaNet state saved (%.2f MiB) in %.1f ms", need / 1048576.0, (now() - t0) * 1e3);
    return 1;
}

// One token of the reasoning (the thinking screen's thread draws meanwhile, see ts_main). At </think>, at the cap, near
// the end of the context, or if the model tries to end the turn, the answer begins.
static void think_one(void) {
    uint32_t t = next_token;
    int natural = t == THINK_END;
    if (!natural && t != IM_END && t != eng.eos2 && think_n < turn_cap && pos + 96 < (int)eng.ctx_limit) {
        const uint8_t* b;
        int nb = tok_piece(&tok, t, &b);
        if (think_len + nb < sizeof(think_text) - 1) memcpy(think_text + think_len, b, nb), think_len += nb;
        double t0 = now();
#ifdef GARLIC_SPEC
        if (spec_active(&eng) && !in_voice_mode) {   // several tokens per cycle (speculative decoding, engine_spec.inc)
            uint32_t ex[8];
            int ne = 0, cap = turn_cap - think_n, room = (int)eng.ctx_limit - 96 - pos;
            if (cap > room) cap = room;
            if (engine_spec_step(&eng, t, pos, THINK_T, seed++, cap > 1 ? cap : 1, ex, &ne, &next_token))
                show_error(eng.error);
            pos++, think_n++;
            for (int i = 0; i < ne; i++) {
                nb = tok_piece(&tok, ex[i], &b);
                if (think_len + nb < sizeof(think_text) - 1) memcpy(think_text + think_len, b, nb), think_len += nb;
                pos++, think_n++;
            }
            double dt = now() - t0;
            view.tok_per_s = view.tok_per_s ? view.tok_per_s * 0.9 + 0.1 * (1 + ne) / dt : (1 + ne) / dt;
            if (think_len < sizeof(think_text)) think_text[think_len] = 0;
            return;
        }
#endif
        if (engine_run(&eng, &t, 1, pos, THINK_T, seed++, &next_token)) show_error(eng.error);
        double dt = now() - t0;
        view.tok_per_s = view.tok_per_s ? view.tok_per_s * 0.9 + 0.1 / dt : 1.0 / dt;
        pos++, think_n++;
        if (think_len < sizeof(think_text)) think_text[think_len] = 0;
        return;
    }
    // (the thread keeps drawing the answer, see reply_append)
    // </think> as the model wrote it; or the closing sentence, </think> and the blank line
    uint32_t ids[64];
    int n = 0;
    if (!natural) n = encode_into(ids, 0, 56, THINK_CLOSE), think_cut = 1;
    ids[n++] = THINK_END;
    if (!natural) n = encode_into(ids, n, 64, "\n\n");
    if (engine_run(&eng, ids, n, pos, view.temperature, seed++, &next_token)) show_error(eng.error);
    pos += n;
    thinking = 0;
    think_t1 = now();
    Msg* m = &msgs[n_msgs - 1];
    if (ts_run) scePthreadMutexLock(&ts_msg_mu);
    m->think_n = think_n, m->think_s = (float)(think_t1 - think_t0), m->think_cut = think_cut;
    if (ts_run) scePthreadMutexUnlock(&ts_msg_mu);
    say("thinking: %d tokens in %.1f s (%.1f tokens/s), %s", think_n, think_t1 - think_t0,
        think_n / (think_t1 > think_t0 ? think_t1 - think_t0 : 1), natural ? "closed by the model" : "cut: closing sentence fed");
}

// What the conversation keeps of a thought turn: an empty block, as in every other turn, then the answer.
static void hist_begin(void) {
    hist_n = hist_done = 0;
    hist_ids[hist_n++] = THINK;
    hist_n = encode_into(hist_ids, hist_n, MAX_REPLY + 16, "\n\n");
    hist_ids[hist_n++] = THINK_END;
    hist_n = encode_into(hist_ids, hist_n, MAX_REPLY + 16, "\n\n");
}


// The end of a thought turn (finish_turn): timings to the log, and the conversation back to where the reasoning
// began; the answer is then read again from there (hist_ahead / start_turn).
static void think_finish(void) {
    double t_end = now();
    Msg* m = &msgs[n_msgs - 1];
    if (thinking) {   // stopped while it was still reasoning: no answer
        thinking = 0;
        think_t1 = t_end;
        m->think_n = think_n, m->think_s = (float)(t_end - think_t0), m->think_cut = 0;
        if (!m->len) msg_append(m, "\xE2\x80\xA6", 3);
        hist_begin();
    }
    think_turn = 0;
    say("thinking turn: prompt %.2f s, reasoning %d tokens %.1f s%s, answer %d tokens %.1f s: %.1f s from the question "
        "to the end", think_t0 - think_t_ask, think_n, think_t1 - think_t0, think_cut ? " (cut)" : "", reply_tokens,
        t_end - think_t1, t_end - think_t_ask);
    if (!think_saved) {   // no copy: the reasoning stays in the conversation
        hist_n = hist_done = 0;
        return;
    }
    think_saved = 0;
    double t0 = now();
    if (engine_state_copy(&eng, eng.asr.ws_begin, 0)) show_error(eng.error);
    say("thinking: reasoning taken out of the history: back to position %d (from %d) in %.1f ms; %d tokens to read again",
        think_pos0, pos, (now() - t0) * 1e3, hist_n);
    pos = think_pos0;
}


// Idle on the chat screen: a few tokens of the thought turn's answer read again (without the reasoning).
static void hist_ahead(int tokens) {
    if (generating || pos <= 0 || hist_done >= hist_n) return;
    int k = hist_n - hist_done < tokens ? hist_n - hist_done : tokens;
    if (pos + k > (int)eng.ctx_limit) return;   // (start_turn starts over)
    uint32_t next;
    static double t_first;
    if (hist_done == 0) t_first = now();
    if (engine_run(&eng, hist_ids + hist_done, k, pos, 0.0f, 1, &next)) show_error(eng.error);
    pos += k, hist_done += k;
    if (hist_done < hist_n) return;
    say("thinking: the answer read again without the reasoning (%d tokens, %.2f s while idle)", hist_n, now() - t_first);
}


// The system prompt goes through the model ahead of time while nothing else runs (a new conversation then starts
// with pos = its length): a few tokens per call, so the screen keeps moving. sys_pre counts the tokens done.
static uint32_t sys_ids[1024];
static int sys_n, sys_pre;

#define SYSTEM_PROMPT_TEXT tr(S_SYSTEM_PROMPT)
#define PY_FIX_MSG_TEXT tr(S_PY_FIX_MSG)
#define PY_FIX_ASK_TEXT tr(S_PY_FIX_ASK)

static int system_ids(uint32_t* ids, int max) {
    int n = 0;
    ids[n++] = IM_START;
    n = encode_into(ids, n, max, "system\n");
    n = encode_into(ids, n, max, SYSTEM_PROMPT_TEXT);
    ids[n++] = IM_END;
    n = encode_into(ids, n, max, "\n");
    return n;
}

static void system_ahead(int tokens) {
    if (pos != 0 || generating) {
        sys_pre = 0;
        return;
    }
    if (sys_pre == 0) {
        sys_n = system_ids(sys_ids, 1024);
        engine_reset(&eng);
    }
    int k = sys_n - sys_pre < tokens ? sys_n - sys_pre : tokens;
    uint32_t next;
    if (engine_run(&eng, sys_ids + sys_pre, k, sys_pre, 0.0f, 1, &next)) {
        sys_pre = 0;
        return;
    }
    sys_pre += k;
    if (sys_pre == sys_n) {   // done: the conversation continues after it
        pos = sys_n, need_close = 0, sys_pre = 0;
        say("system prompt read ahead (%d tokens)", sys_n);
    }
}

static void start_turn(const char* user_text) {
    static uint32_t ids[8192];   // (with what is left of a thought turn's answer, hist_ids)
    int n = 0, max = 8192, think = think_on && !in_voice_mode && eng.qwen35;
    think_t_ask = now();
    sys_pre = 0;   // (a system prompt half read ahead is read again below)
    if (pos == 0) {
        engine_reset(&eng);   // a new conversation: clear the Gated DeltaNet states
        n = system_ids(ids, max);
    } else
    {
        for (int i = hist_done; i < hist_n; i++) ids[n++] = hist_ids[i];   // the thought turn's answer, still unread
        if (hist_done < hist_n)
            say("thinking: the last %d of the answer's %d tokens are read with this prompt", hist_n - hist_done, hist_n);
    }
    hist_n = hist_done = 0;
    if (pos && need_close)
    {   // the previous answer's end token was sampled but never fed to the model
        // A reply the user stopped (or that reached a limit) ends in "…". A bare <|im_end|> after half a sentence
        // teaches the model that its replies stop like that, and it then tends to end the next replies too early.
        if (cut_close) n = encode_into(ids, n, max, "\xE2\x80\xA6");
        ids[n++] = IM_END;
        n = encode_into(ids, n, max, "\n");
    }
    ids[n++] = IM_START;
    n = encode_into(ids, n, max, "user\n");
    n = encode_into(ids, n, max - 8, user_text);
    ids[n++] = IM_END;
    n = encode_into(ids, n, max, "\n");
    ids[n++] = IM_START;
    n = encode_into(ids, n, max, "assistant\n");
    int head = n;   // thinking: the prompt up to here, then the state is saved, then <think>\n
    if (think) {
        ids[n++] = THINK;
        n = encode_into(ids, n, max, "\n");
    } else
    if (eng.qwen35) {   // Qwen3.5 thinks by default; the chat template's enable_thinking=false gives an empty block
        ids[n++] = THINK;
        n = encode_into(ids, n, max, "\n\n");
        ids[n++] = THINK_END;
        n = encode_into(ids, n, max, "\n\n");
    }

    int room = 1024 + (think ? think_cap + 64 : 0);
    if (pos + n + room > (int)eng.ctx_limit) {   // a nearly full context starts over (1024 free for the reply)
        pos = 0;
        need_close = 0;
        Msg* note = msg_new(MSG_NOTE);
        msg_append(note, tr(S_CONTEXT_FULL), strlen(tr(S_CONTEXT_FULL)));
        start_turn(user_text);
        return;
    }
    Msg* m = msg_new(MSG_USER);
    msg_append(m, user_text, strlen(user_text));
    msg_new(MSG_AI);
    if (in_voice_mode) tts_new_reply();
    view.scroll = 0;
    view.reading = 1;
    if (in_voice_mode) show_voice(VM_THINK);
    else show_chat();
    double t0 = now();
    if (think) {   // the prompt up to "assistant\n", the DeltaNet state saved there, then <think>\n
        if (engine_run(&eng, ids, head, pos, view.temperature, seed++, &next_token)) show_error(eng.error);
        think_saved = think_save();
        think_pos0 = pos + head;
        if (engine_run(&eng, ids + head, n - head, pos + head, THINK_T, seed++, &next_token)) show_error(eng.error);
    } else
    {
        if (engine_run(&eng, ids, n, pos, view.temperature, seed++, &next_token)) show_error(eng.error);
    }
    say("prompt of %d tokens in %.2f s (GPU %.2f s), first token %u", n, now() - t0, eng.last_gpu_seconds, next_token);
    view.reading = 0;
    pos += n;
    generating = 1;
    need_close = 0;
    reply_tokens = 0;
    reply_t0 = now();
    n_pending = 0;
    view.tok_per_s = 0;
    think_turn = thinking = think;
    if (think) {
        think_n = think_cut = think_answer = 0, think_len = 0, turn_cap = think_cap;
        think_t0 = now();
        hist_begin();
    }
    if (!in_voice_mode) ts_start();   // (the chat screen's own thread draws the reply; voice mode has its own)
}


// Test of the long context cache (shaders/attn.h ATTN_SPLIT, attn_tq.h), when /data/garlic_kvtest.txt says 1: a
// prompt of more than max_seq / 2 tokens with one fact near the start and one past the middle (and, three >= 1, a
// third past 7600; three = 2, a fourth past 14500, in the second 8K of the 16K TurboQuant cache); the greedy answer
// must have them all. The prompt's ids and the logits after it go to /data/garlic_kvtest_<tokens>.bin, to compare
// with a reference.
static void kv_case(int three) {
    static uint32_t ids[16384];
    int n = 0, max = 16384;
    ids[n++] = IM_START;
    n = encode_into(ids, n, max, "user\nLee este texto con atención. El número secreto es 7351.\n");
    const char* filler = "La lluvia caía despacio sobre los tejados del pueblo mientras los vecinos conversaban en la plaza "
                         "sobre la cosecha, el precio del pan y las fiestas del verano que pronto llegarían. ";
    while (n < 4300) n = encode_into(ids, n, max, filler);
    n = encode_into(ids, n, max, "La palabra secreta es GIRASOL. ");
    if (three) {
        while (n < 7600) n = encode_into(ids, n, max, filler);
        n = encode_into(ids, n, max, "El color secreto es TURQUESA. ");
    }
    if (three == 2) {
        while (n < 14500) n = encode_into(ids, n, max, filler);
        n = encode_into(ids, n, max, "El animal secreto es PELÍCANO. ");
    }
    for (int i = 0; i < 3; i++) n = encode_into(ids, n, max, filler);
    n = encode_into(ids, n, max, three == 2 ? "\n¿Cuál es el número secreto, cuál es la palabra secreta, cuál es el color "
                                              "secreto y cuál es el animal secreto? Responde en una línea."
                                 : three ? "\n¿Cuál es el número secreto, cuál es la palabra secreta y cuál es el color "
                                           "secreto? Responde en una línea."
                                         : "\n¿Cuál es el número secreto y cuál es la palabra secreta? Responde en una línea.");
    ids[n++] = IM_END;
    n = encode_into(ids, n, max, "\n");
    ids[n++] = IM_START;
    n = encode_into(ids, n, max, "assistant\n");
    ids[n++] = THINK;
    n = encode_into(ids, n, max, "\n\n");
    ids[n++] = THINK_END;
    n = encode_into(ids, n, max, "\n\n");
    engine_reset(&eng);
    double t0 = now(), gpu = 0;
    uint32_t next = 0;
    for (int o = 0; o < n; o += 512) {
        int k = n - o < 512 ? n - o : 512;
        if (engine_run(&eng, ids + o, k, o, 0.0f, 1, &next)) {
            say("kv test: %s", eng.error);
            return;
        }
        gpu += eng.last_gpu_seconds;
        loading((void*)"Prueba de contexto largo", (float)(o + k) / n, NULL);
    }
    double dt = now() - t0;
    say("kv test: prompt of %d tokens in %.1f s (%.1f ms per token, GPU %.1f s; %s cache, %s)", n, dt, 1e3 * dt / n, gpu,
        eng.tq ? "TurboQuant 4-bit" : "fp16/int8", eng.pf && !eng.pf_off ? "in batches" : "token by token");
    char path[64];
    snprintf(path, sizeof(path), "/data/garlic_kvtest_%d.bin", n);
    int f = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0666);
    if (f >= 0) {
        write(f, ids, n * 4u);
        write(f, (const void*)(uintptr_t)eng.logits, eng.vocab * 4u);
        close(f);
    }
    char ans[512];
    int al = 0, p = n, steps = 0;
    t0 = now();
    for (int i = 0; i < 48 && next != IM_END && next != eng.eos2; i++) {
        const uint8_t* b;
        int nb = tok_piece(&tok, next, &b);
        if (al + nb < (int)sizeof(ans) - 1) memcpy(ans + al, b, nb), al += nb;
        uint32_t t = next;
#ifdef GARLIC_SPEC
        if (spec_active(&eng)) {   // the answer with the MTP head (its cache filled through the whole prompt)
            uint32_t ex[8];
            int ne = 0;
            if (engine_spec_step(&eng, t, p++, 0.0f, 1, 48 - i, ex, &ne, &next)) break;
            steps++;
            for (int k = 0; k < ne; k++, i++, steps++, p++) {
                nb = tok_piece(&tok, ex[k], &b);
                if (al + nb < (int)sizeof(ans) - 1) memcpy(ans + al, b, nb), al += nb;
            }
            continue;
        }
#endif
        if (engine_run(&eng, &t, 1, p++, 0.0f, 1, &next)) break;
        steps++;
    }
    dt = now() - t0;
    ans[al] = 0;
    say("kv test answer (%d tokens at position %d+, %.1f tokens/s): %s", steps, n, steps / (dt > 0 ? dt : 1), ans);
    engine_reset(&eng);
}

// Speed with a short context: a question, then 64 greedy tokens (the answer goes to the log).
static void speed_case(void) {
    static uint32_t ids[256];
    int n = 0;
    ids[n++] = IM_START;
    n = encode_into(ids, n, 256, "user\n¿Quién es Paolo Guerrero y dónde nació?");
    ids[n++] = IM_END;
    n = encode_into(ids, n, 256, "\n");
    ids[n++] = IM_START;
    n = encode_into(ids, n, 256, "assistant\n");
    ids[n++] = THINK;
    n = encode_into(ids, n, 256, "\n\n");
    ids[n++] = THINK_END;
    n = encode_into(ids, n, 256, "\n\n");
    engine_reset(&eng);
    uint32_t next = 0;
    if (engine_run(&eng, ids, n, 0, 0.0f, 1, &next)) return;
    char ans[600];
    int al = 0, p = n, steps = 0;
    double t0 = now();
    for (int i = 0; i < 64 && next != IM_END && next != eng.eos2; i++) {
        const uint8_t* b;
        int nb = tok_piece(&tok, next, &b);
        if (al + nb < (int)sizeof(ans) - 1) memcpy(ans + al, b, nb), al += nb;
        uint32_t t = next;
        if (engine_run(&eng, &t, 1, p++, 0.0f, 1, &next)) break;
        steps++;
    }
    double dt = now() - t0;
    ans[al] = 0;
    say("speed test (%s cache): %d tokens after a prompt of %d, %.1f tokens/s: %s", eng.tq ? "TurboQuant" : "fp16/int8",
        steps, n, steps / (dt > 0 ? dt : 1), ans);
    engine_reset(&eng);
}


// The start-up tests of the context, run when /data/garlic_kvtest.txt says 1
static void kv_split_test(void) {
    size_t sz;
    char* flag;
    flag = (char*)read_all("/data/garlic_kvtest.txt", &sz);
    int on = flag && sz && flag[0] == '1';
    free(flag);
    if (!on) return;
    speed_case();
    kv_case(0);
    if (eng.ctx_limit >= 8192) kv_case(1);
    if (eng.ctx_limit >= 16384) kv_case(2);
}


static void finish_turn(int ended_by_model) {
    ts_stop();   // (the main loop draws again, and may change the messages)
    generating = 0;
    need_close = 1;   // feed <|im_end|> before the next turn either way
    cut_close = !ended_by_model;
    double t_ans = now() - (think_turn && !thinking ? think_t1 : reply_t0);
    say("reply of %d tokens (%s, %.1f tokens/s in the engine, %.1f on screen, T %.1f): %.300s", reply_tokens,
        ended_by_model ? "ended" : "stopped", view.tok_per_s, t_ans > 0 ? reply_tokens / t_ans : 0.0, view.temperature,
        msgs[n_msgs - 1].text);
#ifdef GARLIC_SPEC
    if (spec_active(&eng)) {
        char sl[300];
        spec_stats(&eng, sl, sizeof(sl));
        say("%s", sl);
        spec_stats_reset();
    }
#endif
    if (think_turn) think_finish();
}

// A token of the reply on screen (only complete UTF-8 characters) and, after a reasoning, in the history.
static void reply_token(uint32_t token) {
    const uint8_t* bytes;
    int nb = tok_piece(&tok, token, &bytes);
    if (think_turn) {   // the answer after the reasoning: not the blank lines after </think>; kept for the history
        if (!think_answer) {
            int blank = 1;
            for (int i = 0; i < nb; i++) blank &= bytes[i] == '\n' || bytes[i] == '\r' || bytes[i] == ' ';
            if (blank) nb = 0;
            else think_answer = 1;
        }
        if (think_answer && hist_n < (int)(sizeof(hist_ids) / sizeof(hist_ids[0]))) hist_ids[hist_n++] = token;
    }
    Msg* m = &msgs[n_msgs - 1];
    for (int i = 0; i < nb; i++) {   // only complete UTF-8 characters go on screen
        pending[n_pending++] = bytes[i];
        uint8_t c0 = pending[0];
        int need = c0 < 0x80 ? 1 : (c0 >> 5) == 6 ? 2 : (c0 >> 4) == 14 ? 3 : (c0 >> 3) == 30 ? 4 : 1;
        if (n_pending >= need || n_pending == 8) {
            reply_append(m, (const char*)pending, n_pending);
            n_pending = 0;
        }
    }
}

// Feeds the sampled token back and adds its text; one token per frame (several with the MTP head, GARLIC_SPEC).
static void generate_one(void) {
    if (thinking) {
        think_one();
        return;
    }
    if (next_token == IM_END || next_token == eng.eos2 || reply_tokens >= MAX_REPLY || pos >= (int)eng.ctx_limit - 1) {
        finish_turn(next_token == IM_END || next_token == eng.eos2);
        return;
    }
    reply_token(next_token);
    uint32_t t = next_token;
    double t0 = now();
#ifdef GARLIC_SPEC
    if (spec_active(&eng) && !in_voice_mode) {   // several tokens per cycle (speculative decoding, engine_spec.inc)
        uint32_t ex[8];
        int ne = 0;
        if (engine_spec_step(&eng, t, pos, view.temperature, seed++, MAX_REPLY - reply_tokens, ex, &ne, &next_token))
            show_error(eng.error);
        pos++;
        reply_tokens++;
        for (int i = 0; i < ne; i++) {
            reply_token(ex[i]);
            pos++;
            reply_tokens++;
        }
        double dt = now() - t0;
        view.tok_per_s = view.tok_per_s ? view.tok_per_s * 0.9 + 0.1 * (1 + ne) / dt : (1 + ne) / dt;
        return;
    }
#endif
    if (engine_run(&eng, &t, 1, pos, view.temperature, seed++, &next_token)) show_error(eng.error);
    double dt = now() - t0;
    view.tok_per_s = view.tok_per_s ? view.tok_per_s * 0.9 + 0.1 / dt : 1.0 / dt;
    pos++;
    reply_tokens++;
}

// ---- microphone (sceAudioIn) -------------------------------------------------------------------------------------
// The controller's headset jack or a USB microphone, 16 kHz mono 16-bit, read by its own thread while a button
// is held. Not declared in the OpenOrbis headers, hence the prototypes here.
int32_t sceAudioInOpen(int32_t userId, uint32_t type, uint32_t index, uint32_t len, uint32_t freq, uint32_t param);
int32_t sceAudioInInput(int32_t handle, void* dest);
int32_t sceAudioInClose(int32_t handle);

#define REC_RATE 16000
#define REC_MAX_S 30
#define N_LEVELS 64
static int16_t rec_buf[REC_RATE * REC_MAX_S];
static volatile int rec_run, rec_n, rec_err, rec_heard, rec_nlev;
static float rec_levels[N_LEVELS];
static int mic = -1, mic_len, mic_decim;
static OrbisPthread rec_thread;

static int mic_open(void) {
    if (mic >= 0) return 0;
    static const struct {
        uint32_t type, len, freq;
    } tries[] = {{0, 256, 16000}, {1, 256, 16000}, {5, 256, 16000}, {1, 768, 48000}};
    for (size_t i = 0; i < sizeof(tries) / sizeof(tries[0]); i++) {
        int h = sceAudioInOpen(user_id, tries[i].type, 0, tries[i].len, tries[i].freq, 0);   // param 0: S16 mono
        say("sceAudioInOpen(type %u, %u Hz) -> 0x%08x", tries[i].type, tries[i].freq, (unsigned)h);
        if (h >= 0) {
            mic = h, mic_len = (int)tries[i].len, mic_decim = (int)(tries[i].freq / REC_RATE);
            return 0;
        }
    }
    return -1;
}

static void* rec_main(void* arg) {
    (void)arg;
    static int16_t chunk[768];
    while (rec_run) {
        int r = sceAudioInInput(mic, chunk);
        if (r < 0) {
            rec_err = r;
            break;
        }
        int n = mic_len / mic_decim;
        double sum = 0;
        for (int i = 0; i < n; i++) {
            int v = 0;
            for (int d = 0; d < mic_decim; d++) v += chunk[i * mic_decim + d];
            v /= mic_decim;
            if (rec_n < REC_RATE * REC_MAX_S) rec_buf[rec_n++] = (int16_t)v;
            if (v) rec_heard = 1;
            sum += (double)v * v;
        }
        float rms = (float)sqrt(sum / n) / 32768.0f;
        float db = rms > 0 ? 20.0f * log10f(rms) : -100.0f;
        float lv = (db + 55.0f) / 45.0f;   // -55 dB .. -10 dB -> 0 .. 1
        rec_levels[rec_nlev % N_LEVELS] = lv < 0 ? 0 : lv > 1 ? 1 : lv;
        rec_nlev++;
    }
    return NULL;
}

static float* read_wav16k(const char* path, int* n) {   // 16 kHz WAV, 16-bit PCM or 32-bit float, mixed to mono
    size_t size;
    uint8_t* d = read_all(path, &size);
    if (!d || size < 44 || memcmp(d, "RIFF", 4) || memcmp(d + 8, "WAVE", 4)) {
        free(d);
        return NULL;
    }
    uint16_t fmt = 0, ch = 0, bits = 0;
    uint32_t rate = 0;
    const uint8_t* data = NULL;
    uint32_t data_size = 0;
    for (size_t o = 12; o + 8 <= size;) {
        uint32_t len;
        memcpy(&len, d + o + 4, 4);
        if (!memcmp(d + o, "fmt ", 4) && len >= 16) {
            memcpy(&fmt, d + o + 8, 2);
            memcpy(&ch, d + o + 10, 2);
            memcpy(&rate, d + o + 12, 4);
            memcpy(&bits, d + o + 22, 2);
        } else if (!memcmp(d + o, "data", 4)) {
            data = d + o + 8;
            data_size = len <= size - o - 8 ? len : (uint32_t)(size - o - 8);
        }
        o += 8 + len + (len & 1);
    }
    if (!data || rate != 16000 || !ch || !((fmt == 1 && bits == 16) || (fmt == 3 && bits == 32))) {
        free(d);
        return NULL;
    }
    int frames = data_size / (bits / 8) / ch;
    float* out = malloc(frames * sizeof(float));
    for (int i = 0; out && i < frames; i++) {
        float v = 0.0f;
        for (int c = 0; c < ch; c++) {
            if (fmt == 1) {
                int16_t s16;
                memcpy(&s16, data + ((size_t)i * ch + c) * 2, 2);
                v += s16 / 32768.0f;
            } else {
                float f;
                memcpy(&f, data + ((size_t)i * ch + c) * 4, 4);
                v += f;
            }
        }
        out[i] = v / ch;
    }
    free(d);
    *n = frames;
    return out;
}

typedef void (*DrawUnder)(void);

// Records while `button` is held (the voice screen over the saved backdrop, plus `under` if given; in voice mode,
// that screen's thread draws), then transcribes. Returns 1 with the text, 0 if there is nothing to send (a toast
// says why).
static void asr_wait(void);

static int voice_to_text(uint32_t button, DrawUnder under, char* text, int max) {
    text[0] = 0;
    asr_wait();   // (after a Python program the dictation weights are read back first)
    if (!eng.asr.ready) {
        toast(S_ASR_OFF);
        return 0;
    }
    float max_s = eng.asr.max_frames * 0.08f - 0.5f;
    if (max_s > REC_MAX_S) max_s = REC_MAX_S;
    int have_mic = mic_open() == 0;
    int vf = open(VOICE_FILE, O_RDONLY);   // (access() always fails on the console)
    int have_file = vf >= 0;
    if (vf >= 0) close(vf);
    if (!have_mic && !have_file) {
        toast(S_NO_MIC);
        return 0;
    }
    rec_n = rec_err = rec_heard = rec_nlev = 0;
    memset(rec_levels, 0, sizeof(rec_levels));
    if (have_mic) {
        rec_run = 1;
        if (scePthreadCreate(&rec_thread, NULL, rec_main, NULL, "garlic_mic")) rec_run = 0, have_mic = 0;
    }
    double t0 = now(), secs = 0;
    float levels[N_LEVELS] = {0};
    while ((read_buttons() & button) && secs < max_s && !rec_err) {
        secs = now() - t0;
        int nl = rec_nlev;
        for (int i = 0; i < N_LEVELS; i++) {
            int k = nl - N_LEVELS + i;
            levels[i] = have_mic && k >= 0 ? rec_levels[k % N_LEVELS] : 0.08f + 0.06f * sinf((float)(secs * 9 + i));
        }
        if (in_voice_mode) {
            vs_publish(0, 1, levels[N_LEVELS - 1]);
            sceKernelUsleep(8000);
        } else {
            ui_backdrop_draw();
            if (under) under(), ui_dim();
            draw_voice(VOICE_RECORDING, levels, N_LEVELS, (float)secs, max_s, now());
            present();
        }
    }
    if (have_mic) {
        rec_run = 0;
        scePthreadJoin(rec_thread, NULL);
    }
    say("recorded %.2f s: %d samples, mic %d, heard %d, error 0x%08x", secs, rec_n, have_mic, rec_heard, (unsigned)rec_err);
    if (secs < 0.4) {
        toast(S_HOLD_LONGER);
        return 0;
    }
    int n = 0;
    float* audio = NULL;
    if (have_mic && !rec_heard) have_mic = 0;   // an open device that only gives zeros: no microphone plugged in
    if (have_mic && rec_n > REC_RATE / 4) {
        audio = malloc(rec_n * sizeof(float));
        for (int i = 0; i < rec_n; i++) audio[i] = rec_buf[i] / 32768.0f;
        n = rec_n;
    } else if (have_file) {   // no microphone (or only silence): the test recording
        audio = read_wav16k(VOICE_FILE, &n);
        say("using %s instead of the microphone", VOICE_FILE);
    }
    if (!audio) {
        toast(have_mic ? S_NO_SPEECH : S_NO_MIC);
        return 0;
    }
    if (in_voice_mode) {
        vs_publish(1, 0, 0);
    } else {
        ui_backdrop_draw();
        if (under) under(), ui_dim();
        draw_voice(VOICE_TRANSCRIBING, levels, N_LEVELS, (float)secs, max_s, now());
        present();
    }
    double t1 = now();
    int r = asr_transcribe(&eng, audio, n, text, max);
    free(audio);
    if (r) {
        say("dictation failed: %s", eng.error);
        toast(S_NO_SPEECH);
        return 0;
    }
    say("dictation: %.1f s of audio in %.2f s (GPU %.2f s): %s", n / 16000.0, now() - t1, eng.last_gpu_seconds, text);
    char* s = text;   // trim
    while (*s == ' ') s++;
    memmove(text, s, strlen(s) + 1);
    for (size_t k = strlen(text); k > 0 && text[k - 1] == ' '; k--) text[k - 1] = 0;
    if (!text[0]) {
        toast(S_NO_SPEECH);
        return 0;
    }
    return 1;
}

// ---- keyboard ----------------------------------------------------------------------------------------------------

static Kb kb;
static const char* kb_text;

static void draw_kb_under(void) { draw_keyboard(&kb, kb_text, now()); }

// The app's own keyboard. Returns 1 with the text in out when the user sends, 0 if they close it. With keep,
// out already holds text to edit.
static int keyboard_ex(char* out, int max, int keep, int over_current) {
    static int ready;
    if (!ready) kb_init(&kb, ui_lang), ready = 1;   // KB_* follow LANG_*: the console's language first
    if (!keep) out[0] = 0;
    kb_text = out;
    if (!over_current) {
        view_update();
        draw_chat(&view);
        ui_dim();
    }
    ui_backdrop_save();
    uint32_t prev = read_buttons();   // the button that opened the keyboard must not type
    const uint32_t DIRS = ORBIS_PAD_BUTTON_UP | ORBIS_PAD_BUTTON_DOWN | ORBIS_PAD_BUTTON_LEFT | ORBIS_PAD_BUTTON_RIGHT;
    double rep_next = 0, rep_start = 0, del_next = 0, del_start = 0;
    for (;;) {
        uint32_t b = read_buttons(), pressed = b & ~prev;
        double t = now();
        // Auto-repeat by time, not by frames: a first step at once, a pause (so a tap moves one key), then
        // steady steps that speed up a little after a second.
        uint32_t dirs = b & DIRS, move = pressed & DIRS;
        if (move) rep_start = t, rep_next = t + 0.30;
        else if (dirs && t >= rep_next) {
            move = dirs;
            rep_next = t + (t - rep_start > 1.3 ? 0.055 : 0.085);
        }
        int del = (pressed & ORBIS_PAD_BUTTON_SQUARE) != 0;
        if (del) del_start = t, del_next = t + 0.40;
        else if ((b & ORBIS_PAD_BUTTON_SQUARE) && t >= del_next) {
            del = 1;
            del_next = t + (t - del_start > 1.5 ? 0.035 : 0.07);
        }
        prev = b;
        if (move & ORBIS_PAD_BUTTON_UP) kb_move(&kb, 0, -1);
        if (move & ORBIS_PAD_BUTTON_DOWN) kb_move(&kb, 0, 1);
        if (move & ORBIS_PAD_BUTTON_LEFT) kb_move(&kb, -1, 0);
        if (move & ORBIS_PAD_BUTTON_RIGHT) kb_move(&kb, 1, 0);
        if (pressed & ORBIS_PAD_BUTTON_CIRCLE) return 0;
        if (del) kb_backspace(out);
        if (pressed & ORBIS_PAD_BUTTON_TRIANGLE) kb_append(out, max, " ");
        if (pressed & ORBIS_PAD_BUTTON_L1) kb.shift = !kb.shift;
        if (pressed & ORBIS_PAD_BUTTON_R1) kb_toggle_emoji(&kb);
        if (pressed & ORBIS_PAD_BUTTON_R2) return out[0] != 0;
        if (pressed & ORBIS_PAD_BUTTON_L2) {   // dictation: hold L2 and speak, the text is added
            static char spoken[2048];
            if (voice_to_text(ORBIS_PAD_BUTTON_L2, draw_kb_under, spoken, sizeof(spoken))) {
                if (out[0] && out[strlen(out) - 1] != ' ') kb_append(out, max, " ");
                kb_append(out, max, spoken);
            }
            prev = read_buttons();
        }
        if ((pressed & ORBIS_PAD_BUTTON_CROSS) && kb_press(&kb, out, max)) return out[0] != 0;
        draw_keyboard(&kb, out, now());
        if (toast_until > now()) {   // a dictation problem: show it over the keyboard too
            view_update();
            draw_toast(view.toast, view.toast_alpha, 340);
        }
        present();
    }
}

static int keyboard(char* out, int max, int keep) { return keyboard_ex(out, max, keep, 0); }

static void utf8_to_utf16(const char* s, wchar_t* out, int max) {
    int n = 0;
    while (*s && n < max - 2) {
        const uint8_t* p = (const uint8_t*)s;
        uint32_t c = *p++;
        int extra = c < 0x80 ? 0 : (c >> 5) == 6 ? 1 : (c >> 4) == 14 ? 2 : (c >> 3) == 30 ? 3 : 0;
        if (extra) c &= 0x3F >> extra;
        for (int k = 0; k < extra && (*p & 0xC0) == 0x80; k++) c = c << 6 | (*p++ & 0x3F);
        s = (const char*)p;
        if (c >= 0x10000) {
            c -= 0x10000;
            out[n++] = (wchar_t)(0xD800 + (c >> 10));
            out[n++] = (wchar_t)(0xDC00 + (c & 0x3FF));
        } else out[n++] = (wchar_t)c;
    }
    out[n] = 0;
}

static void utf16_to_utf8(const uint16_t* in, char* out, int max) {
    int n = 0;
    for (int i = 0; in[i] && n < max - 5; i++) {
        uint32_t c = in[i];
        if (c >= 0xD800 && c < 0xDC00 && in[i + 1] >= 0xDC00 && in[i + 1] < 0xE000)
            c = 0x10000 + ((c - 0xD800) << 10) + (in[++i] - 0xDC00);
        if (c < 0x80) out[n++] = (char)c;
        else if (c < 0x800) out[n++] = 0xC0 | c >> 6, out[n++] = 0x80 | (c & 63);
        else if (c < 0x10000) out[n++] = 0xE0 | c >> 12, out[n++] = 0x80 | (c >> 6 & 63), out[n++] = 0x80 | (c & 63);
        else out[n++] = 0xF0 | c >> 18, out[n++] = 0x80 | (c >> 12 & 63), out[n++] = 0x80 | (c >> 6 & 63),
             out[n++] = 0x80 | (c & 63);
    }
    out[n] = 0;
}

// The system keyboard (OPTIONS), e.g. for a USB keyboard.
static int ask_user(char* out, int max) {
    static wchar_t buf[513], placeholder[80];
    memset(buf, 0, sizeof(buf));
    utf8_to_utf16(tr(S_PLACEHOLDER), placeholder, 80);
    OrbisImeDialogSetting s;
    memset(&s, 0, sizeof(s));
    s.userId = user_id;
    s.type = ORBIS_TYPE_DEFAULT;
    s.enterLabel = ORBIS_BUTTON_LABEL_SEND;
    s.maxTextLength = 512;
    s.inputTextBuffer = buf;
    s.posx = SCREEN_W / 2;
    s.posy = SCREEN_H / 2;
    s.horizontalAlignment = ORBIS_H_CENTER;
    s.verticalAlignment = ORBIS_V_CENTER;
    s.title = L"Garlic LLM";
    s.placeholder = placeholder;
    int r = sceImeDialogInit(&s, NULL);
    if (r < 0) {
        say("sceImeDialogInit -> 0x%08x", (unsigned)r);
        return 0;
    }
    // The dialog can report NONE for a few frames before it is RUNNING; only STOPPED means the user is done.
    int frames = 0, was_running = 0;
    for (;;) {
        OrbisDialogStatus st = sceImeDialogGetStatus();
        if (st == ORBIS_DIALOG_STATUS_STOPPED) break;
        if (st == ORBIS_DIALOG_STATUS_RUNNING) was_running = 1;
        if (st == ORBIS_DIALOG_STATUS_NONE && (was_running || ++frames > 300)) break;
        show_chat();
    }
    OrbisDialogResult res;
    memset(&res, 0, sizeof(res));
    sceImeDialogGetResult(&res);
    sceImeDialogTerm();
    if (res.endstatus != ORBIS_DIALOG_OK || !buf[0]) return 0;
    utf16_to_utf8((const uint16_t*)buf, out, max);
    return out[0] != 0;
}

// ---- push to talk: hold Square on the chat screen, then confirm --------------------------------------------------

// ---- running the Python code of a reply (pyrun.c) ---------------------------------------------------------------

static uint32_t py_buttons(float* lx, float* ly) {
    OrbisPadData pd;
    memset(&pd, 0, sizeof(pd));
    if (g_pad < 0 || scePadReadState(g_pad, &pd) != 0) memset(&pd, 0, sizeof(pd)), pd.leftStick.x = pd.leftStick.y = 128;
    pd.buttons |= inj_mask;   // (the test file's buttons; the main thread's read_buttons moves through them)
    static const struct {
        uint32_t pad, py;
    } map[] = {{ORBIS_PAD_BUTTON_UP, PB_UP},       {ORBIS_PAD_BUTTON_DOWN, PB_DOWN},     {ORBIS_PAD_BUTTON_LEFT, PB_LEFT},
               {ORBIS_PAD_BUTTON_RIGHT, PB_RIGHT}, {ORBIS_PAD_BUTTON_CROSS, PB_CROSS},   {ORBIS_PAD_BUTTON_CIRCLE, PB_CIRCLE},
               {ORBIS_PAD_BUTTON_SQUARE, PB_SQUARE}, {ORBIS_PAD_BUTTON_TRIANGLE, PB_TRIANGLE}, {ORBIS_PAD_BUTTON_L1, PB_L1},
               {ORBIS_PAD_BUTTON_R1, PB_R1},       {ORBIS_PAD_BUTTON_L2, PB_L2},         {ORBIS_PAD_BUTTON_R2, PB_R2},
               {ORBIS_PAD_BUTTON_OPTIONS, PB_OPTIONS}};
    uint32_t b = 0;
    for (size_t i = 0; i < sizeof(map) / sizeof(map[0]); i++)
        if (pd.buttons & map[i].pad) b |= map[i].py;
    if (lx) *lx = (pd.leftStick.x - 128) / 127.0f, *ly = (pd.leftStick.y - 128) / 127.0f;
    return b;
}

static int py_keyboard(char* out, int max) {
    toast_until = 0;
    return keyboard_ex(out, max, 0, 1);
}

// Python's heap straight from the system (flexible memory) and back with munmap after each run: with malloc / free
// the process would keep it.
// With the 9B the heap and the thread's stack are in the dictation weights' memory instead, which stays mapped while
// Python runs (asr_lent; the weights are read back after): a run needs no system RAM of its own.
#define PY_STACK (8u << 20)   // deep recursion in programs (flood fills...)
static uint8_t* py_lent;
static uint64_t py_lent_size;

static void* py_heap_alloc(size_t size) {
    void* p = NULL;
    size = (size + 0x3FFF) & ~(size_t)0x3FFF;
    if (py_lent && size + PY_STACK <= py_lent_size) return py_lent;
    return sceKernelMapFlexibleMemory(&p, size, 0x03 /* CPU read/write */, 0) ? NULL : p;
}

static void py_heap_free(void* p, size_t size) {
    if (p && p != (void*)py_lent) sceKernelMunmap(p, (size + 0x3FFF) & ~(size_t)0x3FFF);
}

static PyHost py_host = {py_buttons, present, now, py_keyboard, ASSETS "/py", 7u << 20, py_heap_alloc, py_heap_free};
static const char* py_code;
static char py_title[64];
static volatile int py_done, py_result;

static void* py_thread(void* arg) {
    (void)arg;
    py_result = py_run(&py_host, py_code, py_title);
    py_done = 1;
    return NULL;
}

static void run_python_once(void) {
    static char code[96 * 1024];
    int m = code_msg();
    if (m < 0 || !py_find_code(msgs[m].text, code, sizeof(code), py_title, sizeof(py_title))) return;
    size_t flex = 0;   // (declared with the wrong type in OpenOrbis: called with a cast, as in engine.c)
    ((int32_t(*)(size_t*))sceKernelAvailableFlexibleMemorySize)(&flex);
    say("running %s (%d bytes); system RAM free: %u MiB", py_title, (int)strlen(code), (unsigned)(flex >> 20));
    view_update();
    draw_chat(&view);
    ui_dim();
    ui_backdrop_save();
    for (double t0 = now(), t = 0; t < 1; t = (now() - t0) / 0.35) {   // a short transition
        ui_backdrop_draw();
        draw_run_intro(py_title, (float)t);
        present();
    }
    for (;;) {
        py_code = code, py_done = 0;
        OrbisPthreadAttr attr;
        OrbisPthread th;
        scePthreadAttrInit(&attr);
        if (py_lent) scePthreadAttrSetstack(&attr, py_lent + py_lent_size - PY_STACK, PY_STACK);
        else scePthreadAttrSetstacksize(&attr, PY_STACK);
        if (scePthreadCreate(&th, &attr, py_thread, NULL, "garlic_py")) {
            say("python thread failed");
            return;
        }
        uint32_t last = read_buttons();
        while (!py_done) {   // ○ also stops a program stuck in a loop that never draws
            sceKernelUsleep(30000);
            uint32_t b = read_buttons();
            if ((b & ~last) & ORBIS_PAD_BUTTON_CIRCLE) py_request_stop();
            last = b;
        }
        scePthreadJoin(th, NULL);
        say("python ended: %d %s", py_result, py_result == PY_ERROR ? py_error() : "");
        ui_backdrop_save();
        uint32_t prev = read_buttons();
        for (;;) {   // the end card: back, again, or ask Garlic to fix the error
            uint32_t b = read_buttons(), pressed = b & ~prev;
            prev = b;
            if (pressed & (ORBIS_PAD_BUTTON_CROSS | ORBIS_PAD_BUTTON_CIRCLE)) return;
            if (pressed & ORBIS_PAD_BUTTON_TRIANGLE) break;
            if ((pressed & ORBIS_PAD_BUTTON_SQUARE) && py_result == PY_ERROR) {
                static char msg[6000];
                snprintf(msg, sizeof(msg), "%s\n```\n%s```\n%s", PY_FIX_MSG_TEXT, py_error(), PY_FIX_ASK_TEXT);
                start_turn(msg);
                return;
            }
            ui_backdrop_draw();
            draw_run_end(py_title, py_result == PY_ERROR ? py_error() : NULL, py_result == PY_STOPPED);
            present();
        }
    }
}

// With the 9B, Parakeet's weights take 338 MiB of system memory, which leaves too little for Python's thread (8 MiB
// of stack) and heap (up to 64 MiB). They are released while the program runs (run_python), and a thread reads them
// back afterwards (~4 s); dictation waits for it if needed.
static OrbisPthread asr_th;
static volatile int asr_th_busy, asr_th_made;

static void* asr_reload_main(void* arg) {
    (void)arg;
    double t0 = now();
    size_t flex = 0;
    ((int32_t(*)(size_t*))sceKernelAvailableFlexibleMemorySize)(&flex);
    int r = asr_reload(&eng);
    say("asr: weights read back in %.2f s (%s); system RAM free before: %u MiB", now() - t0, r ? eng.error : "ok",
        (unsigned)(flex >> 20));
    asr_th_busy = 0;
    return NULL;
}

static void asr_reload_bg(void) {
    if (asr_th_busy || !eng.asr.unloaded) return;
    if (asr_th_made) scePthreadJoin(asr_th, NULL), asr_th_made = 0;
    // the weights need 338 MiB of system memory and only a few MiB more are free: the 8 MB screen copies that a
    // Python run (or anything since) took would leave them out, so they are freed first
    ui_free_caches();
    asr_th_busy = 1;
    if (scePthreadCreate(&asr_th, NULL, asr_reload_main, NULL, "garlic_asr")) asr_th_busy = 0;
    else asr_th_made = 1;
}

static void asr_wait(void) {
    if (eng.asr.unloaded && !asr_th_busy) asr_reload_bg();
    while (asr_th_busy) sceKernelUsleep(10000);   // (started when the program ended: usually done by now)
    if (asr_th_made) scePthreadJoin(asr_th, NULL), asr_th_made = 0;
}

static void run_python(void) {
    asr_wait();   // (a reload still running from the last program)
    asr_release(&eng);
    py_lent = (uint8_t*)asr_lent(&eng, &py_lent_size);
    size_t flex = 0;
    ((int32_t(*)(size_t*))sceKernelAvailableFlexibleMemorySize)(&flex);
    if (py_lent)
        say("python: heap and stack in the dictation weights' memory (%u MiB); system RAM free: %u MiB",
            (unsigned)(py_lent_size >> 20), (unsigned)(flex >> 20));
    run_python_once();
    py_lent = NULL, py_lent_size = 0;
    asr_reload_bg();   // (it first gives back the screen copies the run may have taken)
}

// ---- voice mode: hold Square, talk, Garlic answers; only its words are on screen -----------------------------------

#include "voice_tts.inc"

// ---- the voice screen's own thread -------------------------------------------------------------------------------
// The GPU work comes in chunks of 5-40 ms (a token of the model, six of T3, a stage of S3Gen) and a prompt or a
// dictation takes 0.5-3 s. Drawn from the same loop, the halo and the thinking dots would move only a few times a
// second and stand still while a prompt is read. In voice mode this thread draws every frame instead (60 Hz), from a
// copy of what the main thread publishes after each unit of work; it touches nothing else of the conversation, and
// the main thread draws nothing while it runs. Garlic's loudness, the words already spoken and the time are read
// live, frame by frame.
static struct {
    OrbisPthreadMutex mu;
    OrbisPthread th;
    volatile int run;
    unsigned ver;              // bumped by every publish
    int listen;                // the user is talking (VM_LISTEN, with listen_level)
    float listen_level;
    int busy;                  // Garlic is thinking: the model or the voice has work, or a prompt or dictation is read
    int gen;                   // the model is writing (without the voice: VM_SPEAK while its words come)
    int speak;                 // the voice is on
    int voice_msg, voice_busy; // the reply on screen is the one being spoken / the voice still has work for it
    int len;
    char text[16384];          // the newest reply, whole
    Spoken sp[256];            // which samples say which bytes of it
    int nsp;
    volatile int frames, late; // statistics for the log: frames shown, frames that missed their vertical blank
    volatile double draw_s, frame_max;
} vs;

static void vs_publish(int think, int listen, float level) {
    if (!vs.run) return;
    int i = last_ai(), n = i >= 0 ? (int)msgs[i].len : 0;
    if (n > (int)sizeof(vs.text) - 1) n = sizeof(vs.text) - 1;
    int busy = think || generating || tts_pending(), voice_busy = generating || tts.stage || tts_pending();
    scePthreadMutexLock(&vs.mu);
    vs.listen = listen, vs.listen_level = level;
    vs.busy = busy, vs.gen = generating, vs.speak = tts_loaded;
    vs.voice_msg = tts_loaded && i >= 0 && i == tts.msg, vs.voice_busy = voice_busy;
    if (n) memcpy(vs.text, msgs[i].text, n);
    vs.text[n] = 0, vs.len = n;
    vs.nsp = tts.nsp;
    memcpy(vs.sp, tts.sp, tts.nsp * sizeof(Spoken));
    vs.ver++;
    scePthreadMutexUnlock(&vs.mu);
}

static void* vs_main(void* arg) {
    (void)arg;
    static char text[16384];
    static Spoken sp[256];
    unsigned ver = ~0u;
    int len = 0, nsp = 0, listen = 0, busy = 0, gen = 0, speak = 0, vmsg = 0, vbusy = 0;
    float listen_level = 0, env = 0;
    double prev = now();
    while (vs.run) {
        scePthreadMutexLock(&vs.mu);
        if (vs.ver != ver) {
            ver = vs.ver;
            listen = vs.listen, listen_level = vs.listen_level, busy = vs.busy, gen = vs.gen, speak = vs.speak;
            vmsg = vs.voice_msg, vbusy = vs.voice_busy;
            len = vs.len;
            memcpy(text, vs.text, len + 1);
            nsp = vs.nsp;
            memcpy(sp, vs.sp, nsp * sizeof(Spoken));
        }
        scePthreadMutexUnlock(&vs.mu);
        double t = now(), dt = t - prev;
        prev = t;
        int playing = speak && aout_queued() > aout_played();
        // the voice's loudness, smoothed: the playback thread measures it every 5.3 ms, which flickers at 60 Hz;
        // it follows a rise within ~35 ms and falls back in ~110 ms
        float target = playing ? aout_level() : 0.0f;
        env += (target - env) * (1.0f - expf(-(float)dt / (target > env ? 0.035f : 0.11f)));
        int state;
        float level = 0;
        if (listen) {
            state = VM_LISTEN, level = listen_level;
        } else if (speak) {
            state = playing ? VM_SPEAK : busy ? VM_THINK : VM_IDLE;
            if (state == VM_SPEAK) level = 0.15f + 0.85f * env;
        } else {
            state = gen ? (len ? VM_SPEAK : VM_THINK) : busy ? VM_THINK : VM_IDLE;
            if (state == VM_SPEAK) level = 0.35f + 0.25f * sinf((float)t * 9);
        }
        // Garlic's words: while it speaks, only what has been said
        int n = vmsg && (vbusy || playing) ? spoken_bytes(text, len, sp, nsp, aout_played()) : len;
        char keep = text[n];
        text[n] = 0;
        draw_voice_mode(state, text, level, t);
        text[n] = keep;
        double left = toast_until - t;
        const char* tt = toast_text;
        if (left > 0 && tt) draw_toast(tt, left > 0.4 ? 1.0f : (float)(left / 0.4), 0);
        double t1 = now();
        present();
        vs.draw_s += t1 - t;
        if (dt > vs.frame_max) vs.frame_max = dt;
        if (vs.frames && dt > 0.025) vs.late++;   // more than one vertical blank (16.7 ms) since the last frame
        vs.frames++;
    }
    return NULL;
}

static void vs_start(void) {
    static int init;
    if (!init) scePthreadMutexInit(&vs.mu, NULL, "garlic_screen"), init = 1;
    vs.frames = vs.late = 0, vs.draw_s = vs.frame_max = 0;
    vs.run = 1;
    vs_publish(0, 0, 0);
    OrbisPthreadAttr attr;
    scePthreadAttrInit(&attr);
    scePthreadAttrSetstacksize(&attr, 1 << 20);
    if (scePthreadCreate(&vs.th, &attr, vs_main, NULL, "garlic_screen")) {
        vs.run = 0;
        say("voice screen: no thread, drawing stops");
        return;
    }
    scePthreadSetprio(vs.th, 256 + 128);   // above the GPU loop (normal), below the playback thread (256 + 64)
}

static void vs_stop(void) {
    if (!vs.run) return;
    vs.run = 0;
    scePthreadJoin(vs.th, NULL);
}

static void show_voice(int state) { vs_publish(state == VM_THINK, 0, 0); }

#ifdef GARLIC_SPEC
// The MTP head (speculative decoding, engine_spec.inc) lives in T3's weights memory outside voice mode.
static int spec_lent;

static void spec_lend(void) {
    if (spec_lent || !eng.t3.ready) return;
    double t0 = now();
    if (spec_load(&eng, eng.t3.model, eng.t3.gpu_size, pos) == 0) {   // (it follows the conversation from here)
        spec_lent = 1;
        say("spec: MTP head in T3's memory in %.2f s (following the conversation from position %d)", now() - t0, pos);
    }
}
#endif

static void voice_mode(void) {
    in_voice_mode = 1;
#ifdef GARLIC_SPEC
    if (spec_lent) {   // the MTP head had T3's weights memory: T3 reads its weights again (~3.5 s)
        double t0 = now();
        draw_loading(1.0f, tr(S_VOICE_LOADING));
        present();
        spec_unload(&eng);
        spec_lent = 0;
        int bad = t3_reload(&eng);
        say("voice: T3's weights read again in %.2f s (%s)", now() - t0, bad ? "FAILED" : "ok");
    }
#endif
    int speak = tts_enter();
    if (!speak && tts_ready) toast(S_VOICE_NO_TTS);
    vs_start();
    uint32_t prev = read_buttons();
    double t_stats = now();
    for (;;) {
        uint32_t b = read_buttons(), pressed = b & ~prev;
        prev = b;
        if (pressed & ORBIS_PAD_BUTTON_CIRCLE) {
            if (generating) finish_turn(0);
            break;
        }
        if (pressed & ORBIS_PAD_BUTTON_TRIANGLE) {
            tts_stop();
            if (generating) finish_turn(0);
            msgs_clear();
            pos = 0, need_close = 0, view.scroll = 0;
            tts.msg = -1;
        }
        if (pressed & (ORBIS_PAD_BUTTON_SQUARE | ORBIS_PAD_BUTTON_L2)) {   // talking (□ or L2 held) interrupts Garlic
            tts_stop();
            if (generating) finish_turn(0);
            static char text[2048];
            uint32_t hold = pressed & ORBIS_PAD_BUTTON_SQUARE ? ORBIS_PAD_BUTTON_SQUARE : ORBIS_PAD_BUTTON_L2;
            if (voice_to_text(hold, NULL, text, sizeof(text))) start_turn(text);
            prev = read_buttons();
        }
        // one unit of GPU work: the voice while less than TTS_AHEAD of audio is queued (and a piece is ready for
        // it), else the next token of the reply; then what the screen shows. The screen's thread draws meanwhile.
        if (speak && tts_pending() && (!generating || aout_queued() - aout_played() < TTS_AHEAD)) tts_work();
        else if (generating) generate_one();
        else if (pos == 0) system_ahead(8);   // idle: the system prompt of the next conversation
        else sceKernelUsleep(8000);           // nothing to do: the buttons are read ~100 times a second
        vs_publish(0, 0, 0);
        double tn = now();
        if (tn - t_stats >= 10.0) {   // how smooth the screen is, every 10 s
            int f = vs.frames, late = vs.late;
            say("voice screen: %.1f frames/s, %d of %d late, draw %.1f ms, longest gap %.0f ms", f / (tn - t_stats),
                late, f, f ? 1000 * vs.draw_s / f : 0.0, 1000 * vs.frame_max);
            vs.frames = vs.late = 0, vs.draw_s = vs.frame_max = 0;
            t_stats = tn;
        }
    }
    vs_stop();
    tts_leave();
    in_voice_mode = 0;
#ifdef GARLIC_SPEC
    spec_lend();   // the MTP head back into T3's memory (for the next conversation in the text chat)
#endif
}

static void push_to_talk(void) {
    static char text[2048];
    for (;;) {
        view_update();
        draw_chat(&view);
        ui_dim();
        ui_backdrop_save();
        if (!voice_to_text(ORBIS_PAD_BUTTON_SQUARE, NULL, text, sizeof(text))) return;
        uint32_t prev = read_buttons();
        for (;;) {   // Cross send, Triangle edit, Square again, Circle discard
            uint32_t b = read_buttons(), pressed = b & ~prev;
            prev = b;
            if (pressed & ORBIS_PAD_BUTTON_CROSS) {
                start_turn(text);
                return;
            }
            if (pressed & ORBIS_PAD_BUTTON_TRIANGLE) {
                if (keyboard(text, 1500, 1)) start_turn(text);
                return;
            }
            if (pressed & ORBIS_PAD_BUTTON_CIRCLE) return;
            if (pressed & ORBIS_PAD_BUTTON_SQUARE) break;   // record again (Square is held now)
            ui_backdrop_draw();
            draw_confirm(text);
            present();
        }
    }
}

// The console this runs on, first thing in the log: firmware, PS4 Pro or not, memory. Garlic is only tested on one
// console (FW 11.52); with this line a log from any other one says what it was.
static void boot_report(void) {
    OrbisKernelSwVersion v;
    memset(&v, 0, sizeof(v));
    v.Size = sizeof(v);
    int r = sceKernelGetSystemSwVersion(&v);
    v.VersionString[sizeof(v.VersionString) - 1] = 0;
    const char* fw = v.VersionString;
    while (*fw == ' ' || *fw == '0') fw++;   // " 11.520.001" -> "11.520.001"
    size_t flex = 0;
    ((int32_t(*)(size_t*))sceKernelAvailableFlexibleMemorySize)(&flex);
    const char* app = "stable";
    say("console: firmware %s (0x%08x%s); PS4 Pro hardware %d, Neo mode %d; direct memory %u MiB, system RAM free "
        "%u MiB; CPU TSC %.1f MHz, cpumode %d; app %s", r ? "?" : fw, (unsigned)v.Version, r ? ", query failed" : "",
        sceKernelHasNeoMode(), sceKernelIsNeoMode(), (unsigned)(sceKernelGetDirectMemorySize() >> 20),
        (unsigned)(flex >> 20), sceKernelGetTscFrequency() / 1e6, sceKernelGetCpumode(), app);
}


int main(void) {
    if (rename(LOG_FILE, LOG_PREV)) unlink(LOG_FILE);   // the last log of this app is kept, as LOG_PREV
    say("starting");
    boot_report();
    sceSysmoduleLoadModule(ORBIS_SYSMODULE_IME_DIALOG);
    sceCommonDialogInitialize();
    sceUserServiceInitialize(NULL);
    sceUserServiceGetInitialUser(&user_id);
    scePadInit();
    g_pad = scePadOpen(user_id, 0, 0, NULL);
    int lang = 1;
    sceSystemServiceParamGetInt(ORBIS_SYSTEM_SERVICE_PARAM_ID_LANG, &lang);
    ui_lang = lang_from_system(lang);
    say("user %d, pad %d, system language %d -> ui %d", user_id, g_pad, lang, ui_lang);
    int vr = video_init();
    if (vr) {
        say("video init failed: %d", vr);
        for (;;) sceKernelUsleep(1000000);
    }
    if (ui_init(ASSETS)) {
        say("ui assets missing: %s", ui_error);
        for (;;) sceKernelUsleep(1000000);
    }
    sceSystemServiceHideSplashScreen();   // the system shows pic1.png until the app says it is drawing
    say("video and fonts ready");
    loading((void*)tr(S_PREPARING), 0.0f, NULL);
    size_t tsize;
    uint8_t* tdata = read_all(ASSETS "/tokenizer.bin", &tsize);
    if (!tdata || tok_load(&tok, tdata, tsize)) show_error("tokenizer.bin");
    say("tokenizer: %u tokens", tok.n_vocab);
    eng_margin_mb = 64 - ui_screen_pool_mb();   // (the screen block came out of the system's 64 MiB)
    const char* eng_assets = ASSETS;
    if (engine_init(&eng, eng_assets, loading, (void*)tr(S_LOADING_MODEL))) show_error(eng.error);
    say("language model ready; GPU memory used %u of %u MiB; context cache %s, %u tokens (%u while voice mode "
        "borrows %u MiB)", (unsigned)(eng.next >> 20), (unsigned)(eng.mem_size >> 20),
        eng.tq ? "TurboQuant 4-bit" : "fp16 k + int8 v", eng.max_seq, eng.lend_ctx, (unsigned)(eng.kv_split >> 20));
    if (t3_init(&eng, ASSETS) == 0) {   // text to speech, first half (Chatterbox T3)
        say("T3 ready; GPU memory used %u of %u MiB", (unsigned)(eng.next >> 20), (unsigned)(eng.mem_size >> 20));
    } else {
        say("T3 not loaded: %s", eng.error);
    }
    loading((void*)tr(S_LOADING_VOICE), 0.99f, NULL);
    if (asr_init(&eng, ASSETS) == 0)
        say("speech to text ready: up to %u s of audio", eng.asr.max_frames * 8 / 100);
    else
        say("speech to text not available (missing or no GPU memory left)");
    size_t flex0 = 0;   // system memory left (the 9B's dictation weights live there; Python borrows them)
    ((int32_t(*)(size_t*))sceKernelAvailableFlexibleMemorySize)(&flex0);
    say("GPU memory used: %u of %u MiB; system RAM free: %u MiB", (unsigned)(eng.next >> 20),
        (unsigned)(eng.mem_size >> 20), (unsigned)(flex0 >> 20));
#ifdef GARLIC_SPEC
    if (spec_init(&eng, eng_assets) == 0) spec_lend();   // (before the start-up tests: they use it)
    else say("spec: no MTP head (mtp.idx, the q4s model or a kernel missing)");
#endif
    kv_split_test();
    tts_boot();
    size_t flex1 = 0;
    ((int32_t(*)(size_t*))sceKernelAvailableFlexibleMemorySize)(&flex1);
    say("voice %s; system RAM free: %u MiB", tts_ready ? "ready" : "not available", (unsigned)(flex1 >> 20));
    view.temperature = 0.7f;
    think_config();

    uint32_t prev = 0;
    int scroll_held = 0;
    for (;;) {
        uint32_t buttons = read_buttons();
        uint32_t pressed = buttons & ~prev;
        prev = buttons;
        if (generating) {
            if (pressed & ORBIS_PAD_BUTTON_CIRCLE) finish_turn(0);
            else generate_one();
        } else if (pressed & (ORBIS_PAD_BUTTON_CROSS | ORBIS_PAD_BUTTON_OPTIONS)) {
            static char text[2048];
            int cross = (pressed & ORBIS_PAD_BUTTON_CROSS) != 0;
            if (cross ? keyboard(text, 1500, 0) : ask_user(text, sizeof(text))) {
                say("user wrote %d bytes", (int)strlen(text));
                start_turn(text);
            }
            prev = read_buttons();
        } else if (pressed & ORBIS_PAD_BUTTON_L1) {
            voice_mode();
            prev = read_buttons();
        } else if ((pressed & ORBIS_PAD_BUTTON_R1) && code_msg() >= 0) {
            run_python();
            prev = read_buttons();
        } else if (pressed & ORBIS_PAD_BUTTON_SQUARE) {
            push_to_talk();
            prev = read_buttons();
        } else if (pressed & ORBIS_PAD_BUTTON_TRIANGLE) {
            msgs_clear();
            pos = 0;
            need_close = 0;
            view.scroll = 0;
        }
        else if (pressed & (ORBIS_PAD_BUTTON_R3 | ORBIS_PAD_BUTTON_TOUCH_PAD)) think_toggle();
        if (!generating && !(buttons & (ORBIS_PAD_BUTTON_UP | ORBIS_PAD_BUTTON_DOWN))) hist_ahead(8);
        if (!generating && inj_ask_ready) {   // (test aid: "ask" in /data/garlic_keys.txt)
            say("user wrote %d bytes (test file)", (int)strlen(inj_ask));
            start_turn(inj_ask);
            inj_ask_ready = 0;
        }
        if (pressed & ORBIS_PAD_BUTTON_L2) view.temperature = view.temperature > 0.05f ? view.temperature - 0.1f : 0.0f;
        if (pressed & ORBIS_PAD_BUTTON_R2) view.temperature = view.temperature < 1.45f ? view.temperature + 0.1f : 1.5f;
        int up = (buttons & ORBIS_PAD_BUTTON_UP) != 0, down = (buttons & ORBIS_PAD_BUTTON_DOWN) != 0;
        scroll_held = up || down ? scroll_held + 1 : 0;
        int step = 14 + (scroll_held > 40 ? 40 : scroll_held);   // speeds up while held
        if (up) view.scroll += step;
        if (down) view.scroll = view.scroll > step ? view.scroll - step : 0;
        show_chat();
    }
}
