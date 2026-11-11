// Garlic LLM user interface: everything that draws (theme, fonts, markdown, chat, keyboard, voice, loading),
// the chat messages and the on-screen keyboard's logic. No console calls here, so the same code also runs on
// Windows.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define SCREEN_W 1920
#define SCREEN_H 1080

// ---- provided by the platform (main.c on the PS4) ----
extern uint32_t* back;                               // the frame being drawn, 0x80RRGGBB
void say(const char* fmt, ...);                      // log
uint8_t* read_all(const char* path, size_t* size);   // whole file, malloc'd

// ---- languages: chosen from the console's system language ----
enum { LANG_ES, LANG_EN, LANG_PT, LANG_RU, N_LANGS };
extern int ui_lang;
int lang_from_system(int system_lang);   // OrbisSystemParamLanguage -> LANG_*

typedef enum {
    S_TAGLINE, S_EMPTY_HINT, S_WRITE, S_TALK_HOLD, S_STOP, S_NEW_CHAT, S_SCROLL, S_TEMPERATURE, S_CONTEXT,
    S_READING, S_CONTEXT_FULL, S_PLACEHOLDER, S_KB_TYPE, S_KB_DELETE, S_KB_SPACE, S_KB_SHIFT, S_KB_EMOJI,
    S_KB_LETTERS, S_KB_DICTATE, S_KB_SEND, S_KB_CLOSE, S_SPACE_KEY, S_SEND_KEY, S_LISTENING, S_RELEASE_TO_END,
    S_TRANSCRIBING, S_YOUR_MESSAGE, S_SEND, S_EDIT, S_RETRY, S_DISCARD, S_NO_MIC, S_NO_SPEECH, S_ASR_OFF,
    S_HOLD_LONGER, S_PREPARING, S_LOADING_MODEL, S_LOADING_VOICE, S_ERROR_TITLE, S_ERROR_HINT, S_OFFLINE,
    S_SYSTEM_PROMPT, S_LOCAL_GPU, S_PY_RUN, S_PY_EXIT, S_PY_CLICK, S_PY_RCLICK, S_PY_MOVE, S_PY_KEYS, S_PY_BACK,
    S_PY_AGAIN, S_PY_FIX, S_PY_DONE, S_PY_ERROR, S_PY_STOPPED, S_PY_STARTING, S_PY_FIX_MSG, S_PY_FIX_ASK, S_PY_INPUT, S_VOICE_MODE, S_THINKING, S_LISTENING_SHORT, S_VOICE_LOADING, S_VOICE_NO_TTS,
    S_THINK_OFF_HINT, S_THINK_ON_HINT, S_THINK_ON_TOAST, S_THINK_OFF_TOAST, S_THOUGHT, S_THINK_CUT, S_THINK_CHIP, N_STRINGS
} StrId;
const char* tr(StrId id);

// ---- start-up ----
int ui_init(const char* assets_dir);   // fonts, logo, background; 0 = ok
extern const char* ui_error;

// ---- the conversation ----
typedef struct Layout Layout;
enum { MSG_AI, MSG_USER, MSG_NOTE };
typedef struct {
    int who;
    char* text;
    size_t len, cap;
    Layout* lay;      // cached text layout (rebuilt when the text or width changes)
    int think_n;      // the reply was thought first, this many tokens for think_s seconds (think_cut: up to the
    float think_s;    // cap); a short line over the answer says so. 0 = no reasoning.
    int think_cut;
} Msg;

#define MAX_MSGS 200
extern Msg msgs[MAX_MSGS];
extern int n_msgs;
Msg* msg_new(int who);
void msg_append(Msg* m, const char* s, size_t n);
void msgs_clear(void);

typedef struct {
    const char* model;      // "Qwen3.5-9B"
    int pos, max_seq;       // tokens used / context size
    int generating, reading;
    double tok_per_s;
    float temperature;
    int scroll;             // pixels scrolled up from the newest line (draw_chat clamps it)
    double time;            // seconds, for animations
    const char* toast;      // short message at the bottom, NULL = none
    float toast_alpha;
    int can_run;            // the conversation has Python code: R1 runs it
    // "Think" (think_avail: the model can do it): R3 turns thinking on (think_on, up to think_cap tokens); while
    // the newest reply is still reasoning (thinking), its last lines (think_text, think_len bytes), think_n tokens
    // and think_s seconds are shown in its place
    int think_avail, think_on, think_cap, thinking, think_n;
    double think_s;
    const char* think_text;
    size_t think_len;
} ChatView;

void draw_chat(ChatView* v);
void ui_dim(void);              // darkens the chat area (behind the keyboard, voice and confirm panels)
void ui_backdrop_save(void);    // keeps the current picture to draw modal screens over it every frame
void ui_backdrop_draw(void);
// An 8 MB screen buffer from the screen block (direct memory, CPU cached; outside the system RAM), else malloc.
// ui_screen_pool_mb sets the block up (the first call) and says its size, 0 without it.
uint32_t* ui_screen_buf(void);
int ui_screen_pool_mb(void);
void ui_free_caches(void);      // gives back the backdrop and keyboard copies (8 MB each); they return when needed
void draw_loading(float frac, const char* what);
void draw_error(const char* what);
void draw_toast(const char* text, float alpha, int bottom);   // bottom 0 = above the footer

// ---- footer hints ----
enum { BTN_CROSS, BTN_CIRCLE, BTN_TRIANGLE, BTN_SQUARE, BTN_L1, BTN_R1, BTN_L2, BTN_R2, BTN_UPDOWN, BTN_OPTIONS, BTN_R3 };
typedef struct {
    int btn;
    StrId label;
} Hint;
void draw_footer(const Hint* hints, int n, const char* right);

// ---- the on-screen keyboard (drawn by the app, driven by the controller) ----
enum { KB_ES, KB_EN, KB_PT, KB_RU, N_KB_LAYOUTS };
typedef struct {
    int layout;              // KB_*
    int page;                // 0 = letters, 1.. = emoji category
    int row, col;
    int shift;               // one letter in capitals
    const char* dead;        // pending accent ("´", "~", "^", "`") or NULL
    int last_emoji_page;
} Kb;
void kb_init(Kb* kb, int layout);
void kb_move(Kb* kb, int dx, int dy);
int kb_press(Kb* kb, char* out, int max);   // Cross on the selected key; returns 1 for Send
void kb_toggle_emoji(Kb* kb);
void kb_append(char* out, int max, const char* s);
void kb_backspace(char* out);
void draw_keyboard(const Kb* kb, const char* text, double time);

// ---- voice ----
enum { VOICE_RECORDING, VOICE_TRANSCRIBING };
void draw_voice(int phase, const float* levels, int n_levels, float secs, float max_secs, double time);
void draw_confirm(const char* text);   // the transcribed message, to send / edit / retry / discard

// ---- the Python runner's screens (ui_run.inc, used by pyrun.c) ----
// A program's window (turtle, tkinter, pygame) is a canvas of w x h units, scaled to fit between the header and the
// footer; everything is drawn at the TV's resolution, so shapes and text stay sharp.
void cv_begin(int w, int h, uint32_t bg);
void cv_clear(uint32_t rgb);
void cv_rect(float x, float y, float w, float h, int fill, uint32_t fill_rgb, float lw, uint32_t line_rgb);
void cv_oval(float x, float y, float w, float h, int fill, uint32_t fill_rgb, float lw, uint32_t line_rgb);
void cv_line(float x1, float y1, float x2, float y2, float lw, uint32_t rgb);
void cv_polygon(const float* xy, int n, int fill, uint32_t fill_rgb, float lw, uint32_t line_rgb);
void cv_text(float x, float y, const char* s, float size, int bold, uint32_t rgb, int anchor);   // anchor 0-8: nw n ne w c e sw s se
void cv_text_size(const char* s, float size, int bold, float* w, float* h);
void cv_to_canvas(float sx, float sy, float* x, float* y);
void cv_to_screen(float x, float y, float* sx, float* sy);
void draw_run_chrome(const char* title, int mouse, const int* btns, const char* const* labels, int n);   // header, footer
void cv_fill_alpha(float x, float y, float w, float h, uint32_t rgb, int alpha);
void cursor_draw(float sx, float sy);                            // the pointer, taken away again by cursor_undraw()
void cursor_undraw(void);
void draw_console(const char* text, size_t len, const char* title, int bottom, int scroll);   // print() output
void draw_run_end(const char* title, const char* error, int stopped);   // card over the last picture
void draw_run_intro(const char* title, float t);                 // transition from the chat (t 0..1)

// ---- voice mode: talk to Garlic without the chat; only its words are shown ----
enum { VM_IDLE, VM_LISTEN, VM_THINK, VM_SPEAK };
void draw_voice_mode(int state, const char* reply, float level, double time);
