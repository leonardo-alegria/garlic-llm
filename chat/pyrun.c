// Python runner (see pyrun.h): MicroPython embedded, its console, its input() and the _garlic module.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "py/builtin.h"
#include "py/compile.h"
#include "py/gc.h"
#include "py/lexer.h"
#include "py/mperrno.h"
#include "py/mphal.h"
#include "py/runtime.h"
#include "py/stackctrl.h"
#include "shared/runtime/gchelper.h"

#include "pyrun.h"
#include "ui.h"

static const PyHost* H;
static char* con;              // console text (print output and input echo)
static size_t con_len, con_cap;
static int con_dirty, canvas, mouse_on, chrome_dirty;
static char title[96];
static uint32_t prev_b;
static float mouse_x = SCREEN_W / 2, mouse_y = SCREEN_H / 2;
static double last_pump, last_present;
static volatile int stop_flag;
static char err[4096];
static size_t err_len;
static const char* program;   // the running source (the compatibility modules read which keys it uses)
static int hint_btn[8], n_hints;
static char hint_label[8][32];
static const char* hint_ptr[8];

typedef struct {
    uint8_t down, btn;
} Ev;
static Ev evq[128];
static int n_ev;
static const char* btn_names[13] = {"up", "down", "left", "right", "cross", "circle", "square", "triangle",
                                    "l1", "r1", "l2", "r2", "options"};

uint32_t garlic_random_seed(void) { return (uint32_t)(H ? H->now() * 1e6 : 12345) ^ 0x9E3779B9u; }

// ---- console --------------------------------------------------------------------------------------------------

static void con_append(const char* s, size_t n) {
    if (con_len + n + 1 > con_cap) {
        con_cap = (con_len + n + 1) * 2 + 4096;
        con = realloc(con, con_cap);
    }
    memcpy(con + con_len, s, n);
    con_len += n;
    con[con_len] = 0;
    if (con_len > (4u << 20)) {   // keep the last 2 MB of a very chatty program
        size_t cut = con_len - (2u << 20);
        memmove(con, con + cut, con_len - cut + 1);
        con_len -= cut;
    }
    con_dirty = 1;
}

static void con_strn(void* env, const char* s, size_t n) { (void)env, con_append(s, n); }
static const mp_print_t con_print = {NULL, con_strn};

static void err_strn(void* env, const char* s, size_t n) {
    (void)env;
    if (err_len + n >= sizeof(err)) n = sizeof(err) - 1 - err_len;
    memcpy(err + err_len, s, n);
    err_len += n;
    err[err_len] = 0;
}
static const mp_print_t err_print = {NULL, err_strn};

// ---- frames and the controller ----------------------------------------------------------------------------------

// Shows the current picture and reads the controller; raises KeyboardInterrupt when ○ was pressed.
static void pump(void) {
    float lx = 0, ly = 0;
    uint32_t b = H->buttons(&lx, &ly);
    uint32_t pressed = b & ~prev_b, released = prev_b & ~b;
    prev_b = b;
    if (pressed & PB_CIRCLE) stop_flag = 1;
    double t = H->now();
    float dt = (float)(t - last_pump);
    if (dt > 0.1f || dt < 0) dt = 0.1f;
    last_pump = t;
    if (mouse_on) {   // the left stick moves the pointer, faster the further it is pushed
        float m = sqrtf(lx * lx + ly * ly);
        if (m > 0.18f) {
            float speed = 1300.0f * (m - 0.18f) / 0.82f * (m - 0.18f) / 0.82f + 120.0f;
            mouse_x += lx / m * speed * dt, mouse_y += ly / m * speed * dt;
        }
        if (mouse_x < 0) mouse_x = 0;
        if (mouse_y < 0) mouse_y = 0;
        if (mouse_x > SCREEN_W - 1) mouse_x = SCREEN_W - 1;
        if (mouse_y > SCREEN_H - 1) mouse_y = SCREEN_H - 1;
    }
    for (int i = 0; i < 13; i++) {
        if (i == 5) continue;   // ○ is the runner's own
        if ((pressed >> i & 1) && n_ev < 128) evq[n_ev++] = (Ev){1, (uint8_t)i};
        if ((released >> i & 1) && n_ev < 128) evq[n_ev++] = (Ev){0, (uint8_t)i};
    }
    if (canvas) {
        if (chrome_dirty) draw_run_chrome(title, mouse_on, hint_btn, hint_ptr, n_hints), chrome_dirty = 0;
        if (mouse_on) cursor_draw(mouse_x, mouse_y);
        H->present();
        if (mouse_on) cursor_undraw();
    } else {
        if (con_dirty) draw_console(con, con_len, title, SCREEN_H - 40, 0), con_dirty = 0;
        H->present();
    }
    last_present = H->now();
    if (stop_flag) mp_raise_type(&mp_type_KeyboardInterrupt);
}

static void wait_seconds(double s) {
    double end = H->now() + s;
    do pump();
    while (H->now() < end - 0.004);
}

void py_request_stop(void) {
    stop_flag = 1;
    mp_sched_keyboard_interrupt();
}

// ---- MicroPython's port functions ---------------------------------------------------------------------------------

void mp_hal_stdout_tx_strn_cooked(const char* s, size_t n) {
    con_append(s, n);
    if (H && H->now() - last_present > 0.08) pump();   // let long-running text programs show their output
}
mp_uint_t mp_hal_stdout_tx_strn(const char* s, size_t n) {
    mp_hal_stdout_tx_strn_cooked(s, n);
    return n;
}
void mp_hal_stdout_tx_str(const char* s) { mp_hal_stdout_tx_strn_cooked(s, strlen(s)); }
int mp_hal_stdin_rx_chr(void) { return '\n'; }
uintptr_t mp_hal_stdio_poll(uintptr_t flags) { return flags & 4; }   // stdout is always writable (MP_STREAM_POLL_WR)
mp_uint_t mp_hal_ticks_ms(void) { return (mp_uint_t)(H->now() * 1000.0); }
mp_uint_t mp_hal_ticks_us(void) { return (mp_uint_t)(H->now() * 1e6); }
mp_uint_t mp_hal_ticks_cpu(void) { return (mp_uint_t)(H->now() * 1e6); }
uint64_t mp_hal_time_ns(void) { return 1793000000000000000ull + (uint64_t)(H->now() * 1e9); }
void mp_hal_delay_ms(mp_uint_t ms) { wait_seconds(ms / 1000.0); }
void mp_hal_delay_us(mp_uint_t us) {
    if (us >= 2000) wait_seconds(us / 1e6);
}

void mp_hal_set_interrupt_char(int c) { (void)c; }

void gc_collect(void) {
    gc_collect_start();
    gc_helper_collect_regs_and_stack();
    gc_collect_end();
}

void nlr_jump_fail(void* val) {
    (void)val;
    for (;;) {
    }
}

// open(): programs cannot read files; writing (saving a high score) goes nowhere (replaced in BOOT by a version that
// returns io.StringIO for writing).
static mp_obj_t garlic_open(size_t n, const mp_obj_t* a, mp_map_t* kw) {
    (void)n, (void)a, (void)kw;
    mp_raise_OSError(MP_ENOENT);
}
MP_DEFINE_CONST_FUN_OBJ_KW(mp_builtin_open_obj, 1, garlic_open);

mp_import_stat_t mp_import_stat(const char* path) {
    struct stat st;
    if (stat(path, &st) == 0) return S_ISDIR(st.st_mode) ? MP_IMPORT_STAT_DIR : MP_IMPORT_STAT_FILE;
    return MP_IMPORT_STAT_NO_EXIST;
}

mp_lexer_t* mp_lexer_new_from_file(qstr filename) {
    FILE* f = fopen(qstr_str(filename), "rb");
    if (!f) mp_raise_OSError(MP_ENOENT);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = m_new(char, n > 0 ? n : 1);
    size_t got = fread(buf, 1, n, f);
    fclose(f);
    return mp_lexer_new_from_str_len(filename, buf, got, n > 0 ? n : 1);
}

// ---- the _garlic module ------------------------------------------------------------------------------------------

static uint32_t color_arg(mp_obj_t o) { return (uint32_t)mp_obj_get_int(o) & 0xFFFFFF; }

static mp_obj_t g_begin(size_t n, const mp_obj_t* a) {   // begin(w, h, bg)
    canvas = 1, chrome_dirty = 1;
    cv_begin(mp_obj_get_int(a[0]), mp_obj_get_int(a[1]), n > 2 ? color_arg(a[2]) : 0);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(g_begin_obj, 2, 3, g_begin);

static mp_obj_t g_clear(mp_obj_t c) {
    cv_clear(color_arg(c));
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(g_clear_obj, g_clear);

// rect/oval(x, y, w, h, fill or None, outline or None, width)
static mp_obj_t g_shape(size_t n, const mp_obj_t* a, int oval) {
    int f = a[4] != mp_const_none, o = n > 5 && a[5] != mp_const_none;
    float lw = o ? (n > 6 ? (float)mp_obj_get_float(a[6]) : 1.0f) : 0;
    void (*fn)(float, float, float, float, int, uint32_t, float, uint32_t) = oval ? cv_oval : cv_rect;
    fn((float)mp_obj_get_float(a[0]), (float)mp_obj_get_float(a[1]), (float)mp_obj_get_float(a[2]),
       (float)mp_obj_get_float(a[3]), f, f ? color_arg(a[4]) : 0, lw, o ? color_arg(a[5]) : 0);
    return mp_const_none;
}
static mp_obj_t g_rect(size_t n, const mp_obj_t* a) {
    if (a[4] != mp_const_none) {   // a translucent fill: 0xTTRRGGBB with TT = 255 - alpha
        uint32_t c = (uint32_t)mp_obj_get_int(a[4]);
        if (c >> 24) {
            cv_fill_alpha((float)mp_obj_get_float(a[0]), (float)mp_obj_get_float(a[1]), (float)mp_obj_get_float(a[2]),
                          (float)mp_obj_get_float(a[3]), c & 0xFFFFFF, 255 - (int)(c >> 24));
            mp_obj_t b[7] = {a[0], a[1], a[2], a[3], mp_const_none, n > 5 ? a[5] : mp_const_none, n > 6 ? a[6] : mp_obj_new_int(1)};
            return b[5] == mp_const_none ? mp_const_none : g_shape(7, b, 0);
        }
    }
    return g_shape(n, a, 0);
}
static mp_obj_t g_oval(size_t n, const mp_obj_t* a) { return g_shape(n, a, 1); }
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(g_rect_obj, 5, 7, g_rect);
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(g_oval_obj, 5, 7, g_oval);

static mp_obj_t g_line(size_t n, const mp_obj_t* a) {   // line(x1, y1, x2, y2, color, width)
    cv_line((float)mp_obj_get_float(a[0]), (float)mp_obj_get_float(a[1]), (float)mp_obj_get_float(a[2]),
            (float)mp_obj_get_float(a[3]), n > 5 ? (float)mp_obj_get_float(a[5]) : 1.0f, color_arg(a[4]));
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(g_line_obj, 5, 6, g_line);

static mp_obj_t g_poly(size_t n, const mp_obj_t* a) {   // poly([x0, y0, x1, y1, ...], fill or None, outline or None, width)
    size_t len;
    mp_obj_t* items;
    mp_obj_get_array(a[0], &len, &items);
    static float xy[1024];
    int m = 0;
    for (size_t i = 0; i + 1 < len && m < 512; i += 2, m++)
        xy[2 * m] = (float)mp_obj_get_float(items[i]), xy[2 * m + 1] = (float)mp_obj_get_float(items[i + 1]);
    int f = a[1] != mp_const_none, o = n > 2 && a[2] != mp_const_none;
    cv_polygon(xy, m, f, f ? color_arg(a[1]) : 0, o ? (n > 3 ? (float)mp_obj_get_float(a[3]) : 1.0f) : 0,
               o ? color_arg(a[2]) : 0);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(g_poly_obj, 2, 4, g_poly);

static mp_obj_t g_text(size_t n, const mp_obj_t* a) {   // text(x, y, s, size, bold, color, anchor)
    cv_text((float)mp_obj_get_float(a[0]), (float)mp_obj_get_float(a[1]), mp_obj_str_get_str(a[2]),
            (float)mp_obj_get_float(a[3]), mp_obj_is_true(a[4]), color_arg(a[5]), n > 6 ? mp_obj_get_int(a[6]) : 0);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(g_text_obj, 6, 7, g_text);

static mp_obj_t g_text_size(mp_obj_t s, mp_obj_t size, mp_obj_t bold) {
    float w, h;
    cv_text_size(mp_obj_str_get_str(s), (float)mp_obj_get_float(size), mp_obj_is_true(bold), &w, &h);
    mp_obj_t t[2] = {mp_obj_new_float(w), mp_obj_new_float(h)};
    return mp_obj_new_tuple(2, t);
}
static MP_DEFINE_CONST_FUN_OBJ_3(g_text_size_obj, g_text_size);

static mp_obj_t g_frame(void) {
    pump();
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_0(g_frame_obj, g_frame);

static mp_obj_t g_wait(mp_obj_t s) {
    wait_seconds(mp_obj_get_float(s));
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(g_wait_obj, g_wait);

static mp_obj_t g_ticks(void) { return mp_obj_new_float(H->now()); }
static MP_DEFINE_CONST_FUN_OBJ_0(g_ticks_obj, g_ticks);

static mp_obj_t g_events(void) {   // [(kind "down"/"up", button, x, y)], x y = pointer in window units
    mp_obj_t list = mp_obj_new_list(0, NULL);
    float x, y;
    cv_to_canvas(mouse_x, mouse_y, &x, &y);
    for (int i = 0; i < n_ev; i++) {
        mp_obj_t t[4] = {MP_OBJ_NEW_QSTR(qstr_from_str(evq[i].down ? "down" : "up")),
                         MP_OBJ_NEW_QSTR(qstr_from_str(btn_names[evq[i].btn])), mp_obj_new_float(x), mp_obj_new_float(y)};
        mp_obj_list_append(list, mp_obj_new_tuple(4, t));
    }
    n_ev = 0;
    return list;
}
static MP_DEFINE_CONST_FUN_OBJ_0(g_events_obj, g_events);

static mp_obj_t g_held(void) {
    mp_obj_t list = mp_obj_new_list(0, NULL);
    for (int i = 0; i < 13; i++)
        if (i != 5 && (prev_b >> i & 1)) mp_obj_list_append(list, MP_OBJ_NEW_QSTR(qstr_from_str(btn_names[i])));
    return list;
}
static MP_DEFINE_CONST_FUN_OBJ_0(g_held_obj, g_held);

static mp_obj_t g_mouse(void) {
    float x, y;
    cv_to_canvas(mouse_x, mouse_y, &x, &y);
    mp_obj_t t[2] = {mp_obj_new_float(x), mp_obj_new_float(y)};
    return mp_obj_new_tuple(2, t);
}
static MP_DEFINE_CONST_FUN_OBJ_0(g_mouse_obj, g_mouse);

static mp_obj_t g_mouse_mode(mp_obj_t on) {
    int v = mp_obj_is_true(on);
    if (v != mouse_on) mouse_on = v, chrome_dirty = 1;
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(g_mouse_mode_obj, g_mouse_mode);

static mp_obj_t g_set_title(mp_obj_t s) {
    snprintf(title, sizeof(title), "%s", mp_obj_str_get_str(s));
    chrome_dirty = 1;
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(g_set_title_obj, g_set_title);

static mp_obj_t g_console_clear(void) {
    con_len = 0;
    if (con) con[0] = 0;
    con_dirty = 1;
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_0(g_console_clear_obj, g_console_clear);

static mp_obj_t g_input(size_t n, const mp_obj_t* a) {
    if (n) mp_obj_print_helper(&con_print, a[0], PRINT_STR);
    if (!canvas) draw_console(con, con_len, title, 356, 0), con_dirty = 0;   // keep the question above the keyboard
    H->present();
    static char buf[1024];
    if (!H->keyboard(buf, sizeof(buf))) mp_raise_type(&mp_type_KeyboardInterrupt);
    con_append(buf, strlen(buf));
    con_append("\n", 1);
    if (canvas) chrome_dirty = 1;
    prev_b = H->buttons(NULL, NULL);   // the button that closed the keyboard is not a game press
    return mp_obj_new_str(buf, strlen(buf));
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(g_input_obj, 0, 1, g_input);

static mp_obj_t g_lang(void) {
    static const char* codes[] = {"es", "en", "pt", "ru"};
    return MP_OBJ_NEW_QSTR(qstr_from_str(codes[ui_lang >= 0 && ui_lang < N_LANGS ? ui_lang : 1]));
}
static MP_DEFINE_CONST_FUN_OBJ_0(g_lang_obj, g_lang);

static mp_obj_t g_source(void) { return mp_obj_new_str(program, strlen(program)); }
static MP_DEFINE_CONST_FUN_OBJ_0(g_source_obj, g_source);

static mp_obj_t g_hints(mp_obj_t list) {   // hints([(button, label), ...]) for the footer
    size_t len;
    mp_obj_t* items;
    mp_obj_get_array(list, &len, &items);
    static const char* names[] = {"up", "down", "left", "right", "cross", "circle", "square", "triangle", "l1", "r1", "l2", "r2"};
    static const int btns[] = {BTN_UPDOWN, BTN_UPDOWN, BTN_UPDOWN, BTN_UPDOWN, BTN_CROSS, BTN_CIRCLE, BTN_SQUARE, BTN_TRIANGLE, BTN_L1, BTN_R1, BTN_L2, BTN_R2};
    n_hints = 0;
    for (size_t i = 0; i < len && n_hints < 8; i++) {
        size_t m;
        mp_obj_t* pair;
        mp_obj_get_array(items[i], &m, &pair);
        const char* b = mp_obj_str_get_str(pair[0]);
        for (int k = 0; k < 12; k++)
            if (!strcmp(b, names[k])) {
                hint_btn[n_hints] = btns[k];
                snprintf(hint_label[n_hints], 32, "%s", mp_obj_str_get_str(pair[1]));
                hint_ptr[n_hints] = hint_label[n_hints];
                n_hints++;
                break;
            }
    }
    chrome_dirty = 1;
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(g_hints_obj, g_hints);

static void add_fn(mp_obj_t mod, const char* name, const void* fn) { mp_store_attr(mod, qstr_from_str(name), MP_OBJ_FROM_PTR(fn)); }

static void make_module(void) {
    mp_obj_t m = mp_obj_new_module(qstr_from_str("_garlic"));
    add_fn(m, "begin", &g_begin_obj), add_fn(m, "clear", &g_clear_obj), add_fn(m, "rect", &g_rect_obj);
    add_fn(m, "oval", &g_oval_obj), add_fn(m, "line", &g_line_obj), add_fn(m, "poly", &g_poly_obj);
    add_fn(m, "text", &g_text_obj), add_fn(m, "text_size", &g_text_size_obj), add_fn(m, "frame", &g_frame_obj);
    add_fn(m, "wait", &g_wait_obj), add_fn(m, "ticks", &g_ticks_obj), add_fn(m, "events", &g_events_obj);
    add_fn(m, "held", &g_held_obj), add_fn(m, "mouse", &g_mouse_obj), add_fn(m, "mouse_mode", &g_mouse_mode_obj);
    add_fn(m, "set_title", &g_set_title_obj), add_fn(m, "console_clear", &g_console_clear_obj);
    add_fn(m, "input", &g_input_obj), add_fn(m, "source", &g_source_obj), add_fn(m, "hints", &g_hints_obj);
    add_fn(m, "lang", &g_lang_obj);
}

// ---- running ---------------------------------------------------------------------------------------------------

static int exec_source(const char* src, size_t len, const char* name) {
    nlr_buf_t nlr;
    if (nlr_push(&nlr) == 0) {
        mp_lexer_t* lex = mp_lexer_new_from_str_len(qstr_from_str(name), src, len, 0);
        qstr source_name = lex->source_name;
        mp_parse_tree_t tree = mp_parse(lex, MP_PARSE_FILE_INPUT);
        mp_obj_t fun = mp_compile(&tree, source_name, false);
        mp_call_function_0(fun);
        nlr_pop();
        return PY_OK;
    }
    mp_obj_t exc = MP_OBJ_FROM_PTR(nlr.ret_val);
    const mp_obj_type_t* type = mp_obj_get_type(exc);
    if (mp_obj_is_subclass_fast(MP_OBJ_FROM_PTR(type), MP_OBJ_FROM_PTR(&mp_type_SystemExit))) return PY_OK;
    if (stop_flag || mp_obj_is_subclass_fast(MP_OBJ_FROM_PTR(type), MP_OBJ_FROM_PTR(&mp_type_KeyboardInterrupt)))
        return PY_STOPPED;
    err_len = 0;
    mp_obj_print_exception(&err_print, exc);
    con_append("\n", 1);
    mp_obj_print_exception(&con_print, exc);
    return PY_ERROR;
}

const char* py_error(void) { return err; }

int py_run(const PyHost* host, const char* source, const char* name) {
    // The heap: 64 MiB if the system RAM has it, else less: Parakeet's dictation weights (338 MiB) live in that
    // RAM, so 64 MiB are not always free.
    static char* heap;
    static size_t heap_size;
    static const unsigned mib[] = {64, 48, 32, 24, 16, 12, 8};
    for (size_t i = 0; !heap && i < sizeof(mib) / sizeof(mib[0]); i++) {
#ifdef PYRUN_TEST_HEAP_MIB   // (for testing: as if only that much were free)
        if (mib[i] > PYRUN_TEST_HEAP_MIB) continue;
#endif
        if ((heap = host->heap_alloc ? host->heap_alloc((size_t)mib[i] << 20) : malloc((size_t)mib[i] << 20))) {
            heap_size = (size_t)mib[i] << 20;
            void say(const char* fmt, ...);
            say("python: heap of %u MiB", mib[i]);
        }
    }
    if (!heap) {
        snprintf(err, sizeof(err), "MemoryError: no memory for Python");
        return PY_ERROR;
    }
    H = host;
    con_len = 0, con_dirty = 1, canvas = 0, mouse_on = 0, chrome_dirty = 1, n_ev = 0, stop_flag = 0, err[0] = 0, err_len = 0;
    program = source, n_hints = 0;
    if (con) con[0] = 0;
    snprintf(title, sizeof(title), "%s", name);
    prev_b = H->buttons(NULL, NULL);
    last_pump = last_present = H->now();
    mouse_x = SCREEN_W / 2, mouse_y = SCREEN_H / 2;
    volatile int stack_top;
    mp_stack_set_top((void*)&stack_top);
    mp_stack_set_limit(host->stack_limit);
    gc_init(heap, heap + heap_size);
    mp_init();
    mp_obj_list_append(mp_sys_path, mp_obj_new_str(host->modules, strlen(host->modules)));
    make_module();
    static const char boot[] =
        "import builtins, _garlic, io\n"
        "builtins.input = _garlic.input\n"
        "def _open(name, mode='r', *args, **kw):\n"
        "    if 'w' in mode or 'a' in mode or 'x' in mode:\n"
        "        return io.BytesIO() if 'b' in mode else io.StringIO()\n"
        "    raise OSError(2, 'No such file: ' + str(name))\n"
        "builtins.open = _open\n"
        "_print = print\n"
        "def _p(*args, sep=' ', end='\\n', file=None, flush=False):\n"   // CPython's flush= and file= (all to the console)
        "    _print(*args, sep=sep, end=end)\n"
        "builtins.print = _p\n"
        "del _open, _p, builtins, _garlic, io\n";
    int r = exec_source(boot, sizeof(boot) - 1, "<garlic>");
    if (r == PY_OK) r = exec_source(source, strlen(source), name);
    if (!canvas) draw_console(con, con_len, title, SCREEN_H - 40, 0);   // the final text, for the end card
    mp_deinit();
    // back to the system after each run (the dictation weights need that memory back, and free() alone may keep
    // it in the process)
    if (host->heap_free) host->heap_free(heap, heap_size);
    else free(heap);
    heap = NULL, heap_size = 0;
    return r;
}

// ---- finding the code in a reply -----------------------------------------------------------------------------------

int py_find_code(const char* text, char* out, int max, char* ttl, int tmax) {
    const char* best = NULL;
    size_t best_len = 0;
    int best_py = 0;
    for (const char* p = text; (p = strstr(p, "```"));) {
        const char* lang = p + 3;
        const char* nl = strchr(lang, '\n');
        if (!nl) break;
        const char* body = nl + 1;
        const char* end = strstr(body, "```");
        if (!end) break;   // still being written
        int is_py = (nl - lang >= 2 && !strncmp(lang, "py", 2)) ||
                    (nl == lang && (strstr(body, "print(") || strstr(body, "import ") || strstr(body, "def ")));
        while (end > body && (end[-1] == '\n' || end[-1] == ' ')) end--;
        size_t len = end - body;
        if (is_py && (len > best_len || !best_py)) best = body, best_len = len, best_py = 1;
        p = strstr(body, "```") + 3;
    }
    if (!best || (int)best_len >= max) return 0;
    memcpy(out, best, best_len);
    out[best_len] = 0;
    snprintf(ttl, tmax, "programa.py");
    for (const char* p = text; (p = strstr(p, ".py")); p += 3) {   // a file name mentioned in the reply
        if (p[3] >= 'a' && p[3] <= 'z') continue;
        const char* s = p;
        while (s > text && (s[-1] == '_' || s[-1] == '-' || (s[-1] >= 'a' && s[-1] <= 'z') || (s[-1] >= 'A' && s[-1] <= 'Z') ||
                            (s[-1] >= '0' && s[-1] <= '9')))
            s--;
        if (p - s >= 2 && p - s < tmax - 4) {
            snprintf(ttl, tmax, "%.*s.py", (int)(p - s), s);
            break;
        }
    }
    return (int)best_len;
}
