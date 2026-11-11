// Runs the Python programs Garlic writes, with MicroPython (chat/py/mp). Text programs get a console (print and
// input through the app's keyboard); graphical ones (turtle, tkinter, pygame: compatibility modules written in
// Python, assets/py/) draw through the small _garlic module into a scaled window, played with the controller.
#pragma once
#include <stddef.h>
#include <stdint.h>

// Controller buttons as the host reports them.
enum {
    PB_UP = 1, PB_DOWN = 2, PB_LEFT = 4, PB_RIGHT = 8, PB_CROSS = 16, PB_CIRCLE = 32, PB_SQUARE = 64, PB_TRIANGLE = 128,
    PB_L1 = 256, PB_R1 = 512, PB_L2 = 1024, PB_R2 = 2048, PB_OPTIONS = 4096
};

typedef struct {
    uint32_t (*buttons)(float* lx, float* ly);   // PB_* held now; left stick -1..1
    void (*present)(void);                       // show `back` (waits for the next frame)
    double (*now)(void);                         // seconds
    int (*keyboard)(char* out, int max);         // the app's keyboard over the current picture; 0 if closed
    const char* modules;                         // folder with the Python compatibility modules
    size_t stack_limit;                          // usable C stack of the calling thread, bytes
    // optional: the heap straight from the system and back after each run (NULL: malloc / free, which may keep
    // the memory in the process; on the PS4 the dictation weights need it back)
    void* (*heap_alloc)(size_t size);
    void (*heap_free)(void* p, size_t size);
} PyHost;

enum { PY_OK, PY_ERROR, PY_STOPPED };

int py_run(const PyHost* host, const char* source, const char* title);   // until the program ends or ○
const char* py_error(void);              // traceback of the last run that failed
void py_request_stop(void);              // from another thread: stop the running program (as ○)
// The Python code in a reply: the longest ```python block (or the longest block that looks like Python).
// Returns its length (0 = none) and a title for it (a file name mentioned in the text, or "programa.py").
int py_find_code(const char* text, char* out, int max, char* title, int tmax);
