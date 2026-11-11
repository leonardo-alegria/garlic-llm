// Garlic LLM user interface (see ui.h). Software rendering into `back` (1920 x 1080, 0x80RRGGBB): a dark,
// blue theme; anti-aliased shapes from signed distances; bitmap fonts from pc/export_chat.py; a small markdown
// renderer (headings, lists, bold, italics, code, quotes, tables); the chat, the keyboard and the voice screens.
#include "ui.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32   // (this file also builds for Windows, which uses malloc instead)
#include <orbis/libkernel.h>
#endif

// ---- theme ---------------------------------------------------------------------------------------------------
#define C_BG_TOP 0x0B1C3A
#define C_BG_BOT 0x040A18
#define C_ACCENT 0x0070D1
#define C_ACCENT_HI 0x3FA9F5
#define C_TEXT 0xE8EFFA
#define C_TEXT_DIM 0xA3B5D2
#define C_TEXT_MUTED 0x6A80A3
#define C_SURFACE 0x0E1D3A
#define C_SURFACE_HI 0x15284B
#define C_LINE 0x1C3358
#define C_LINE_HI 0x2A4672
#define C_USER 0x0B5DC6
#define C_PANEL 0x0A1631
#define C_KEY 0x152A4E
#define C_KEY_SPECIAL 0x0F1F3E
#define C_GREEN 0x3DDC97
#define C_RED 0xFF6B76
#define C_CROSS 0x8AB4FF
#define C_CIRCLE 0xFF7A85
#define C_TRIANGLE 0x45D6A0
#define C_SQUARE 0xF59ADB

#define HEADER_H 100
#define FOOTER_H 84
#define COL_W 1440
#define COL_X ((SCREEN_W - COL_W) / 2)
#define AI_INDENT 84
#define USER_MAX_W 1000
#define BUB_PAD_X 28
#define BUB_PAD_Y 16
#define MSG_GAP 40

int ui_lang = LANG_ES;
const char* ui_error = "";

// ---- strings ---------------------------------------------------------------------------------------------------

static const char* strings[N_STRINGS][N_LANGS] = {
    [S_TAGLINE] = {"%s en la GPU de tu PS4.", "%s on your PS4's GPU.", "%s na GPU do seu PS4.", "%s на GPU вашей PS4."},   // the model's name
    [S_EMPTY_HINT] = {"100% local y sin internet.", "100% local, no internet.", "100% local e sem internet.",
                      "100% локально, без интернета."},
    [S_WRITE] = {"Escribir", "Type", "Escrever", "Написать"},
    [S_TALK_HOLD] = {"Mantén para hablar", "Hold to talk", "Segure para falar", "Голос (удерживать)"},
    [S_STOP] = {"Detener", "Stop", "Parar", "Остановить"},
    [S_NEW_CHAT] = {"Nueva conversación", "New chat", "Nova conversa", "Новый чат"},
    [S_SCROLL] = {"Desplazar", "Scroll", "Rolar", "Прокрутка"},
    [S_TEMPERATURE] = {"Temperatura", "Temperature", "Temperatura", "Температура"},
    [S_CONTEXT] = {"contexto", "context", "contexto", "контекст"},
    [S_READING] = {"Leyendo tu mensaje…", "Reading your message…", "Lendo sua mensagem…", "Читаю сообщение…"},
    [S_CONTEXT_FULL] = {"La conversación llenó la memoria del modelo; empiezo una nueva.",
                        "The conversation filled the model's memory; starting a new one.",
                        "A conversa encheu a memória do modelo; começando uma nova.",
                        "Разговор заполнил память модели; начинаю новый."},
    [S_PLACEHOLDER] = {"Escribe un mensaje…", "Type a message…", "Digite uma mensagem…", "Введите сообщение…"},
    [S_KB_TYPE] = {"Escribir", "Type", "Digitar", "Ввод"},
    [S_KB_DELETE] = {"Borrar", "Delete", "Apagar", "Удалить"},
    [S_KB_SPACE] = {"Espacio", "Space", "Espaço", "Пробел"},
    [S_KB_SHIFT] = {"Mayúsculas", "Shift", "Maiúsculas", "Регистр"},
    [S_KB_EMOJI] = {"Emoji", "Emoji", "Emoji", "Эмодзи"},
    [S_KB_LETTERS] = {"Letras", "Letters", "Letras", "Буквы"},
    [S_KB_DICTATE] = {"Dictar (mantener)", "Dictate (hold)", "Ditar (segurar)", "Диктовка (удерж.)"},
    [S_KB_SEND] = {"Enviar", "Send", "Enviar", "Отправить"},
    [S_KB_CLOSE] = {"Cerrar", "Close", "Fechar", "Закрыть"},
    [S_SPACE_KEY] = {"espacio", "space", "espaço", "пробел"},
    [S_SEND_KEY] = {"Enviar", "Send", "Enviar", "Отправить"},
    [S_LISTENING] = {"Escuchando…", "Listening…", "Ouvindo…", "Слушаю…"},
    [S_RELEASE_TO_END] = {"Suelta para terminar", "Release to finish", "Solte para terminar", "Отпустите, чтобы закончить"},
    [S_TRANSCRIBING] = {"Transcribiendo…", "Transcribing…", "Transcrevendo…", "Распознаю речь…"},
    [S_YOUR_MESSAGE] = {"Tu mensaje", "Your message", "Sua mensagem", "Ваше сообщение"},
    [S_SEND] = {"Enviar", "Send", "Enviar", "Отправить"},
    [S_EDIT] = {"Editar", "Edit", "Editar", "Изменить"},
    [S_RETRY] = {"Repetir", "Retry", "Repetir", "Заново"},
    [S_DISCARD] = {"Descartar", "Discard", "Descartar", "Отмена"},
    [S_NO_MIC] = {"No se detecta micrófono: conecta unos auriculares al mando.",
                  "No microphone found: plug a headset into the controller.",
                  "Nenhum microfone detectado: conecte um fone ao controle.",
                  "Микрофон не найден: подключите гарнитуру к геймпаду."},
    [S_NO_SPEECH] = {"No se entendió nada. Inténtalo otra vez.", "Didn't catch that. Try again.",
                     "Não entendi nada. Tente de novo.", "Ничего не распознано. Попробуйте ещё раз."},
    [S_ASR_OFF] = {"El reconocimiento de voz no está disponible.", "Speech recognition is not available.",
                   "O reconhecimento de voz não está disponível.", "Распознавание речи недоступно."},
    [S_HOLD_LONGER] = {"Mantén □ pulsado mientras hablas.", "Keep □ held while you speak.",
                       "Mantenha □ pressionado enquanto fala.", "Держите □, пока говорите."},
    [S_PREPARING] = {"Preparando", "Getting ready", "Preparando", "Подготовка"},
    [S_LOADING_MODEL] = {"Cargando el modelo", "Loading the model", "Carregando o modelo", "Загрузка модели"},
    [S_LOADING_VOICE] = {"Cargando el reconocimiento de voz", "Loading speech recognition",
                         "Carregando o reconhecimento de voz", "Загрузка распознавания речи"},
    [S_ERROR_TITLE] = {"No se pudo iniciar", "Couldn't start", "Não foi possível iniciar", "Не удалось запустить"},
    [S_ERROR_HINT] = {"Pulsa el botón PS para salir.", "Press the PS button to exit.", "Pressione o botão PS para sair.",
                      "Нажмите кнопку PS, чтобы выйти."},
    [S_OFFLINE] = {"100 % sin conexión", "100% offline", "100% offline", "100% офлайн"},
    [S_SYSTEM_PROMPT] = {
        "Eres Garlic LLM, un asistente útil que funciona sin internet en la GPU de una PlayStation 4. "
        "Responde en el idioma del usuario, de forma clara y sin rodeos.",
        "You are Garlic LLM, a helpful assistant running offline on a PlayStation 4's GPU. Answer in the "
        "user's language, clearly and directly.",
        "Você é o Garlic LLM, um assistente útil que funciona sem internet na GPU de um PlayStation 4. "
        "Responda no idioma do usuário, de forma clara e direta.",
        "Ты — Garlic LLM, полезный ассистент, который работает без интернета на GPU PlayStation 4. Отвечай на "
        "языке пользователя, ясно и по делу."},
    [S_LOCAL_GPU] = {"GPU local", "local GPU", "GPU local", "локальный GPU"},
    [S_PY_RUN] = {"Ejecutar Python", "Run Python", "Executar Python", "Запустить Python"},
    [S_PY_EXIT] = {"Salir", "Exit", "Sair", "Выйти"},
    [S_PY_CLICK] = {"Clic", "Click", "Clique", "Клик"},
    [S_PY_RCLICK] = {"Clic derecho", "Right click", "Clique direito", "Правый клик"},
    [S_PY_MOVE] = {"Mover", "Move", "Mover", "Двигать"},
    [S_PY_KEYS] = {"Flechas", "Arrows", "Setas", "Стрелки"},
    [S_PY_BACK] = {"Volver al chat", "Back to chat", "Voltar ao chat", "Назад в чат"},
    [S_PY_AGAIN] = {"Ejecutar de nuevo", "Run again", "Executar de novo", "Запустить снова"},
    [S_PY_FIX] = {"Pedir arreglo", "Ask for a fix", "Pedir correção", "Попросить исправить"},
    [S_PY_DONE] = {"Programa terminado", "Program finished", "Programa terminado", "Программа завершена"},
    [S_PY_ERROR] = {"El programa tuvo un error", "The program hit an error", "O programa teve um erro", "В программе ошибка"},
    [S_PY_STOPPED] = {"Programa detenido", "Program stopped", "Programa interrompido", "Программа остановлена"},
    [S_PY_STARTING] = {"Ejecutando", "Running", "Executando", "Запуск"},
    // "Ask for a fix": states that the program ran on the PS4 itself; without that context the model may answer
    // that Python cannot run on a PS4
    [S_PY_FIX_MSG] = {
        "Ejecuté tu programa aquí mismo, en la PS4, y falló con este error:",
        "I ran your program right here on the PS4, and it failed with this error:",
        "Executei o seu programa aqui mesmo, no PS4, e ele falhou com este erro:",
        "Я запустил твою программу прямо здесь, на PS4, и она завершилась с ошибкой:"},
    // "Ask for a fix", the request: lists what Garlic's Python has and what it lacks
    [S_PY_FIX_ASK] = {
        "Corrígelo para el Python 3 de Garlic (MicroPython): hay turtle, pygame y tkinter adaptados a la tele "
        "(las flechas, Espacio y Enter van al mando), input(), print(), random, math, time, collections, "
        "itertools, string y copy; no hay numpy, matplotlib, archivos, red ni sonido. Casi siempre el fallo "
        "está en el propio código: busca la causa y corrígela sin quitar nada de lo pedido. Escribe el "
        "programa completo en un solo bloque ```python.",
        "Fix it for Garlic's Python 3 (MicroPython): it has turtle, pygame and tkinter adapted to the TV "
        "(arrows, Space and Enter are on the controller), input(), print(), random, math, time, collections, "
        "itertools, string and copy; there is no numpy, matplotlib, files, network or sound. The bug is "
        "almost always in the code itself: find the cause and fix it without dropping anything that was asked "
        "for. Write the whole program in a single ```python block.",
        "Corrija-o para o Python 3 do Garlic (MicroPython): há turtle, pygame e tkinter adaptados à TV (as "
        "setas, Espaço e Enter ficam no controle), input(), print(), random, math, time, collections, "
        "itertools, string e copy; não há numpy, matplotlib, arquivos, rede nem som. Quase sempre a falha "
        "está no próprio código: encontre a causa e corrija sem tirar nada do que foi pedido. Escreva o "
        "programa completo em um único bloco ```python.",
        "Исправь её для Python 3 в Garlic (MicroPython): есть turtle, pygame и tkinter для телевизора "
        "(стрелки, пробел и Enter — на геймпаде), input(), print(), random, math, time, collections, "
        "itertools, string и copy; нет numpy, matplotlib, файлов, сети и звука. Почти всегда ошибка в самом "
        "коде: найди причину и исправь её, ничего не убирая из того, что просили. Напиши всю программу "
        "целиком в одном блоке ```python."},
    [S_PY_INPUT] = {"Responde al programa…", "Answer the program…", "Responda ao programa…", "Ответьте программе…"},
    [S_VOICE_MODE] = {"Modo voz", "Voice mode", "Modo voz", "Голосовой режим"},
    [S_THINKING] = {"Pensando…", "Thinking…", "Pensando…", "Думаю…"},
    [S_LISTENING_SHORT] = {"Te escucho…", "Listening…", "Estou ouvindo…", "Слушаю…"},
    [S_VOICE_LOADING] = {"Preparando la voz…", "Getting the voice ready…", "Preparando a voz…", "Готовлю голос…"},
    [S_VOICE_NO_TTS] = {"La voz no está disponible: Garlic responderá con texto", "The voice is not available: Garlic will answer in text",
                        "A voz não está disponível: o Garlic responderá em texto", "Голос недоступен: Garlic ответит текстом"},
    // "Think" (R3)
    [S_THINK_OFF_HINT] = {"Pensar: no", "Think: off", "Pensar: não", "Думать: нет"},
    [S_THINK_ON_HINT] = {"Pensar: sí", "Think: on", "Pensar: sim", "Думать: да"},
    [S_THINK_ON_TOAST] = {"Pensar activado: más acierto en preguntas difíciles, pero tarda ~1 min",
                          "Thinking on: better at hard questions, but takes ~1 min",
                          "Pensar ativado: acerta mais em perguntas difíceis, mas demora ~1 min",
                          "Размышление включено: точнее в трудных вопросах, но около минуты"},
    [S_THINK_OFF_TOAST] = {"Pensar desactivado: respuestas al momento", "Thinking off: instant answers",
                           "Pensar desativado: respostas na hora", "Размышление выключено: ответы сразу"},
    [S_THOUGHT] = {"Pensó %d s · %d tokens", "Thought for %d s · %d tokens", "Pensou %d s · %d tokens",
                   "Думал %d с · токенов: %d"},
    [S_THINK_CUT] = {"tope alcanzado", "limit reached", "limite atingido", "лимит"},
    [S_THINK_CHIP] = {"Pensar", "Think", "Pensar", "Думать"},
};

const char* tr(StrId id) { return strings[id][ui_lang] ? strings[id][ui_lang] : strings[id][LANG_EN]; }

int lang_from_system(int l) {
    if (l == 3 || l == 20) return LANG_ES;   // Spanish, Spanish (Latin America)
    if (l == 7 || l == 17) return LANG_PT;   // Portuguese (Portugal, Brazil)
    if (l == 8) return LANG_RU;
    return LANG_EN;
}

// ---- pixels and shapes -------------------------------------------------------------------------------------------

static int clip_x0, clip_y0, clip_x1 = SCREEN_W, clip_y1 = SCREEN_H;
static int text_alpha = 255;   // multiplies text and shapes drawn by the fading toast

static void set_clip(int x0, int y0, int x1, int y1) {
    clip_x0 = x0 < 0 ? 0 : x0, clip_y0 = y0 < 0 ? 0 : y0;
    clip_x1 = x1 > SCREEN_W ? SCREEN_W : x1, clip_y1 = y1 > SCREEN_H ? SCREEN_H : y1;
}
static void no_clip(void) { set_clip(0, 0, SCREEN_W, SCREEN_H); }

static inline void blend(uint32_t* d, uint32_t rgb, uint32_t a) {
    if (a >= 255) {
        *d = 0x80000000 | rgb;
        return;
    }
    uint32_t s = *d, ia = 255 - a;
    uint32_t r = ((s >> 16 & 255) * ia + (rgb >> 16 & 255) * a) * 257 >> 16;
    uint32_t g = ((s >> 8 & 255) * ia + (rgb >> 8 & 255) * a) * 257 >> 16;
    uint32_t b = ((s & 255) * ia + (rgb & 255) * a) * 257 >> 16;
    *d = 0x80000000 | r << 16 | g << 8 | b;
}

static void fill_a(int x, int y, int w, int h, uint32_t rgb, int alpha) {
    int x0 = x < clip_x0 ? clip_x0 : x, y0 = y < clip_y0 ? clip_y0 : y;
    int x1 = x + w > clip_x1 ? clip_x1 : x + w, y1 = y + h > clip_y1 ? clip_y1 : y + h;
    alpha = alpha * text_alpha / 255;
    for (int j = y0; j < y1; j++) {
        uint32_t* row = back + j * SCREEN_W;
        if (alpha >= 255)
            for (int i = x0; i < x1; i++) row[i] = 0x80000000 | rgb;
        else
            for (int i = x0; i < x1; i++) blend(row + i, rgb, alpha);
    }
}
static void fill(int x, int y, int w, int h, uint32_t rgb) { fill_a(x, y, w, h, rgb, 255); }

static inline float clamp01(float v) { return v < 0 ? 0 : v > 1 ? 1 : v; }

// Signed distance from (px, py) to a rounded box (centre, half size, corner radius r); negative inside.
static inline float rbox_sd(float px, float py, float cx, float cy, float hw, float hh, float r) {
    float qx = fabsf(px - cx) - (hw - r), qy = fabsf(py - cy) - (hh - r);
    float ox = qx > 0 ? qx : 0, oy = qy > 0 ? qy : 0;
    float in = qx > qy ? qx : qy;
    if (in > 0) in = 0;
    float out = ox == 0 ? oy : oy == 0 ? ox : sqrtf(ox * ox + oy * oy);
    return out + in - r;
}

// Rounded rectangle, filled (thick 0) or outlined; radii top-left, top-right, bottom-right, bottom-left.
static void rrect4(int x, int y, int w, int h, const int rad[4], int thick, uint32_t rgb, int alpha) {
    if (w <= 0 || h <= 0) return;
    alpha = alpha * text_alpha / 255;
    float cx = x + w * 0.5f, cy = y + h * 0.5f, hw = w * 0.5f, hh = h * 0.5f;
    int rmax = rad[0];
    for (int k = 1; k < 4; k++)
        if (rad[k] > rmax) rmax = rad[k];
    int band = (thick > rmax ? thick : rmax) + 1;
    int y0 = y < clip_y0 ? clip_y0 : y, y1 = y + h > clip_y1 ? clip_y1 : y + h;
    int xa = x < clip_x0 ? clip_x0 : x, xb = x + w > clip_x1 ? clip_x1 : x + w;
    for (int py = y0; py < y1; py++) {
        uint32_t* row = back + py * SCREEN_W;
        float fy = py + 0.5f;
        int top = fy < cy;
        int edge_row = py < y + band || py >= y + h - band;
        for (int px = xa; px < xb; px++) {
            if (!edge_row && px >= x + band && px < x + w - band) {   // the plain middle of the row
                if (thick) {
                    px = x + w - band - 1;
                    continue;
                }
                int stop = x + w - band < xb ? x + w - band : xb;
                for (; px < stop; px++) blend(row + px, rgb, alpha);
                px--;
                continue;
            }
            float fx = px + 0.5f;
            float r = (float)(fx < cx ? (top ? rad[0] : rad[3]) : (top ? rad[1] : rad[2]));
            if (r > hw) r = hw;
            if (r > hh) r = hh;
            float d = rbox_sd(fx, fy, cx, cy, hw, hh, r);
            float cov = clamp01(0.5f - d);
            if (thick) cov -= clamp01(0.5f - (d + thick));
            if (cov > 0.004f) blend(row + px, rgb, (uint32_t)(cov * alpha));
        }
    }
}
static void rrect(int x, int y, int w, int h, int r, uint32_t rgb, int alpha) {
    int rad[4] = {r, r, r, r};
    rrect4(x, y, w, h, rad, 0, rgb, alpha);
}
static void rrect_line(int x, int y, int w, int h, int r, int thick, uint32_t rgb, int alpha) {
    int rad[4] = {r, r, r, r};
    rrect4(x, y, w, h, rad, thick, rgb, alpha);
}

// Disc (thick 0) or ring of radius r around (cx, cy). Per row, only the pixels near an edge need their distance:
// further out they are not covered, and the inner span is either fully covered (a disc: cov 1) or the ring's hole
// (cov 0). The margins of 0.01 px keep those spans strictly where the per-pixel formula gives exactly 1 or 0, so the
// result is the same as computing every pixel (this keeps the voice screen's big halo discs cheap to draw).
static void circle(float cx, float cy, float r, float thick, uint32_t rgb, int alpha) {
    alpha = alpha * text_alpha / 255;
    int x0 = (int)floorf(cx - r - 1), x1 = (int)ceilf(cx + r + 1), y0 = (int)floorf(cy - r - 1), y1 = (int)ceilf(cy + r + 1);
    if (x0 < clip_x0) x0 = clip_x0;
    if (y0 < clip_y0) y0 = clip_y0;
    if (x1 > clip_x1) x1 = clip_x1;
    if (y1 > clip_y1) y1 = clip_y1;
    const float ro = r + 0.51f, ri = thick > 0 ? r - thick - 0.51f : r - 0.51f;
    for (int py = y0; py < y1; py++) {
        float dy = py + 0.5f - cy, ady = fabsf(dy);
        if (ady >= ro) continue;   // the whole row is outside (d > 0.5: cov 0)
        float ho = sqrtf(ro * ro - dy * dy);
        int o0 = (int)floorf(cx - ho - 0.5f), o1 = (int)ceilf(cx + ho - 0.5f) + 1;   // outside [o0, o1): cov 0
        if (o0 < x0) o0 = x0;
        if (o1 > x1) o1 = x1;
        int a0 = o1, a1 = o1;   // [a0, a1): the inner span
        if (ri > ady) {
            float hi = sqrtf(ri * ri - dy * dy);
            a0 = (int)ceilf(cx - hi - 0.5f), a1 = (int)floorf(cx + hi - 0.5f) + 1;
            if (a0 < o0) a0 = o0;
            if (a1 > o1) a1 = o1;
            if (a1 < a0) a1 = a0;
        }
        uint32_t* row = back + py * SCREEN_W;
        for (int px = o0; px < o1; px++) {
            if (px == a0) {
                if (thick <= 0)
                    for (; px < a1; px++) blend(row + px, rgb, (uint32_t)(1.0f * alpha));
                px = a1;
                if (px >= o1) break;
            }
            float dx = px + 0.5f - cx;
            float d = sqrtf(dx * dx + dy * dy) - r;
            float cov = clamp01(0.5f - d);
            if (thick > 0) cov -= clamp01(0.5f - (d + thick));
            if (cov > 0.004f) blend(row + px, rgb, (uint32_t)(cov * alpha));
        }
    }
}

// Line with round ends, width wd.
static void line(float x0, float y0, float x1, float y1, float wd, uint32_t rgb, int alpha) {
    alpha = alpha * text_alpha / 255;
    float r = wd * 0.5f;
    int bx0 = (int)floorf(fminf(x0, x1) - r - 1), bx1 = (int)ceilf(fmaxf(x0, x1) + r + 1);
    int by0 = (int)floorf(fminf(y0, y1) - r - 1), by1 = (int)ceilf(fmaxf(y0, y1) + r + 1);
    if (bx0 < clip_x0) bx0 = clip_x0;
    if (by0 < clip_y0) by0 = clip_y0;
    if (bx1 > clip_x1) bx1 = clip_x1;
    if (by1 > clip_y1) by1 = clip_y1;
    float vx = x1 - x0, vy = y1 - y0, len2 = vx * vx + vy * vy;
    for (int py = by0; py < by1; py++)
        for (int px = bx0; px < bx1; px++) {
            float wx = px + 0.5f - x0, wy = py + 0.5f - y0;
            float t = len2 > 0 ? (wx * vx + wy * vy) / len2 : 0;
            t = clamp01(t);
            float dx = wx - vx * t, dy = wy - vy * t;
            float cov = clamp01(r - sqrtf(dx * dx + dy * dy) + 0.5f);
            if (cov > 0.004f) blend(back + py * SCREEN_W + px, rgb, (uint32_t)(cov * alpha));
        }
}

// ---- background: a dark blue gradient with a soft glow, computed once --------------------------------------------

static uint32_t* bg;

// The four 8 MB screen copies (the frame being drawn, this background, the backdrop of modal screens and the
// keyboard's picture) come from a 32 MiB block of direct memory of their own, CPU cached (WB Onion) like malloc's, so
// they stay out of the system RAM (flexible memory), where the 9B's dictation weights leave ~20 MiB free.
// The engine then leaves 32 MiB less for the system (ui_screen_pool_mb). Without the block: malloc.
static uint8_t* scr_pool;
static int scr_tried, scr_used;
#define SCR_POOL (32u << 20)
#define SCR_ONE (((size_t)SCREEN_W * SCREEN_H * 4 + 0xFFFF) & ~(size_t)0xFFFF)

int ui_screen_pool_mb(void) {
#ifndef _WIN32
    if (!scr_tried) {
        scr_tried = 1;
        off_t off = 0;
        void* m = NULL;
        if (!sceKernelAllocateDirectMemory(0, sceKernelGetDirectMemorySize(), SCR_POOL, 0x200000, 0 /* WB Onion */, &off)) {
            if (!sceKernelMapDirectMemory(&m, SCR_POOL, 0x03 /* CPU read/write */, 0, off, 0x200000)) scr_pool = m;
            else sceKernelReleaseDirectMemory(off, SCR_POOL);
        }
    }
#endif
    return scr_pool ? (int)(SCR_POOL >> 20) : 0;
}

static int in_pool(const void* p) { return scr_pool && (const uint8_t*)p >= scr_pool && (const uint8_t*)p < scr_pool + SCR_POOL; }

uint32_t* ui_screen_buf(void) {
    ui_screen_pool_mb();
    if (scr_pool && (size_t)(scr_used + 1) * SCR_ONE <= SCR_POOL) return (uint32_t*)(scr_pool + SCR_ONE * scr_used++);
    return malloc(SCREEN_W * SCREEN_H * 4);
}

static void make_background(void) {
    bg = ui_screen_buf();
    uint32_t seed = 1;
    for (int y = 0; y < SCREEN_H; y++) {
        float t = (float)y / (SCREEN_H - 1);
        float base[3] = {(C_BG_TOP >> 16 & 255) + ((C_BG_BOT >> 16 & 255) - (C_BG_TOP >> 16 & 255)) * t,
                         (C_BG_TOP >> 8 & 255) + ((C_BG_BOT >> 8 & 255) - (C_BG_TOP >> 8 & 255)) * t,
                         (C_BG_TOP & 255) + ((C_BG_BOT & 255) - (C_BG_TOP & 255)) * t};
        for (int x = 0; x < SCREEN_W; x++) {
            float dx = (x - SCREEN_W * 0.5f) / (SCREEN_W * 0.75f), dy = (y + 200.0f) / (SCREEN_H * 1.1f);
            float g = 1.0f - (dx * dx + dy * dy);
            g = g > 0 ? g * g * 0.55f : 0;
            seed = seed * 1664525u + 1013904223u;
            float dither = ((seed >> 24) / 255.0f - 0.5f) * 1.2f;   // breaks up banding in the gradient
            int r = (int)(base[0] + g * 6 + dither), gr = (int)(base[1] + g * 34 + dither), b = (int)(base[2] + g * 80 + dither);
            r = r < 0 ? 0 : r > 255 ? 255 : r, gr = gr < 0 ? 0 : gr > 255 ? 255 : gr, b = b < 0 ? 0 : b > 255 ? 255 : b;
            if (y < HEADER_H || y >= SCREEN_H - FOOTER_H) {   // header and footer bars: darker
                r = r * 5 / 9 + 1, gr = gr * 5 / 9 + 3, b = b * 5 / 9 + 8;
                if (y == HEADER_H - 1 || y == SCREEN_H - FOOTER_H) r = C_LINE >> 16, gr = C_LINE >> 8 & 255, b = C_LINE & 255;
            }
            bg[y * SCREEN_W + x] = 0x80000000 | r << 16 | gr << 8 | b;
        }
    }
}

static void draw_background(void) { memcpy(back, bg, SCREEN_W * SCREEN_H * 4); }

// Modal screens (keyboard, voice) draw the scene behind them once, keep it here and restore it every frame.
// Little system memory may be left (the dictation weights are there), so neither copy is guaranteed:
// without one, the plain background stands in.
static uint32_t* backdrop;
static int kb_cache_sig = -1;
static uint32_t* kb_cache;
void ui_backdrop_save(void) {
    kb_cache_sig = -1;   // the keyboard's cached picture contains the old backdrop
    if (!backdrop) backdrop = ui_screen_buf();
    if (backdrop) memcpy(backdrop, back, SCREEN_W * SCREEN_H * 4);
}
void ui_backdrop_draw(void) {
    if (backdrop) memcpy(back, backdrop, SCREEN_W * SCREEN_H * 4);
    else draw_background();
}
// Gives back the two 8 MB screen copies (they come back when next needed, if there is room); those in the screen
// block are not system RAM and stay.
void ui_free_caches(void) {
    if (backdrop && !in_pool(backdrop)) free(backdrop), backdrop = NULL;
    if (kb_cache && !in_pool(kb_cache)) free(kb_cache), kb_cache = NULL;
    kb_cache_sig = -1;
}

// Darkens the area between the header and the footer by half (cheap: one shift per pixel).
void ui_dim(void) {
    for (uint32_t* p = back + HEADER_H * SCREEN_W; p < back + (SCREEN_H - FOOTER_H) * SCREEN_W; p++)
        *p = 0x80000000 | ((*p & 0xFEFEFE) >> 1) + 0x010307;
}

// Rows y0..y1 fade from the background (alpha a0) to the current picture (a1): soft edges for scrolled text.
static void fade_rows(int y0, int y1, int a0, int a1) {
    for (int y = y0; y < y1; y++) {
        int a = a0 + (a1 - a0) * (y - y0) / (y1 - y0);
        uint32_t* d = back + y * SCREEN_W;
        const uint32_t* s = bg + y * SCREEN_W;
        for (int x = 0; x < SCREEN_W; x++) blend(d + x, s[x] & 0xFFFFFF, 255 - a);
    }
}

// ---- fonts (PFN2 from pc/export_chat.py) ----------------------------------------------------------------------
// Header "<4sHHII" (magic, line height, ascent, glyph count, bitmap bytes); per glyph "<IHHhhHBBI" (codepoint,
// width, height, left, top, advance, kind 0 = alpha / 1 = RGBA, pad, bitmap offset), sorted by codepoint; then
// the bitmaps; then optionally "SEQS" + count + per sequence "<BBHI" (length, 0, 0, glyph index) and its
// codepoints: emoji made of several characters (flags, keycaps, joined emoji), drawn as one image.
// Emoji images are Twemoji (CC-BY 4.0, credited in assets/CREDITOS.txt).
#define FONT_HEADER 16
#define FONT_REC 20

typedef struct Font {
    uint16_t line, ascent;
    uint32_t n;
    const uint8_t* recs;
    const uint8_t* bitmaps;
    uint32_t* cps;
    int fallback;
    const struct Font* next;   // font to ask when this one lacks a character
    uint32_t n_seq;
    uint32_t* seq_first;
    uint8_t* seq_len;
    uint32_t* seq_glyph;
    const uint8_t** seq_cps;
    int space;                 // advance of ' '
} Font;

typedef struct {
    uint16_t w, h, adv;
    int16_t bx, by;
    uint8_t kind;
    const uint8_t* bm;
} Glyph;

static Font font, font_bold, font_italic, font_small, font_big, mono;

static uint32_t utf8_next(const char** s) {
    const uint8_t* p = (const uint8_t*)*s;
    uint32_t c = *p++;
    int extra = c < 0x80 ? 0 : (c >> 5) == 6 ? 1 : (c >> 4) == 14 ? 2 : (c >> 3) == 30 ? 3 : 0;
    if (extra) c &= 0x3F >> extra;
    for (int k = 0; k < extra && (*p & 0xC0) == 0x80; k++) c = c << 6 | (*p++ & 0x3F);
    *s = (const char*)p;
    return c;
}

static int utf8_put(uint32_t c, char* o) {
    if (c < 0x80) return o[0] = (char)c, 1;
    if (c < 0x800) return o[0] = (char)(0xC0 | c >> 6), o[1] = (char)(0x80 | (c & 63)), 2;
    if (c < 0x10000)
        return o[0] = (char)(0xE0 | c >> 12), o[1] = (char)(0x80 | (c >> 6 & 63)), o[2] = (char)(0x80 | (c & 63)), 3;
    o[0] = (char)(0xF0 | c >> 18), o[1] = (char)(0x80 | (c >> 12 & 63)), o[2] = (char)(0x80 | (c >> 6 & 63));
    o[3] = (char)(0x80 | (c & 63));
    return 4;
}

static void glyph_at(const Font* f, int i, Glyph* g) {
    const uint8_t* r = f->recs + i * FONT_REC;
    uint32_t off;
    memcpy(&g->w, r + 4, 2);
    memcpy(&g->h, r + 6, 2);
    memcpy(&g->bx, r + 8, 2);
    memcpy(&g->by, r + 10, 2);
    memcpy(&g->adv, r + 12, 2);
    g->kind = r[14];
    memcpy(&off, r + 16, 4);
    g->bm = f->bitmaps + off;
}

static int find_cp(const Font* f, uint32_t cp) {
    int lo = 0, hi = (int)f->n - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (f->cps[mid] == cp) return mid;
        if (f->cps[mid] < cp) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

static int font_load(Font* f, const char* path) {
    size_t size;
    uint8_t* d = read_all(path, &size);
    if (!d || size < FONT_HEADER || memcmp(d, "PFN2", 4)) return -1;
    memset(f, 0, sizeof(*f));
    uint32_t bitmap_bytes;
    memcpy(&f->line, d + 4, 2);
    memcpy(&f->ascent, d + 6, 2);
    memcpy(&f->n, d + 8, 4);
    memcpy(&bitmap_bytes, d + 12, 4);
    f->recs = d + FONT_HEADER;
    f->bitmaps = d + FONT_HEADER + f->n * FONT_REC;
    f->cps = malloc(f->n * 4);
    for (uint32_t i = 0; i < f->n; i++) {
        memcpy(&f->cps[i], f->recs + i * FONT_REC, 4);
        if (f->cps[i] == 0xFFFD) f->fallback = (int)i;
    }
    const uint8_t* p = f->bitmaps + bitmap_bytes;
    if ((size_t)(p - d) + 8 <= size && !memcmp(p, "SEQS", 4)) {
        memcpy(&f->n_seq, p + 4, 4);
        p += 8;
        f->seq_first = malloc(f->n_seq * 4);
        f->seq_len = malloc(f->n_seq);
        f->seq_glyph = malloc(f->n_seq * 4);
        f->seq_cps = malloc(f->n_seq * sizeof(*f->seq_cps));
        for (uint32_t k = 0; k < f->n_seq; k++) {
            f->seq_len[k] = p[0];
            memcpy(&f->seq_glyph[k], p + 4, 4);
            memcpy(&f->seq_first[k], p + 8, 4);
            f->seq_cps[k] = p + 12;
            p += 8 + 4 * f->seq_len[k];
        }
    }
    int sp = find_cp(f, ' ');
    Glyph g;
    if (sp >= 0) glyph_at(f, sp, &g), f->space = g.adv;
    else f->space = f->line / 4;
    return 0;
}

// Joiners, variation selectors, keycap marks and skin-tone modifiers outside a known sequence draw as nothing.
static int invisible(uint32_t cp) {
    return (cp >= 0x200B && cp <= 0x200F) || (cp >= 0xFE00 && cp <= 0xFE0F) || (cp >= 0x1F3FB && cp <= 0x1F3FF) ||
           (cp >= 0xE0000 && cp <= 0xE007F) || cp == 0x20E3;
}

static void missing(uint32_t cp) {   // each character no font can draw is logged once
    static uint32_t seen[64];
    static int n;
    for (int i = 0; i < n; i++)
        if (seen[i] == cp) return;
    if (n < 64) seen[n++] = cp;
    say("no glyph for U+%04X", (unsigned)cp);
}

static int match_sequence(const Font* f, uint32_t cp, const char* s, const char* end, const char** after) {
    if (!f->n_seq) return -1;
    int lo = 0, hi = (int)f->n_seq;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (f->seq_first[mid] < cp) lo = mid + 1;
        else hi = mid;
    }
    int best = -1, best_len = 0;
    for (int k = lo; k < (int)f->n_seq && f->seq_first[k] == cp; k++) {
        const char* q = s;
        int ok = 1;
        for (int j = 1; j < f->seq_len[k] && ok; j++) {
            uint32_t want;
            memcpy(&want, f->seq_cps[k] + 4 * (j - 1), 4);
            uint32_t got = 0;
            do {
                if (q >= end || !*q) {
                    ok = 0;
                    break;
                }
                got = utf8_next(&q);
            } while (got == 0xFE0F);
            if (ok && got != want) ok = 0;
        }
        if (ok && f->seq_len[k] > best_len) {
            best = k, best_len = f->seq_len[k];
            *after = q;
        }
    }
    return best;
}

static int next_glyph(const Font* f, const char** s, const char* end, Glyph* g) {
    while (*s < end && **s) {
        uint32_t cp = utf8_next(s);
        if (invisible(cp)) continue;
        if (cp == '\t') cp = ' ';
        for (const Font* t = f; t; t = t->next) {
            const char* after;
            int k = match_sequence(t, cp, *s, end, &after);
            if (k >= 0) {
                *s = after;
                glyph_at(t, (int)t->seq_glyph[k], g);
                if (t != f) g->by += (int)f->ascent - (int)t->ascent;   // keep the baseline of the asking font
                return 1;
            }
            int i = find_cp(t, cp);
            if (i >= 0) {
                glyph_at(t, i, g);
                if (t != f) g->by += (int)f->ascent - (int)t->ascent;
                return 1;
            }
        }
        missing(cp);
        glyph_at(f, f->fallback, g);
        return 1;
    }
    return 0;
}

static int text_width(const Font* f, const char* s, const char* end) {
    int w = 0;
    Glyph g;
    while (next_glyph(f, &s, end, &g)) w += g.adv;
    return w;
}
static int str_width(const Font* f, const char* s) { return text_width(f, s, s + strlen(s)); }

// Draws s[0..end) with its top-left at (x, y); returns the x after the text. Emoji keep their own colours.
static int draw_text(const Font* f, int x, int y, const char* s, const char* end, uint32_t rgb) {
    Glyph g;
    while (next_glyph(f, &s, end, &g)) {
        int gx = x + g.bx, gy = y + g.by;
        if (gx + g.w > clip_x0 && gx < clip_x1 && gy + g.h > clip_y0 && gy < clip_y1)
            for (int j = 0; j < g.h; j++) {
                int py = gy + j;
                if (py < clip_y0 || py >= clip_y1) continue;
                uint32_t* row = back + py * SCREEN_W;
                for (int i = 0; i < g.w; i++) {
                    int px = gx + i;
                    if (px < clip_x0 || px >= clip_x1) continue;
                    if (g.kind == 1) {
                        const uint8_t* c = g.bm + (j * g.w + i) * 4;
                        if (c[3]) blend(row + px, (uint32_t)c[0] << 16 | c[1] << 8 | c[2], c[3] * text_alpha / 255);
                    } else {
                        uint32_t a = g.bm[j * g.w + i];
                        if (a) blend(row + px, rgb, a * text_alpha / 255);
                    }
                }
            }
        x += g.adv;
    }
    return x;
}
static int draw_str(const Font* f, int x, int y, const char* s, uint32_t rgb) { return draw_text(f, x, y, s, s + strlen(s), rgb); }
static void draw_center(const Font* f, int cx, int y, const char* s, uint32_t rgb) { draw_str(f, cx - str_width(f, s) / 2, y, s, rgb); }

// ---- images (PIMG from pc/make_brand.py) ----------------------------------------------------------------------

typedef struct {
    int w, h;
    const uint8_t* px;
} Image;
static Image logo_s, logo_l;

static int images_load(const char* path) {
    size_t size;
    uint8_t* d = read_all(path, &size);
    if (!d || size < 8 || memcmp(d, "PIMG", 4)) return -1;
    uint32_t n;
    memcpy(&n, d + 4, 4);
    Image* out[2] = {&logo_s, &logo_l};
    for (uint32_t i = 0; i < n && i < 2; i++) {
        uint16_t w, h;
        uint32_t off;
        memcpy(&w, d + 8 + i * 8, 2);
        memcpy(&h, d + 10 + i * 8, 2);
        memcpy(&off, d + 12 + i * 8, 4);
        out[i]->w = w, out[i]->h = h, out[i]->px = d + off;
    }
    return 0;
}

static void draw_image(const Image* im, int x, int y, int alpha) {
    if (!im->px) return;
    alpha = alpha * text_alpha / 255;
    for (int j = 0; j < im->h; j++) {
        int py = y + j;
        if (py < clip_y0 || py >= clip_y1) continue;
        for (int i = 0; i < im->w; i++) {
            int px = x + i;
            if (px < clip_x0 || px >= clip_x1) continue;
            const uint8_t* c = im->px + (j * im->w + i) * 4;
            if (c[3]) blend(back + py * SCREEN_W + px, (uint32_t)c[0] << 16 | c[1] << 8 | c[2], c[3] * alpha / 255);
        }
    }
}

int ui_init(const char* dir) {
    static const struct {
        Font* f;
        const char* name;
    } files[] = {{&font, "font.bin"},           {&font_bold, "font_bold.bin"}, {&font_italic, "font_italic.bin"},
                 {&font_small, "font_small.bin"}, {&font_big, "font_big.bin"},  {&mono, "mono.bin"}};
    char path[256];
    for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); i++) {
        snprintf(path, sizeof(path), "%s/%s", dir, files[i].name);
        if (font_load(files[i].f, path)) {
            ui_error = files[i].name;
            return -1;
        }
    }
    font_bold.next = font_italic.next = font_small.next = font_big.next = mono.next = &font;
    snprintf(path, sizeof(path), "%s/logo.bin", dir);
    if (images_load(path)) say("logo.bin missing");
    make_background();
    return 0;
}

// ---- messages --------------------------------------------------------------------------------------------------

Msg msgs[MAX_MSGS];
int n_msgs;
static void layout_free(Layout* L);

void msg_append(Msg* m, const char* s, size_t n) {
    if (m->len + n + 1 > m->cap) {
        m->cap = (m->len + n + 1) * 2;
        m->text = realloc(m->text, m->cap);
    }
    memcpy(m->text + m->len, s, n);
    m->len += n;
    m->text[m->len] = 0;
}

Msg* msg_new(int who) {
    if (n_msgs == MAX_MSGS) {
        free(msgs[0].text);
        layout_free(msgs[0].lay);
        memmove(msgs, msgs + 1, sizeof(Msg) * (MAX_MSGS - 1));
        n_msgs--;
    }
    Msg* m = &msgs[n_msgs++];
    memset(m, 0, sizeof(*m));
    m->who = who;
    msg_append(m, "", 0);
    return m;
}

void msgs_clear(void) {
    for (int i = 0; i < n_msgs; i++) free(msgs[i].text), layout_free(msgs[i].lay);
    n_msgs = 0;
}

// ---- markdown tables -------------------------------------------------------------------------------------------
// Complete table lines ("| a | b |") become grid rows:
//   '\x01' + kind (T top, R row, M separator, B bottom) + column widths in pixels ("120,80,") + ';' + cells
// separated by '\x02'.

#define MAX_COLS 12
#define CELL_PAD 18

static int split_cells(const char* a, const char* b, const char** cs, const char** ce) {
    while (a < b && *a == ' ') a++;
    if (a < b && *a == '|') a++;
    while (b > a && (b[-1] == ' ' || b[-1] == '\r')) b--;
    if (b > a && b[-1] == '|') b--;
    int n = 0;
    while (a <= b && n < MAX_COLS) {
        const char* p = a;
        while (p < b && *p != '|') p++;
        const char* s = a, * e = p;
        while (s < e && *s == ' ') s++;
        while (e > s && e[-1] == ' ') e--;
        cs[n] = s, ce[n] = e, n++;
        if (p >= b) break;
        a = p + 1;
    }
    return n;
}

static int is_separator(const char** cs, const char** ce, int n) {
    for (int i = 0; i < n; i++) {
        if (cs[i] == ce[i]) return 0;
        for (const char* p = cs[i]; p < ce[i]; p++)
            if (*p != '-' && *p != ':' && *p != ' ') return 0;
    }
    return n > 0;
}

static int clean_cell(const char* s, const char* e, char* out, int max) {   // drops ** and ` markers
    int n = 0;
    while (s < e && n < max - 1) {
        if (s + 1 < e && s[0] == '*' && s[1] == '*') {
            s += 2;
            continue;
        }
        if (*s == '`') {
            s++;
            continue;
        }
        out[n++] = *s++;
    }
    out[n] = 0;
    return n;
}

static void put_row(char** o, char kind, const int* widths, int ncols) {
    *(*o)++ = '\x01';
    *(*o)++ = kind;
    for (int i = 0; i < ncols; i++) *o += sprintf(*o, "%d,", widths[i]);
    *(*o)++ = ';';
}

static char* format_tables(const char* text) {
    size_t len = strlen(text);
    char* out = malloc(len * 2 + 4096);
    char* o = out;
    const char* s = text;
    char cell[512];
    while (*s) {
        const char* nl = strchr(s, '\n');
        const char* t = s;
        while (*t == ' ') t++;
        if (!nl || *t != '|') {
            const char* e = nl ? nl + 1 : s + strlen(s);
            memcpy(o, s, e - s);
            o += e - s;
            s = e;
            continue;
        }
        int rows = 0, widths[MAX_COLS] = {0}, ncols = 0;
        const char* cs[MAX_COLS], * ce[MAX_COLS];
        const char* p = s;
        while (*p) {
            const char* q = p;
            while (*q == ' ') q++;
            const char* e = strchr(p, '\n');
            if (*q != '|' || !e) break;
            int n = split_cells(p, e, cs, ce);
            if (!is_separator(cs, ce, n))
                for (int i = 0; i < n; i++) {
                    int cl = clean_cell(cs[i], ce[i], cell, sizeof(cell));
                    int w = text_width(&font_bold, cell, cell + cl);
                    if (w > widths[i]) widths[i] = w;
                }
            if (n > ncols) ncols = n;
            rows++;
            p = e + 1;
        }
        if (rows < 2) {
            memcpy(o, s, nl + 1 - s);
            o += nl + 1 - s;
            s = nl + 1;
            continue;
        }
        put_row(&o, 'T', widths, ncols);
        *o++ = '\n';
        for (const char* r = s; r < p;) {
            const char* e = strchr(r, '\n');
            int n = split_cells(r, e, cs, ce);
            if (is_separator(cs, ce, n)) {
                put_row(&o, 'M', widths, ncols);
            } else {
                put_row(&o, 'R', widths, ncols);
                for (int i = 0; i < n && i < ncols; i++) {
                    if (i) *o++ = '\x02';
                    o += clean_cell(cs[i], ce[i], o, (int)(ce[i] - cs[i]) + 1);
                }
            }
            *o++ = '\n';
            r = e + 1;
        }
        put_row(&o, 'B', widths, ncols);
        *o++ = '\n';
        s = p;
    }
    *o = 0;
    return out;
}

static void draw_table_line(const char* a, const char* b, int x, int top, int h, int header) {
    char kind = a[1];
    int widths[MAX_COLS], n = 0;
    const char* p = a + 2;
    while (p < b && *p != ';' && n < MAX_COLS) {
        widths[n++] = atoi(p);
        while (p < b && *p != ',') p++;
        p++;
    }
    p++;
    int xs[MAX_COLS + 1];
    xs[0] = x;
    for (int i = 0; i < n; i++) xs[i + 1] = xs[i] + widths[i] + 2 * CELL_PAD;
    if (kind != 'R') {
        fill(xs[0], top, xs[n] - xs[0] + 2, 2, kind == 'M' ? C_ACCENT : C_LINE_HI);
        return;
    }
    if (header) fill(xs[0], top, xs[n] - xs[0], h, 0x12274C);
    for (int i = 0; i <= n; i++) fill(xs[i], top, 2, h, C_LINE_HI);
    const Font* f = header ? &font_bold : &font;
    for (int i = 0; i < n && p <= b; i++) {
        const char* e = p;
        while (e < b && *e != '\x02') e++;
        draw_text(f, xs[i] + CELL_PAD, top + (h - f->line) / 2, p, e, header ? 0xFFFFFF : C_TEXT);
        p = e + 1;
    }
}

// ---- text layout: markdown blocks and inline styles, word wrap ----------------------------------------------

enum { ST_BOLD = 1, ST_ITALIC = 2, ST_CODE = 4, ST_STRIKE = 8, ST_LINK = 16 };
enum { DECO_NONE, DECO_CODE_TOP, DECO_CODE, DECO_CODE_BOT, DECO_TABLE, DECO_HR };

typedef struct {
    uint32_t text, strong, italic, code, code_bg, link, head, quote, marker;
} Palette;
static const Palette PAL_AI = {C_TEXT, 0xFFFFFF, 0xD2E2F8, 0x9CD6FF, 0x10264D, 0x63BBFF, 0xFFFFFF, 0xA8BAD6, C_ACCENT_HI};
static const Palette PAL_USER = {0xFFFFFF, 0xFFFFFF, 0xE6F0FF, 0xFFFFFF, 0x0A4AA6, 0xDDEEFF, 0xFFFFFF, 0xDCE9FF, 0xCFE6FF};
static const Palette PAL_NOTE = {C_TEXT_MUTED, C_TEXT_DIM, C_TEXT_MUTED, C_TEXT_DIM, C_SURFACE, C_TEXT_DIM, C_TEXT_DIM,
                                 C_TEXT_MUTED, C_TEXT_MUTED};
static const Palette PAL_FIELD = {C_TEXT, C_TEXT, C_TEXT, C_TEXT, 0, C_TEXT, C_TEXT, C_TEXT, C_TEXT};

typedef struct {
    const Font* f;
    uint32_t color;
    int x, w;
    const char* s, * e;
    uint8_t style;
} Run;

typedef struct {
    int y, h, base;            // top, height, baseline offset
    uint8_t deco, quote, header;
    int run0, nrun;
    const char* a, * b;        // table row text
} Line;

struct Layout {
    char* src;                 // copy of the source, to see whether it changed
    size_t src_len;
    char* text;                // source with the tables converted; runs point into it
    int width, plain;
    const Palette* pal;
    Line* lines;
    int n_lines, cap_lines;
    Run* runs;
    int n_runs, cap_runs;
    int height, max_w;
    int line_start;            // first run of the line being built
};

#define TEXT_LINE_H (font.line + 7)
#define QUOTE_INDENT 30
#define LIST_INDENT 38

static void layout_free(Layout* L) {
    if (!L) return;
    free(L->src), free(L->text), free(L->lines), free(L->runs), free(L);
}

static void add_run(Layout* L, const Font* f, uint32_t color, int x, int w, const char* s, const char* e, int style) {
    if (L->n_runs > L->line_start) {   // merge with the previous run when it is the same style and text follows on
        Run* r = &L->runs[L->n_runs - 1];
        if (r->f == f && r->color == color && r->style == style && !(style & ST_CODE) &&
            (r->e == s || (r->e + 1 == s && *r->e == ' ' && x == r->x + r->w + f->space))) {
            r->e = e;
            r->w = x + w - r->x;
            return;
        }
    }
    if (L->n_runs == L->cap_runs) {
        L->cap_runs = L->cap_runs ? L->cap_runs * 2 : 64;
        L->runs = realloc(L->runs, L->cap_runs * sizeof(Run));
    }
    Run* r = &L->runs[L->n_runs++];
    r->f = f, r->color = color, r->x = x, r->w = w, r->s = s, r->e = e, r->style = (uint8_t)style;
}

static Line* end_line(Layout* L, int h, int base, int deco, int quote) {
    if (L->n_lines == L->cap_lines) {
        L->cap_lines = L->cap_lines ? L->cap_lines * 2 : 32;
        L->lines = realloc(L->lines, L->cap_lines * sizeof(Line));
    }
    Line* ln = &L->lines[L->n_lines++];
    memset(ln, 0, sizeof(*ln));
    ln->y = L->height, ln->h = h, ln->base = base, ln->deco = (uint8_t)deco, ln->quote = (uint8_t)quote;
    ln->run0 = L->line_start, ln->nrun = L->n_runs - L->line_start;
    for (int i = ln->run0; i < L->n_runs; i++)
        if (L->runs[i].x + L->runs[i].w > L->max_w) L->max_w = L->runs[i].x + L->runs[i].w;
    L->height += h;
    L->line_start = L->n_runs;
    return ln;
}

typedef struct {
    const char* s, * e;
    uint8_t style, space;
} Piece;

#define MAX_PIECES 2048
static Piece pieces[MAX_PIECES];
static int n_pieces;

static void piece(const char* s, const char* e, int style, int* space) {
    if (e <= s || n_pieces == MAX_PIECES) return;
    pieces[n_pieces].s = s, pieces[n_pieces].e = e, pieces[n_pieces].style = (uint8_t)style;
    pieces[n_pieces].space = (uint8_t)*space;
    n_pieces++;
    *space = 0;
}

static int is_alnum(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (uint8_t)c >= 0x80; }

// Is there a closing single marker c in p..b (preceded by a non-space, not part of a double marker)?
static int has_closer(const char* p, const char* b, char c) {
    for (const char* q = p + 1; q < b; q++)
        if (*q == c && q[-1] != ' ' && q[-1] != c && (q + 1 >= b || q[1] != c)) {
            if (c == '_' && q + 1 < b && is_alnum(q[1])) continue;
            return 1;
        }
    return 0;
}

// Splits a[..b) into pieces at spaces and style markers (markers are dropped).
static void parse_inline(const char* a, const char* b, int plain) {
    n_pieces = 0;
    int style = 0, space = 0;
    const char* cur = a;
    const char* p = a;
    while (p < b) {
        char c = *p;
        if (c == ' ') {
            piece(cur, p, style, &space);
            while (p < b && *p == ' ') p++;
            cur = p;
            space = 1;
            continue;
        }
        if (plain) {
            p++;
            continue;
        }
        if (style & ST_CODE) {
            if (c == '`') {
                piece(cur, p, style, &space);
                style &= ~ST_CODE;
                cur = ++p;
            } else p++;
            continue;
        }
        if (c == '\\' && p + 1 < b && (p[1] == '*' || p[1] == '_' || p[1] == '`' || p[1] == '#' || p[1] == '~' || p[1] == '|')) {
            piece(cur, p, style, &space);
            cur = p + 1;
            p += 2;
            continue;
        }
        if (c == '`' && memchr(p + 1, '`', b - p - 1)) {
            piece(cur, p, style, &space);
            style |= ST_CODE;
            cur = ++p;
            continue;
        }
        if (c == '~' && p + 1 < b && p[1] == '~') {
            piece(cur, p, style, &space);
            style ^= ST_STRIKE;
            p += 2;
            cur = p;
            continue;
        }
        if (c == '*' || c == '_') {
            int n = 1;
            while (p + n < b && p[n] == c && n < 3) n++;
            char left = p > a ? p[-1] : ' ', right = p + n < b ? p[n] : ' ';
            int can_open = right != ' ', can_close = left != ' ';
            if (c == '_') can_open = can_open && !is_alnum(left), can_close = can_close && !(right != ' ' && is_alnum(right));
            int toggled = 0, ns = style;
            if (n >= 2) {
                if ((style & ST_BOLD) && can_close) ns &= ~ST_BOLD, toggled = 1;
                else if (!(style & ST_BOLD) && can_open) ns |= ST_BOLD, toggled = 1;
            }
            if (n == 1 || n == 3) {
                if ((style & ST_ITALIC) && can_close) ns &= ~ST_ITALIC, toggled = 1;
                else if (!(style & ST_ITALIC) && can_open && has_closer(p + n - 1, b, c)) ns |= ST_ITALIC, toggled = 1;
                else if (n == 3 && toggled) n = 2;   // *** that only opens bold: keep one * as text
            }
            if (!toggled && p + n == b && !can_close) {   // a marker at the end of the text still being written
                piece(cur, p, style, &space);
                p += n;
                cur = p;
                continue;
            }
            if (toggled) {
                piece(cur, p, style, &space);
                style = ns;
                p += n;
                cur = p;
                continue;
            }
            p += n;
            continue;
        }
        if (c == '[') {   // [text](link): show the text as a link
            const char* q = memchr(p, ']', b - p);
            if (q && q + 1 < b && q[1] == '(') {
                const char* r = memchr(q, ')', b - q);
                if (r) {
                    piece(cur, p, style, &space);
                    const char* t = p + 1;
                    while (t < q) {
                        const char* w = t;
                        while (w < q && *w != ' ') w++;
                        piece(t, w, style | ST_LINK, &space);
                        while (w < q && *w == ' ') w++, space = 1;
                        t = w;
                    }
                    p = r + 1;
                    cur = p;
                    continue;
                }
            }
        }
        p++;
    }
    piece(cur, b, style, &space);
}

static const Font* style_font(int st, int heading) {
    if (st & ST_CODE) return &mono;
    if (heading) return heading <= 2 ? &font_big : &font_bold;
    if (st & ST_BOLD) return &font_bold;
    if (st & ST_ITALIC) return &font_italic;
    return &font;
}

static uint32_t style_color(const Palette* pal, int st, int heading, int quote) {
    if (st & ST_CODE) return pal->code;
    if (st & ST_LINK) return pal->link;
    if (heading) return pal->head;
    if (st & ST_BOLD) return pal->strong;
    if (st & ST_ITALIC) return pal->italic;
    return quote ? pal->quote : pal->text;
}

// Places the pieces in lines from x_first (first line) / x_cont (the others) to `right`. An optional marker
// (list bullet or number) starts the first line.
static void flow(Layout* L, int x_first, int x_cont, int right, int heading, int quote, const Run* marker) {
    const Font* base_font = heading ? style_font(0, heading) : &font;
    int lh = heading ? base_font->line + (heading <= 2 ? 14 : 10) : TEXT_LINE_H;
    int base = (lh - base_font->line) / 2 + base_font->ascent;
    int x = x_first, has = 0;
    if (marker) add_run(L, marker->f, marker->color, marker->x, marker->w, marker->s, marker->e, marker->style);
    static int pw[MAX_PIECES];
    for (int i = 0; i < n_pieces; i++) pw[i] = text_width(style_font(pieces[i].style, heading), pieces[i].s, pieces[i].e);
    for (int i = 0; i < n_pieces;) {
        int j = i + 1;
        while (j < n_pieces && !pieces[j].space) j++;
        int ww = 0;
        for (int k = i; k < j; k++) ww += pw[k];
        int sp = has && pieces[i].space ? base_font->space : 0;
        if (has && x + sp + ww > right) {
            end_line(L, lh, base, DECO_NONE, quote);
            x = x_cont, has = 0, sp = 0;
        }
        x += sp;
        for (int k = i; k < j; k++) {
            const Font* f = style_font(pieces[k].style, heading);
            uint32_t col = style_color(L->pal, pieces[k].style, heading, quote);
            if (ww <= right - x_cont || x + pw[k] <= right) {
                add_run(L, f, col, x, pw[k], pieces[k].s, pieces[k].e, pieces[k].style);
                x += pw[k];
                has = 1;
                continue;
            }
            for (const char* c = pieces[k].s; c < pieces[k].e;) {   // a word longer than the line: cut it
                const char* nx = c;
                utf8_next(&nx);
                int cw = text_width(f, c, nx);
                if (has && x + cw > right) {
                    end_line(L, lh, base, DECO_NONE, quote);
                    x = x_cont, has = 0;
                }
                add_run(L, f, col, x, cw, c, nx, pieces[k].style);
                x += cw, has = 1, c = nx;
            }
        }
        i = j;
    }
    if (has || marker || !n_pieces) end_line(L, lh, base, DECO_NONE, quote);
}

static int is_fence(const char* t, const char* e) { return e - t >= 3 && !strncmp(t, "```", 3); }

static int is_rule(const char* t, const char* e) {
    while (e > t && (e[-1] == ' ' || e[-1] == '\r')) e--;
    if (e - t < 3 || (*t != '-' && *t != '*' && *t != '_')) return 0;
    int n = 0;
    for (const char* p = t; p < e; p++) {
        if (*p == *t) n++;
        else if (*p != ' ') return 0;
    }
    return n >= 3;
}

static void layout_build(Layout* L, const char* src, size_t len, int width, const Palette* pal, int plain) {
    free(L->src), free(L->text);
    L->src = malloc(len + 1);
    memcpy(L->src, src, len + 1);
    L->src_len = len, L->width = width, L->pal = pal, L->plain = plain;
    L->text = plain ? strdup(src) : format_tables(src);
    L->n_lines = L->n_runs = L->height = L->max_w = L->line_start = 0;
    int in_code = 0, blank = 0, header = 0;
    const int code_pad = 24;
    for (const char* s = L->text; *s;) {
        const char* nl = strchr(s, '\n');
        const char* e = nl ? nl : s + strlen(s);
        const char* t = s;
        int spaces = 0;
        while (t < e && (*t == ' ' || *t == '\t')) spaces += *t == '\t' ? 4 : 1, t++;
        if (plain) {
            parse_inline(s, e, 1);
            flow(L, 0, 0, width, 0, 0, NULL);
        } else if (is_fence(t, e)) {
            if (!in_code) {
                const char* lang = t + 3;
                const char* le = e;
                while (le > lang && (le[-1] == ' ' || le[-1] == '\r')) le--;
                if (le > lang) add_run(L, &font_small, C_TEXT_MUTED, code_pad, text_width(&font_small, lang, le), lang, le, 0);
                end_line(L, le > lang ? font_small.line + 14 : 16, 10 + font_small.ascent, DECO_CODE_TOP, 0);
            } else {
                end_line(L, 16, 0, DECO_CODE_BOT, 0);
            }
            in_code = !in_code;
            blank = 0;
        } else if (in_code) {   // code keeps its spaces; long lines are cut at the panel's edge
            const char* c = s;
            int x = code_pad, lh = mono.line + 4;
            const char* run_s = c;
            int run_x = x;
            while (c < e) {
                const char* nx = c;
                utf8_next(&nx);
                int cw = text_width(&mono, c, nx);
                if (x + cw > width - code_pad && c > run_s) {
                    add_run(L, &mono, 0xC7E2FF, run_x, x - run_x, run_s, c, 0);
                    end_line(L, lh, 2 + mono.ascent, DECO_CODE, 0);
                    run_s = c, x = run_x = code_pad + 2 * mono.space;
                }
                x += cw, c = nx;
            }
            if (e > run_s) add_run(L, &mono, 0xC7E2FF, run_x, x - run_x, run_s, e, 0);
            end_line(L, lh, 2 + mono.ascent, DECO_CODE, 0);
        } else if (*s == '\x01') {   // table row
            if (s[1] == 'T') header = 1;
            if (s[1] == 'M') header = 0;
            Line* ln = end_line(L, s[1] == 'R' ? font.line + 16 : 2, 0, DECO_TABLE, 0);
            ln->a = s, ln->b = e, ln->header = (uint8_t)(header && s[1] == 'R');
            if (s[1] == 'B') L->height += 10;
            blank = 0;
        } else if (t == e || (e - t == 1 && *t == '\r')) {   // blank line: paragraph gap
            if (!blank && L->n_lines) L->height += 14;
            blank = 1;
        } else if (is_rule(t, e)) {
            end_line(L, 28, 0, DECO_HR, 0);
            blank = 0;
        } else {
            blank = 0;
            int quote = 0;
            while (t < e && *t == '>') {
                quote++, t++;
                while (t < e && *t == ' ') t++;
            }
            int heading = 0;
            if (*t == '#') {
                const char* h = t;
                while (h < e && *h == '#' && heading < 6) h++, heading++;
                if (h < e && *h == ' ') {
                    t = h + 1;
                    if (L->n_lines) L->height += 10;
                } else heading = 0;
            }
            int x0 = quote * QUOTE_INDENT;
            Run marker = {0};
            int has_marker = 0;
            if (!heading && t + 1 < e && (*t == '-' || *t == '*' || *t == '+') && t[1] == ' ') {
                int level = (spaces + 1) / 3 > 3 ? 3 : (spaces + 1) / 3;   // 2-4 spaces: second level
                static const char* dots[] = {"•", "◦", "▪", "▪"};
                const char* d = dots[level];
                marker.f = &font, marker.color = L->pal->marker, marker.s = d, marker.e = d + strlen(d);
                marker.x = x0 + level * LIST_INDENT + 6, marker.w = text_width(&font, marker.s, marker.e);
                x0 += level * LIST_INDENT + LIST_INDENT;
                t += 2;
                has_marker = 1;
            } else if (!heading && *t >= '0' && *t <= '9') {
                const char* d = t;
                while (d < e && *d >= '0' && *d <= '9' && d - t < 4) d++;
                if (d + 1 < e && (*d == '.' || *d == ')') && d[1] == ' ') {
                    int level = (spaces + 1) / 3 > 3 ? 3 : (spaces + 1) / 3;   // 2-4 spaces: second level
                    marker.f = &font_bold, marker.color = L->pal->marker, marker.s = t, marker.e = d + 1;
                    marker.w = text_width(&font_bold, t, d + 1);
                    int col = x0 + level * LIST_INDENT;
                    int tx = col + (marker.w + 12 > LIST_INDENT ? marker.w + 12 : LIST_INDENT);
                    marker.x = tx - 10 - marker.w;
                    x0 = tx;
                    t = d + 2;
                    has_marker = 1;
                }
            }
            parse_inline(t, e, 0);
            flow(L, x0, x0, width, heading, quote, has_marker ? &marker : NULL);
        }
        s = nl ? nl + 1 : e;
    }
    if (in_code) end_line(L, 16, 0, DECO_CODE_BOT, 0);   // a code block still being written
}

// The layout of text at this width, rebuilt only when something changed.
static Layout* layout_get(Layout** pl, const char* text, size_t len, int width, const Palette* pal, int plain) {
    Layout* L = *pl;
    if (!L) L = *pl = calloc(1, sizeof(Layout));
    if (!L->src || L->src_len != len || L->width != width || L->pal != pal || L->plain != plain || memcmp(L->src, text, len))
        layout_build(L, text, len, width, pal, plain);
    return L;
}

static void draw_layout(const Layout* L, int ox, int oy) {
    for (int i = 0; i < L->n_lines; i++) {
        const Line* ln = &L->lines[i];
        int ly = oy + ln->y;
        if (ly > clip_y1) break;
        if (ln->deco == DECO_CODE_TOP) {   // the whole code block's panel, even if its top is scrolled away
            int j = i;
            while (j + 1 < L->n_lines && L->lines[j].deco != DECO_CODE_BOT) j++;
            int bottom = oy + L->lines[j].y + L->lines[j].h;
            rrect(ox, ly, L->width, bottom - ly, 14, 0x061127, 255);
            rrect_line(ox, ly, L->width, bottom - ly, 14, 1, C_LINE, 255);
        }
        if (ly + ln->h < clip_y0) continue;
        if (ln->deco == DECO_HR) fill(ox, ly + ln->h / 2, L->width, 2, C_LINE_HI);
        if (ln->deco == DECO_TABLE) draw_table_line(ln->a, ln->b, ox, ly, ln->h, ln->header);
        for (int q = 0; q < ln->quote; q++) rrect(ox + q * QUOTE_INDENT + 4, ly, 4, ln->h, 2, C_ACCENT, 255);
        for (int k = 0; k < ln->nrun; k++) {
            const Run* r = &L->runs[ln->run0 + k];
            int ty = ly + ln->base - r->f->ascent;
            if (r->style & ST_CODE) rrect(ox + r->x - 5, ly + ln->base - mono.ascent - 4, r->w + 10, mono.line + 6, 8,
                                          L->pal->code_bg, 255);
            draw_text(r->f, ox + r->x, ty, r->s, r->e, r->color);
            if (r->style & ST_STRIKE) fill(ox + r->x, ly + ln->base - r->f->ascent / 3, r->w, 2, r->color);
            if (r->style & ST_LINK) fill(ox + r->x, ly + ln->base + 3, r->w, 1, r->color);
        }
    }
}

// Where the text ends (for the caret): x after the last run of the last line, and that line's top and height.
static void layout_end(const Layout* L, int* x, int* y, int* h) {
    *x = 0, *y = 0, *h = TEXT_LINE_H;
    if (!L->n_lines) return;
    const Line* ln = &L->lines[L->n_lines - 1];
    *y = ln->y, *h = ln->h;
    if (ln->nrun) {
        const Run* r = &L->runs[ln->run0 + ln->nrun - 1];
        *x = r->x + r->w;
    }
}

// ---- buttons and footer --------------------------------------------------------------------------------------

// Draws a controller button with its left edge at x, centred on cy; returns its width.
static int draw_btn(int btn, int x, int cy) {
    const int d = 34;
    float cx = x + d * 0.5f;
    if (btn <= BTN_SQUARE || btn == BTN_UPDOWN) {
        circle(cx, cy, d * 0.5f, 0, 0x1A2D52, 255);
        circle(cx, cy, d * 0.5f, 1.2f, 0x34548A, 255);
        switch (btn) {
        case BTN_CROSS:
            line(cx - 6.5f, cy - 6.5f, cx + 6.5f, cy + 6.5f, 3.2f, C_CROSS, 255);
            line(cx - 6.5f, cy + 6.5f, cx + 6.5f, cy - 6.5f, 3.2f, C_CROSS, 255);
            break;
        case BTN_CIRCLE: circle(cx, cy, 8.5f, 3.0f, C_CIRCLE, 255); break;
        case BTN_TRIANGLE:
            line(cx, cy - 8, cx + 8.5f, cy + 6, 2.8f, C_TRIANGLE, 255);
            line(cx + 8.5f, cy + 6, cx - 8.5f, cy + 6, 2.8f, C_TRIANGLE, 255);
            line(cx - 8.5f, cy + 6, cx, cy - 8, 2.8f, C_TRIANGLE, 255);
            break;
        case BTN_SQUARE: rrect_line((int)cx - 8, cy - 8, 16, 16, 2, 3, C_SQUARE, 255); break;
        default:   // up / down
            line(cx - 5, cy - 3, cx, cy - 8, 2.6f, 0xDCE6F5, 255);
            line(cx, cy - 8, cx + 5, cy - 3, 2.6f, 0xDCE6F5, 255);
            line(cx - 5, cy + 3, cx, cy + 8, 2.6f, 0xDCE6F5, 255);
            line(cx, cy + 8, cx + 5, cy + 3, 2.6f, 0xDCE6F5, 255);
        }
        return d;
    }
    static const char* names[] = {[BTN_L1] = "L1", [BTN_R1] = "R1", [BTN_L2] = "L2", [BTN_R2] = "R2", [BTN_OPTIONS] = "OPTIONS",
                                  [BTN_R3] = "R3"};
    const char* s = names[btn];
    int w = str_width(&font_small, s) + 20, h = 30;
    int rad[4] = {8, 8, 8, 8};
    if (btn == BTN_L2 || btn == BTN_R2) rad[0] = rad[1] = 14, rad[2] = rad[3] = 6;
    rrect4(x, cy - h / 2, w, h, rad, 0, 0x1A2D52, 255);
    rrect4(x, cy - h / 2, w, h, rad, 1, 0x34548A, 255);
    draw_str(&font_small, x + 10, cy - font_small.line / 2, s, 0xDCE6F5);
    return w;
}

static int hint_width(const Hint* h) {
    static const int glyph_w[] = {34, 34, 34, 34, 0, 0, 0, 0, 34, 0, 0};
    int g = glyph_w[h->btn];
    if (!g) {
        static const char* names[] = {[BTN_L1] = "L1", [BTN_R1] = "R1", [BTN_L2] = "L2", [BTN_R2] = "R2", [BTN_OPTIONS] = "OPTIONS",
                                      [BTN_R3] = "R3"};
        g = str_width(&font_small, names[h->btn]) + 20;
    }
    return g + 10 + str_width(&font_small, tr(h->label));
}

static int footer_extra;   // (chat screen) room kept on the right for the L2 / R2 buttons drawn after the footer

void draw_footer(const Hint* hints, int n, const char* right) {
    const int y = SCREEN_H - FOOTER_H, cy = y + FOOTER_H / 2;
    memcpy(back + y * SCREEN_W, bg + y * SCREEN_W, FOOTER_H * SCREEN_W * 4);   // a clean footer bar
    int right_w = right ? str_width(&font_small, right) + 40 : 0;
    int avail = SCREEN_W - 2 * 72 - right_w - footer_extra, total = 0;
    for (int i = 0; i < n; i++) total += hint_width(&hints[i]);
    int gap = n > 1 ? (avail - total) / (n - 1) : 0;
    if (gap > 44) gap = 44;
    if (gap < 14) gap = 14;
    int x = 72;
    for (int i = 0; i < n; i++) {
        int w = draw_btn(hints[i].btn, x, cy);
        draw_str(&font_small, x + w + 10, cy - font_small.line / 2, tr(hints[i].label), C_TEXT_DIM);
        x += hint_width(&hints[i]) + gap;
    }
    if (right) draw_str(&font_small, SCREEN_W - 72 - str_width(&font_small, right), cy - font_small.line / 2, right, C_TEXT_MUTED);
}

// ---- header ----------------------------------------------------------------------------------------------------

static void fmt_k(char* out, int n) {   // 1234 -> "1.2k"
    if (n < 1000) sprintf(out, "%d", n);
    else if (n < 9950) sprintf(out, "%d.%dk", (n + 50) / 1000, (n + 50) % 1000 / 100);
    else sprintf(out, "%dk", n / 1000);
}

// A rounded label at the right edge x_right; returns its left edge.
static int chip(int x_right, int cy, const char* text, uint32_t color, uint32_t dot, float dot_alpha, int bar_frac_x1000) {
    int tw = str_width(&font_small, text);
    int w = tw + 36 + (dot ? 20 : 0) + (bar_frac_x1000 >= 0 ? 96 : 0), h = 44, x = x_right - w;
    rrect(x, cy - h / 2, w, h, h / 2, 0x0D1D3B, 230);
    rrect_line(x, cy - h / 2, w, h, h / 2, 1, C_LINE_HI, 255);
    int tx = x + 18;
    if (dot) {
        circle(tx + 5, cy, 5, 0, dot, (int)(255 * dot_alpha));
        tx += 20;
    }
    draw_str(&font_small, tx, cy - font_small.line / 2, text, color);
    if (bar_frac_x1000 >= 0) {
        int bx = tx + tw + 14;
        rrect(bx, cy - 3, 78, 6, 3, 0x1B3156, 255);
        int fw = 78 * bar_frac_x1000 / 1000;
        if (fw > 0) rrect(bx, cy - 3, fw < 6 ? 6 : fw, 6, 3, bar_frac_x1000 > 850 ? 0xF2A65A : C_ACCENT_HI, 255);
    }
    return x;
}

static void draw_header(const ChatView* v) {
    draw_image(&logo_s, 64, (HEADER_H - logo_s.h) / 2 - 2, 255);
    int tx = 64 + (logo_s.w ? logo_s.w + 14 : 0);
    tx = draw_str(&font_big, tx, (HEADER_H - font_big.line) / 2, "Garlic", 0xFFFFFF);
    draw_str(&font_big, tx + font_big.space, (HEADER_H - font_big.line) / 2, "LLM", C_ACCENT_HI);
    int cy = HEADER_H / 2, x = SCREEN_W - 64;
    char a[24], b[24], ctx[96];
    fmt_k(a, v->pos), fmt_k(b, v->max_seq);
    snprintf(ctx, sizeof(ctx), "%s  %s / %s", tr(S_CONTEXT), a, b);
    int frac = v->max_seq ? (int)(1000.0 * v->pos / v->max_seq) : 0;
    x = chip(x, cy, ctx, C_TEXT_DIM, 0, 0, frac) - 14;
    float pulse = v->generating || v->reading ? 0.55f + 0.45f * sinf((float)v->time * 6.0f) : 1.0f;
    x = chip(x, cy, v->model ? v->model : "", 0xFFFFFF, v->generating || v->reading ? C_ACCENT_HI : C_GREEN, pulse, -1) - 14;
    if (v->think_on) {   // the next question is thought first
        char t[48];
        snprintf(t, sizeof(t), "%s · %d", tr(S_THINK_CHIP), v->think_cap);
        x = chip(x, cy, t, C_ACCENT_HI, C_ACCENT_HI, 1.0f, -1) - 14;
    }
    if (v->generating && v->tok_per_s > 0) {
        char s[48];
        snprintf(s, sizeof(s), "%.1f tok/s", v->tok_per_s);
        chip(x, cy, s, C_ACCENT_HI, 0, 0, -1);
    } else if (v->reading) {
        chip(x, cy, tr(S_READING), C_TEXT_DIM, 0, 0, -1);
    }
}

// ---- chat ------------------------------------------------------------------------------------------------------

static int msg_layout_width(int who) { return who == MSG_USER ? USER_MAX_W - 2 * BUB_PAD_X : who == MSG_AI ? COL_W - AI_INDENT : COL_W - 240; }
static const Palette* msg_palette(int who) { return who == MSG_USER ? &PAL_USER : who == MSG_AI ? &PAL_AI : &PAL_NOTE; }

// ---- thinking: the reasoning's last lines while the model thinks; then a short line over the answer ----------------

#define THINK_LINES 4
#define THINK_HEAD_H 50
#define THINK_PAD 16
#define THINK_TAIL 900   // bytes of the reasoning laid out (from the start of a line): its last lines are shown

static int think_panel_h(void) { return THINK_HEAD_H + 2 * THINK_PAD + THINK_LINES * TEXT_LINE_H; }
static int think_label_h(void) { return font_small.line + 14; }

static void draw_thinking(const ChatView* v, int x, int y, int w) {
    static Layout* lay;
    static char tail[THINK_TAIL + 3200];
    int cy = y + 24;   // level with the logo next to the reply
    float a = 0.5f + 0.5f * sinf((float)v->time * 6.0f);
    circle(x + 8, cy, 7, 0, C_ACCENT_HI, (int)(110 + 145 * a));
    draw_str(&font, x + 28, cy - font.line / 2, tr(S_THINKING), C_ACCENT_HI);
    char s[64];
    snprintf(s, sizeof(s), "%d / %d  ·  %d s", v->think_n, v->think_cap, (int)v->think_s);
    draw_str(&font_small, x + w - str_width(&font_small, s), cy - font_small.line / 2, s, C_TEXT_MUTED);
    int by = y + THINK_HEAD_H - 4, n = v->think_n < v->think_cap ? v->think_n : v->think_cap;
    rrect(x, by, w, 4, 2, 0x1B3156, 255);
    int fw = v->think_cap > 0 ? (int)((long long)w * n / v->think_cap) : 0;
    if (fw > 0) rrect(x, by, fw < 4 ? 4 : fw, 4, 2, C_ACCENT_HI, 255);
    int py = y + THINK_HEAD_H + 8, ph = 2 * THINK_PAD + THINK_LINES * TEXT_LINE_H - 8;
    rrect(x, py, w, ph, 18, C_SURFACE, 210);
    rrect_line(x, py, w, ph, 18, 1, C_LINE, 255);
    rrect(x + 14, py + 14, 4, ph - 28, 2, C_ACCENT, 170);
    const char* t = v->think_text;
    size_t len = v->think_len;
    if (!t || !len) return;
    size_t from = 0;   // the start of the line THINK_TAIL bytes from the end (or of a character, in a very long one)
    if (len > THINK_TAIL) {
        from = len - THINK_TAIL;
        while (from > 0 && t[from - 1] != '\n' && len - from < sizeof(tail) - 8) from--;
        if (from > 0 && t[from - 1] != '\n') {
            from = len - THINK_TAIL;
            while (from < len && ((uint8_t)t[from] & 0xC0) == 0x80) from++;
        }
    }
    int fences = 0;   // inside a code block there: open one, so that code stays code
    for (size_t i = 0; i + 2 < from; i++)
        if (t[i] == '`' && t[i + 1] == '`' && t[i + 2] == '`') fences++, i += 2;
    size_t k = 0;
    if (fences & 1) memcpy(tail, "```\n", 4), k = 4;
    memcpy(tail + k, t + from, len - from);
    k += len - from;
    tail[k] = 0;   // (layout_build reads up to the NUL; the buffer may still hold the end of a longer, earlier text)
    int tx = x + 34, tw = w - 34 - THINK_PAD;
    Layout* L = layout_get(&lay, tail, k, tw, &PAL_NOTE, 0);
    int cx0 = clip_x0, cy0 = clip_y0, cx1 = clip_x1, cy1 = clip_y1;
    int top = py + 10, bottom = py + ph - 10;
    // the top rows of the text fade into the panel (the line half shown there): the bare panel is kept first
    enum { FADE = 46 };
    static uint32_t band[FADE * SCREEN_W];
    int fx0 = tx - 4 > cx0 ? tx - 4 : cx0, fx1 = x + w - 4 < cx1 ? x + w - 4 : cx1;
    for (int r = 0; r < FADE && fx1 > fx0; r++)
        if (top + r >= cy0 && top + r < cy1) memcpy(band + r * SCREEN_W, back + (top + r) * SCREEN_W + fx0, (fx1 - fx0) * 4u);
    set_clip(cx0, top > cy0 ? top : cy0, cx1, bottom < cy1 ? bottom : cy1);
    draw_layout(L, tx, py + ph - THINK_PAD + 4 - L->height);
    set_clip(cx0, cy0, cx1, cy1);
    for (int r = 0; r < FADE && fx1 > fx0; r++) {
        if (top + r < cy0 || top + r >= cy1) continue;
        uint32_t a = 255 - 255u * r / FADE, *d = back + (top + r) * SCREEN_W + fx0;
        const uint32_t* b0 = band + r * SCREEN_W;
        for (int i = 0; i < fx1 - fx0; i++) blend(d + i, b0[i] & 0xFFFFFF, a);
    }
}

static void draw_thought(const Msg* m, int x, int y) {
    char s[96];
    int n = snprintf(s, sizeof(s), tr(S_THOUGHT), (int)(m->think_s + 0.5f), m->think_n);
    if (m->think_cut && n > 0 && n < (int)sizeof(s)) snprintf(s + n, sizeof(s) - n, " · %s", tr(S_THINK_CUT));
    circle(x + 6, y + font_small.line / 2, 5, 0, C_ACCENT, 255);
    draw_str(&font_small, x + 20, y, s, C_TEXT_MUTED);
}

static int msg_height(Msg* m, int typing) {
    Layout* L = layout_get(&m->lay, m->text, m->len, msg_layout_width(m->who), msg_palette(m->who), m->who == MSG_USER);
    if (m->who == MSG_USER) return L->height + 2 * BUB_PAD_Y;
    if (m->who == MSG_NOTE) return L->height + 8;
    int h = L->height + (m->think_n ? think_label_h() : 0);
    if (typing && h < 56) h = 56;
    return h < 56 ? 56 : h;
}

static void draw_empty(const ChatView* v) {
    int cx = SCREEN_W / 2, y = 250;
    circle(cx, y + logo_l.h / 2, 120, 0, C_ACCENT, 34);
    circle(cx, y + logo_l.h / 2, 92, 0, C_ACCENT_HI, 26);
    draw_image(&logo_l, cx - logo_l.w / 2, y, 255);
    y += logo_l.h + 26;
    int w1 = str_width(&font_big, "Garlic"), w2 = str_width(&font_big, "LLM");
    int x = cx - (w1 + font_big.space + w2) / 2;
    x = draw_str(&font_big, x, y, "Garlic", 0xFFFFFF);
    draw_str(&font_big, x + font_big.space, y, "LLM", C_ACCENT_HI);
    y += font_big.line + 14;
    char line_[160];
    snprintf(line_, sizeof(line_), tr(S_TAGLINE), v->model ? v->model : "");
    draw_center(&font, cx, y, line_, C_TEXT);
    y += font.line + 6;
    draw_center(&font_small, cx, y, tr(S_EMPTY_HINT), C_TEXT_MUTED);
    y += font_small.line + 44;
    // two action cards
    const Hint acts[2] = {{BTN_CROSS, S_WRITE}, {BTN_SQUARE, S_TALK_HOLD}};
    int ws[2], total = 0;
    for (int i = 0; i < 2; i++) ws[i] = 34 + 14 + str_width(&font, tr(acts[i].label)) + 56, total += ws[i];
    total += 28;
    x = cx - total / 2;
    for (int i = 0; i < 2; i++) {
        rrect(x, y, ws[i], 76, 20, C_SURFACE, 255);
        rrect_line(x, y, ws[i], 76, 20, 1, C_LINE_HI, 255);
        int bw = draw_btn(acts[i].btn, x + 28, y + 38);
        draw_str(&font, x + 28 + bw + 14, y + 38 - font.line / 2, tr(acts[i].label), 0xFFFFFF);
        x += ws[i] + 28;
    }
}

void draw_toast(const char* text, float alpha, int y) {
    if (!text || alpha <= 0) return;
    if (y <= 0) y = SCREEN_H - FOOTER_H - 36;
    text_alpha = (int)(255 * alpha);
    int tw = str_width(&font, text), w = tw + 64, h = 64, x = (SCREEN_W - w) / 2;
    rrect(x, y - h, w, h, h / 2, 0x16335F, 245);
    rrect_line(x, y - h, w, h, h / 2, 1, C_ACCENT_HI, 140);
    draw_str(&font, x + 32, y - h + (h - font.line) / 2, text, 0xFFFFFF);
    text_alpha = 255;
}

void draw_chat(ChatView* v) {
    draw_background();
    const int top = HEADER_H, bottom = SCREEN_H - FOOTER_H;
    const int area_top = top + 28, area_bottom = bottom - 24;
    if (!n_msgs) {
        draw_empty(v);
    } else {
        int typing_last = v->generating || v->reading;
        int total = 0;
        static int heights[MAX_MSGS];
        for (int i = 0; i < n_msgs; i++) {
            heights[i] = msg_height(&msgs[i], typing_last && i == n_msgs - 1);
            if (v->thinking && i == n_msgs - 1 && msgs[i].who == MSG_AI) heights[i] = think_panel_h();
            total += heights[i] + (i ? MSG_GAP : 0);
        }
        int view = area_bottom - area_top, max_scroll = total > view ? total - view : 0;
        if (v->scroll > max_scroll) v->scroll = max_scroll;
        if (v->scroll < 0) v->scroll = 0;
        int y = area_top + (total <= view ? 0 : view - total + v->scroll);
        set_clip(0, top + 1, SCREEN_W, bottom);
        for (int i = 0; i < n_msgs; i++) {
            Msg* m = &msgs[i];
            int h = heights[i];
            if (y + h >= top && y <= bottom) {
                Layout* L = m->lay;
                if (m->who == MSG_USER) {
                    int bw = L->max_w + 2 * BUB_PAD_X;
                    if (bw < 96) bw = 96;
                    int bx = COL_X + COL_W - bw;
                    int rad[4] = {26, 26, 8, 26};
                    rrect4(bx, y, bw, h, rad, 0, C_USER, 255);
                    draw_layout(L, bx + BUB_PAD_X, y + BUB_PAD_Y);
                } else if (m->who == MSG_NOTE) {
                    int nx = (SCREEN_W - L->max_w) / 2;
                    fill(COL_X + 120, y + h / 2, nx - COL_X - 150, 1, C_LINE);
                    fill(nx + L->max_w + 30, y + h / 2, COL_X + COL_W - 120 - nx - L->max_w - 30, 1, C_LINE);
                    draw_layout(L, nx, y + 4);
                } else {
                    circle(COL_X + 28, y + 28, 28, 0, 0x10244A, 255);
                    circle(COL_X + 28, y + 28, 28, 1.2f, C_LINE_HI, 255);
                    if (logo_s.px) draw_image(&logo_s, COL_X + 28 - logo_s.w / 2, y + 28 - logo_s.h / 2, 255);
                    int tx = COL_X + AI_INDENT, ty = y + 4;
                    if (v->thinking && i == n_msgs - 1) {   // still reasoning: its last lines instead
                        draw_thinking(v, tx, ty, COL_W - AI_INDENT);
                        y += h + MSG_GAP;
                        continue;
                    }
                    if (m->think_n) draw_thought(m, tx, ty), ty += think_label_h();
                    draw_layout(L, tx, ty);
                    if (typing_last && i == n_msgs - 1) {
                        if (!L->n_lines || !L->n_runs) {   // nothing written yet: three bouncing dots
                            for (int k = 0; k < 3; k++) {
                                float ph = (float)v->time * 5.0f - k * 0.7f;
                                float a = 0.35f + 0.65f * (0.5f + 0.5f * sinf(ph));
                                circle(tx + 10 + k * 26, ty + 24 - 5 * (0.5f + 0.5f * sinf(ph)), 7, 0, C_ACCENT_HI, (int)(255 * a));
                            }
                        } else {
                            int ex, ey, eh;
                            layout_end(L, &ex, &ey, &eh);
                            float a = 0.5f + 0.5f * sinf((float)v->time * 7.0f);
                            circle(tx + ex + 14, ty + ey + eh / 2, 7, 0, C_ACCENT_HI, (int)(90 + 165 * a));
                        }
                    }
                }
            }
            y += h + MSG_GAP;
        }
        no_clip();
        fade_rows(top + 1, top + 30, 0, 255);
        fade_rows(bottom - 30, bottom, 255, 0);
        if (max_scroll > 0) {   // scroll bar
            int track = area_bottom - area_top, x = COL_X + COL_W + 48;
            int th = track * view / total;
            if (th < 40) th = 40;
            int ty = area_top + (track - th) * (max_scroll - v->scroll) / max_scroll;
            rrect(x, area_top, 6, track, 3, 0x12244A, 255);
            rrect(x, ty, 6, th, 3, C_ACCENT_HI, 200);
        }
    }
    draw_header(v);
    Hint h[8];
    int n = 0;
    if (v->generating) h[n++] = (Hint){BTN_CIRCLE, S_STOP};
    else h[n++] = (Hint){BTN_CROSS, S_WRITE}, h[n++] = (Hint){BTN_SQUARE, S_TALK_HOLD};
    h[n++] = (Hint){BTN_TRIANGLE, S_NEW_CHAT};
    if (n_msgs) h[n++] = (Hint){BTN_UPDOWN, S_SCROLL};
    if (!v->generating) h[n++] = (Hint){BTN_L1, S_VOICE_MODE};
    if (v->can_run && !v->generating) h[n++] = (Hint){BTN_R1, S_PY_RUN};
    char right[64];
    snprintf(right, sizeof(right), "%s %.1f", tr(S_TEMPERATURE), v->temperature);
    if (v->think_avail) {   // R3: thinking. The footer leaves room for L2 / R2; without enough (Portuguese and
        footer_extra = 140;   // Russian with R1 shown) the scroll hint goes, then this one
        if (!v->generating) h[n++] = (Hint){BTN_R3, v->think_on ? S_THINK_ON_HINT : S_THINK_OFF_HINT};
        int avail = SCREEN_W - 2 * 72 - str_width(&font_small, right) - 40 - footer_extra, total = -14;
        for (int i = 0; i < n; i++) total += hint_width(&h[i]) + 14;
        for (int i = 0; i < n && total > avail; i++)
            if (h[i].btn == BTN_UPDOWN) {
                total -= hint_width(&h[i]) + 14;
                memmove(&h[i], &h[i + 1], (n - i - 1) * sizeof(Hint));
                n--;
            }
        if (total > avail && h[n - 1].btn == BTN_R3) n--;
    }
    draw_footer(h, n, right);
    footer_extra = 0;
    int rx = SCREEN_W - 72 - str_width(&font_small, right) - 18;   // L2 / R2 next to the temperature
    int w2 = draw_btn(BTN_R2, rx - 50, SCREEN_H - FOOTER_H / 2);
    draw_btn(BTN_L2, rx - 50 - w2 - 8, SCREEN_H - FOOTER_H / 2);
    draw_toast(v->toast, v->toast_alpha, 0);
}

// ---- loading and error screens --------------------------------------------------------------------------------

static void brand_block(int y) {
    int cx = SCREEN_W / 2;
    circle(cx, y + logo_l.h / 2, 130, 0, C_ACCENT, 30);
    circle(cx, y + logo_l.h / 2, 98, 0, C_ACCENT_HI, 22);
    draw_image(&logo_l, cx - logo_l.w / 2, y, 255);
    y += logo_l.h + 28;
    int w1 = str_width(&font_big, "Garlic"), w2 = str_width(&font_big, "LLM");
    int x = cx - (w1 + font_big.space + w2) / 2;
    x = draw_str(&font_big, x, y, "Garlic", 0xFFFFFF);
    draw_str(&font_big, x + font_big.space, y, "LLM", C_ACCENT_HI);
}

void draw_loading(float frac, const char* what) {
    draw_background();
    brand_block(270);
    int cx = SCREEN_W / 2, y = 600, w = 560;
    rrect(cx - w / 2, y, w, 10, 5, 0x12264A, 255);
    int fw = (int)(w * clamp01(frac));
    if (fw > 0) {
        rrect(cx - w / 2, y, fw < 10 ? 10 : fw, 10, 5, C_ACCENT, 255);
        rrect(cx - w / 2 + (fw > 40 ? fw - 40 : 0), y, fw > 40 ? 40 : fw, 10, 5, C_ACCENT_HI, 200);
    }
    char line_[160];
    snprintf(line_, sizeof(line_), "%s…  %d %%", what, (int)(clamp01(frac) * 100));
    draw_center(&font_small, cx, y + 30, line_, C_TEXT_DIM);
    draw_center(&font_small, cx, SCREEN_H - 90, "Qwen3.5  ·  Parakeet  ·  Chatterbox", C_TEXT_MUTED);
}

void draw_error(const char* what) {
    draw_background();
    brand_block(220);
    int cx = SCREEN_W / 2, y = 560;
    draw_center(&font_big, cx, y, tr(S_ERROR_TITLE), C_RED);
    static Layout* L;
    Layout* l = layout_get(&L, what, strlen(what), 1200, &PAL_AI, 1);
    draw_layout(l, cx - l->max_w / 2, y + font_big.line + 20);
    draw_center(&font_small, cx, y + font_big.line + 40 + l->height, tr(S_ERROR_HINT), C_TEXT_MUTED);
}

// ---- on-screen keyboard ----------------------------------------------------------------------------------------

#define KB_ROWS 6
#define KB_COLS 12
// Special keys: ⇧ shift, ⌫ delete, ␣ space, ⏎ send, [emoji], [lang] (next layout); dead keys (accents) are
// listed per layout in kb_dead. A key repeated in neighbouring columns is one wide key.
static const char* kb_layouts[N_KB_LAYOUTS][KB_ROWS][KB_COLS] = {
    [KB_ES] = {{"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "¿", "?"},
               {"q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "´", "⌫"},
               {"a", "s", "d", "f", "g", "h", "j", "k", "l", "ñ", "¡", "!"},
               {"⇧", "z", "x", "c", "v", "b", "n", "m", ",", ".", ":", "-"},
               {";", "_", "+", "=", "*", "/", "%", "&", "<", ">", "(", ")"},
               {"[lang]", "\"", "'", "@", "#", "␣", "␣", "␣", "[emoji]", "[emoji]", "⏎", "⏎"}},
    [KB_EN] = {{"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "?", "!"},
               {"q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "'", "⌫"},
               {"a", "s", "d", "f", "g", "h", "j", "k", "l", ":", ";", "\""},
               {"⇧", "z", "x", "c", "v", "b", "n", "m", ",", ".", "-", "/"},
               {"_", "+", "=", "*", "%", "&", "<", ">", "(", ")", "[", "]"},
               {"[lang]", "@", "#", "$", "~", "␣", "␣", "␣", "[emoji]", "[emoji]", "⏎", "⏎"}},
    [KB_PT] = {{"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "?", "!"},
               {"q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "´", "⌫"},
               {"a", "s", "d", "f", "g", "h", "j", "k", "l", "ç", "~", "^"},
               {"⇧", "z", "x", "c", "v", "b", "n", "m", ",", ".", ":", "-"},
               {";", "_", "+", "=", "*", "/", "%", "&", "(", ")", "\"", "'"},
               {"[lang]", "@", "#", "`", "$", "␣", "␣", "␣", "[emoji]", "[emoji]", "⏎", "⏎"}},
    [KB_RU] = {{"1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "?", "!"},
               {"й", "ц", "у", "к", "е", "н", "г", "ш", "щ", "з", "х", "⌫"},
               {"ф", "ы", "в", "а", "п", "р", "о", "л", "д", "ж", "э", "ъ"},
               {"⇧", "я", "ч", "с", "м", "и", "т", "ь", "б", "ю", "ё", ","},
               {".", ":", ";", "-", "_", "+", "=", "*", "/", "(", ")", "\""},
               {"[lang]", "@", "#", "%", "'", "␣", "␣", "␣", "[emoji]", "[emoji]", "⏎", "⏎"}},
};
static const char* kb_names[N_KB_LAYOUTS] = {"ES", "EN", "PT", "RU"};
static const char* kb_dead[N_KB_LAYOUTS][4] = {[KB_ES] = {"´"}, [KB_PT] = {"´", "~", "^", "`"}};

static int is_dead(int layout, const char* key) {
    for (int i = 0; i < 4 && kb_dead[layout][i]; i++)
        if (!strcmp(kb_dead[layout][i], key)) return 1;
    return 0;
}

#include "emoji_pages.h"
static const char* kb_emoji_bar[KB_COLS] = {"[abc]", "[abc]", "[t0]", "[t1]", "[t2]", "[t3]", "[t4]",
                                            "[t5]", "[t6]", "⌫", "⏎", "⏎"};
_Static_assert(EMOJI_ROWS == KB_ROWS - 1 && EMOJI_COLS == KB_COLS && EMOJI_PAGES == 7, "emoji layout");
static const char* emoji_titles[N_LANGS][EMOJI_PAGES] = {
    {"Caras", "Gestos y personas", "Animales y naturaleza", "Comida y bebida", "Actividades y viajes", "Objetos",
     "Corazones y símbolos"},
    {"Smileys", "Gestures & people", "Animals & nature", "Food & drink", "Activities & travel", "Objects",
     "Hearts & symbols"},
    {"Carinhas", "Gestos e pessoas", "Animais e natureza", "Comida e bebida", "Atividades e viagens", "Objetos",
     "Corações e símbolos"},
    {"Смайлики", "Жесты и люди", "Животные и природа", "Еда и напитки", "Занятия и путешествия", "Предметы",
     "Сердца и символы"},
};

static const char* kb_key(const Kb* kb, int row, int col) {
    if (kb->page == 0) return kb_layouts[kb->layout][row][col];
    if (row < EMOJI_ROWS) return emoji_keys[kb->page - 1][row * EMOJI_COLS + col];
    return kb_emoji_bar[col];
}

static int kb_span(const Kb* kb, int row, int col) {
    int span = 1;
    while (col + span < KB_COLS && !strcmp(kb_key(kb, row, col + span), kb_key(kb, row, col))) span++;
    return span;
}

void kb_init(Kb* kb, int layout) {
    memset(kb, 0, sizeof(*kb));
    kb->layout = layout, kb->row = 1, kb->last_emoji_page = 1;
}

void kb_move(Kb* kb, int dx, int dy) {
    if (dy) kb->row = (kb->row + KB_ROWS + dy) % KB_ROWS;
    if (dx < 0) {
        while (kb->col > 0 && !strcmp(kb_key(kb, kb->row, kb->col - 1), kb_key(kb, kb->row, kb->col))) kb->col--;
        kb->col = (kb->col + KB_COLS - 1) % KB_COLS;
    }
    if (dx > 0) {
        while (kb->col < KB_COLS - 1 && !strcmp(kb_key(kb, kb->row, kb->col + 1), kb_key(kb, kb->row, kb->col))) kb->col++;
        kb->col = (kb->col + 1) % KB_COLS;
    }
}

void kb_toggle_emoji(Kb* kb) { kb->page = kb->page ? 0 : kb->last_emoji_page; }

void kb_append(char* out, int max, const char* s) {
    size_t n = strlen(out), m = strlen(s);
    if (n + m < (size_t)max) memcpy(out + n, s, m + 1);
}

void kb_backspace(char* out) {
    size_t n = strlen(out);
    while (n > 0) {
        n--;
        if (((uint8_t)out[n] & 0xC0) != 0x80) break;
    }
    out[n] = 0;
}

static uint32_t upper_cp(uint32_t c) {
    if (c >= 'a' && c <= 'z') return c - 32;
    if (c >= 0xE0 && c <= 0xFE && c != 0xF7) return c - 32;
    if (c >= 0x430 && c <= 0x44F) return c - 32;
    if (c >= 0x450 && c <= 0x45F) return c - 80;
    return c;
}

// A key's text given shift and a pending accent ("´" + "a" = "á").
static void kb_compose(const char* key, int shift, const char* dead, char* out, int max) {
    static const char* table[][3] = {{"´", "a", "á"}, {"´", "e", "é"}, {"´", "i", "í"}, {"´", "o", "ó"}, {"´", "u", "ú"},
                                     {"~", "a", "ã"}, {"~", "o", "õ"}, {"~", "n", "ñ"}, {"^", "a", "â"}, {"^", "e", "ê"},
                                     {"^", "o", "ô"}, {"`", "a", "à"}};
    const char* base = key;
    if (dead)
        for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++)
            if (!strcmp(table[i][0], dead) && !strcmp(table[i][1], key)) base = table[i][2];
    int n = 0;
    for (const char* p = base; *p && n < max - 5;) {
        uint32_t c = utf8_next(&p);
        n += utf8_put(shift ? upper_cp(c) : c, out + n);
    }
    out[n] = 0;
}

int kb_press(Kb* kb, char* out, int max) {
    const char* key = kb_key(kb, kb->row, kb->col);
    if (!strcmp(key, "⏎")) return 1;
    if (!strcmp(key, "⌫")) kb_backspace(out);
    else if (!strcmp(key, "⇧")) kb->shift = !kb->shift;
    else if (!strcmp(key, "␣")) kb_append(out, max, " "), kb->dead = NULL;
    else if (!strcmp(key, "[emoji]")) kb->page = kb->last_emoji_page;
    else if (!strcmp(key, "[abc]")) kb->page = 0;
    else if (!strcmp(key, "[lang]")) kb->layout = (kb->layout + 1) % N_KB_LAYOUTS, kb->shift = 0, kb->dead = NULL;
    else if (!strncmp(key, "[t", 2)) kb->page = kb->last_emoji_page = 1 + (key[2] - '0');
    else if (kb->page == 0 && is_dead(kb->layout, key)) {
        if (kb->dead && !strcmp(kb->dead, key)) kb_append(out, max, key), kb->dead = NULL;   // twice: the accent itself
        else kb->dead = key;
    } else {
        char s[32];
        kb_compose(key, kb->shift, kb->dead, s, sizeof(s));
        if (kb->dead && !strcmp(s, key)) kb_append(out, max, kb->dead);   // accent that does not combine: keep it
        kb_append(out, max, s);
        kb->dead = NULL, kb->shift = 0;   // shift applies to one letter, like a phone keyboard
    }
    if (kb->page) kb->last_emoji_page = kb->page;
    return 0;
}

#define KEY_W 112
#define KEY_H 64
#define KEY_GAP 10
#define KEYS_W (KB_COLS * (KEY_W + KEY_GAP) - KEY_GAP)
#define FIELD_H 116
#define PANEL_PAD 30
#define PANEL_H (PANEL_PAD + FIELD_H + 24 + KB_ROWS * (KEY_H + KEY_GAP) - KEY_GAP + PANEL_PAD)
#define PANEL_Y (SCREEN_H - FOOTER_H - PANEL_H)

static void draw_field(int x, int y, int w, const char* text, double time) {
    rrect(x, y, w, FIELD_H, 20, 0x050E22, 255);
    rrect_line(x, y, w, FIELD_H, 20, 2, C_ACCENT_HI, 170);
    int pad = 28, inner_w = w - 2 * pad;
    int caret_x, caret_y, caret_h;
    int blink = fmod(time, 1.0) < 0.6;
    if (!text[0]) {
        draw_str(&font, x + pad, y + (FIELD_H - font.line) / 2, tr(S_PLACEHOLDER), C_TEXT_MUTED);
        if (blink) rrect(x + pad - 4, y + (FIELD_H - 36) / 2, 3, 36, 1, C_ACCENT_HI, 255);
        return;
    }
    static Layout* L;
    Layout* l = layout_get(&L, text, strlen(text), inner_w, &PAL_FIELD, 1);
    int area = FIELD_H - 20;
    int oy = y + 10 + (l->height <= area ? (area - l->height) / 2 : area - l->height);   // keep the end in view
    set_clip(x + 4, y + 4, x + w - 4, y + FIELD_H - 4);
    draw_layout(l, x + pad, oy);
    layout_end(l, &caret_x, &caret_y, &caret_h);
    size_t n = strlen(text);
    for (size_t i = n; i > 0 && text[i - 1] == ' '; i--) caret_x += font.space;   // trailing spaces
    if (blink) rrect(x + pad + caret_x + 2, oy + caret_y + (caret_h - 36) / 2, 3, 36, 1, C_ACCENT_HI, 255);
    no_clip();
}

static void key_label(const Kb* kb, const char* key, char* out, int max, const Font** f) {
    *f = &font;
    if (!strcmp(key, "␣")) snprintf(out, max, "%s", tr(S_SPACE_KEY)), *f = &font_small;
    else if (!strcmp(key, "⏎")) snprintf(out, max, "%s", tr(S_SEND_KEY)), *f = &font_bold;
    else if (!strcmp(key, "[emoji]")) snprintf(out, max, "😀");
    else if (!strcmp(key, "[abc]")) snprintf(out, max, "%s", kb_layouts[kb->layout][1][0][0] & 0x80 ? "АБВ" : "ABC"), *f = &font_bold;
    else if (!strcmp(key, "[lang]")) snprintf(out, max, "🌐 %s", kb_names[kb->layout]), *f = &font_small;
    else if (!strncmp(key, "[t", 2)) snprintf(out, max, "%s", emoji_tab[key[2] - '0']), *f = &font_big;
    else if (kb->page > 0) snprintf(out, max, "%s", key), *f = &font_big;   // emoji at ~25 px
    else if (is_dead(kb->layout, key)) snprintf(out, max, "%s", key), *f = &font_big;   // accents are tiny: bigger
    else if (!strcmp(key, "⇧") || !strcmp(key, "⌫")) snprintf(out, max, "%s", key);
    else kb_compose(key, kb->shift, kb->dead, out, max);
}

// One key (its first column k), plain or selected.
static void kb_draw_key(const Kb* kb, int r, int k, int x0, int y0, int sel) {
    const char* key = kb_key(kb, r, k);
    int span = kb_span(kb, r, k);
    int x = x0 + k * (KEY_W + KEY_GAP), ky = y0 + r * (KEY_H + KEY_GAP), w = span * (KEY_W + KEY_GAP) - KEY_GAP;
    int special = key[0] == '[' || !strcmp(key, "⇧") || !strcmp(key, "⌫") || !strcmp(key, "␣") ||
                  (kb->page == 0 && is_dead(kb->layout, key));
    int enter = !strcmp(key, "⏎");
    int on = (!strcmp(key, "⇧") && kb->shift) || (kb->dead && !strcmp(key, kb->dead)) ||
             (kb->page > 0 && !strncmp(key, "[t", 2) && key[2] - '0' == kb->page - 1);
    if (sel) {
        rrect(x - 7, ky - 7, w + 14, KEY_H + 14, 20, C_ACCENT_HI, 40);
        rrect(x - 3, ky - 3, w + 6, KEY_H + 6, 16, C_ACCENT_HI, 90);
        rrect(x, ky, w, KEY_H, 13, 0x1C86EA, 255);
    } else {
        rrect(x, ky, w, KEY_H, 13, enter ? C_ACCENT : special ? C_KEY_SPECIAL : C_KEY, 255);
        if (on) rrect_line(x, ky, w, KEY_H, 13, 2, C_ACCENT_HI, 255);
    }
    char label[48];
    const Font* f;
    key_label(kb, key, label, sizeof(label), &f);
    int tw = str_width(f, label);
    uint32_t col = sel || enter ? 0xFFFFFF : on ? C_ACCENT_HI : special ? C_TEXT_DIM : 0xEEF4FF;
    draw_str(f, x + (w - tw) / 2, ky + (KEY_H - f->line) / 2, label, col);
}

// The keyboard is drawn in two layers: everything that only changes with the layout, page, shift or accent
// (backdrop, panel, every key, footer) is kept in kb_cache; each frame copies it and draws only the selected
// key and the text field on top. So moving around costs almost nothing, whatever is behind. (kb_cache is declared
// with the backdrop; without memory for it, every frame draws the whole keyboard.)

void draw_keyboard(const Kb* kb, const char* text, double time) {
    int px = (SCREEN_W - KEYS_W) / 2 - PANEL_PAD;
    int x0 = px + PANEL_PAD, y = PANEL_Y + PANEL_PAD, y0 = y + FIELD_H + 24;
    int sig = kb->layout | kb->page << 3 | kb->shift << 7 | ui_lang << 8 |
              (kb->dead ? 1 + (int)((uintptr_t)kb->dead % 4093) : 0) << 12;
    if (!kb_cache) kb_cache = ui_screen_buf();
    if (sig != kb_cache_sig || !kb_cache) {
        ui_backdrop_draw();
        int rad[4] = {30, 30, 0, 0};
        rrect4(px, PANEL_Y, KEYS_W + 2 * PANEL_PAD, PANEL_H + 1, rad, 0, C_PANEL, 255);
        rrect4(px, PANEL_Y, KEYS_W + 2 * PANEL_PAD, PANEL_H + 1, rad, 1, C_LINE_HI, 255);
        for (int r = 0; r < KB_ROWS; r++)
            for (int k = 0; k < KB_COLS; k++)
                if (k == 0 || strcmp(kb_key(kb, r, k), kb_key(kb, r, k - 1))) kb_draw_key(kb, r, k, x0, y0, 0);
        const Hint h_letters[] = {{BTN_CROSS, S_KB_TYPE}, {BTN_SQUARE, S_KB_DELETE}, {BTN_TRIANGLE, S_KB_SPACE},
                                  {BTN_L1, S_KB_SHIFT},   {BTN_R1, S_KB_EMOJI},      {BTN_L2, S_KB_DICTATE},
                                  {BTN_R2, S_KB_SEND},    {BTN_CIRCLE, S_KB_CLOSE}};
        const Hint h_emoji[] = {{BTN_CROSS, S_KB_TYPE}, {BTN_SQUARE, S_KB_DELETE}, {BTN_TRIANGLE, S_KB_SPACE},
                                {BTN_R1, S_KB_LETTERS}, {BTN_R2, S_KB_SEND},      {BTN_CIRCLE, S_KB_CLOSE}};
        if (kb->page) draw_footer(h_emoji, 6, emoji_titles[ui_lang][kb->page - 1]);
        else draw_footer(h_letters, 8, NULL);
        if (kb_cache) {
            memcpy(kb_cache, back, SCREEN_W * SCREEN_H * 4);
            kb_cache_sig = sig;
        }
    } else {
        memcpy(back, kb_cache, SCREEN_W * SCREEN_H * 4);
    }
    int k = kb->col;
    while (k > 0 && !strcmp(kb_key(kb, kb->row, k - 1), kb_key(kb, kb->row, k))) k--;   // first column of a wide key
    kb_draw_key(kb, kb->row, k, x0, y0, 1);
    draw_field(x0, y, KEYS_W, text, time);
}

// ---- voice -------------------------------------------------------------------------------------------------------

static void mic_icon(float cx, float cy, float s, uint32_t col) {   // s = 1: about 60 px tall
    rrect((int)(cx - 13 * s), (int)(cy - 30 * s), (int)(26 * s), (int)(42 * s), (int)(13 * s), col, 255);
    int sx0 = clip_x0, sy0 = clip_y0, sx1 = clip_x1, sy1 = clip_y1;
    set_clip((int)(cx - 40 * s), (int)(cy - 6 * s), (int)(cx + 40 * s), (int)(cy + 40 * s));
    circle(cx, cy - 8 * s, 21 * s, 4 * s, col, 255);
    set_clip(sx0, sy0, sx1, sy1);
    line(cx, cy + 13 * s, cx, cy + 25 * s, 4 * s, col, 255);
    line(cx - 11 * s, cy + 26 * s, cx + 11 * s, cy + 26 * s, 4 * s, col, 255);
}

void draw_voice(int phase, const float* levels, int n_levels, float secs, float max_secs, double time) {
    int w = 760, h = 420, x = (SCREEN_W - w) / 2, y = HEADER_H + (SCREEN_H - HEADER_H - FOOTER_H - h) / 2;
    rrect(x, y, w, h, 34, C_PANEL, 255);
    rrect_line(x, y, w, h, 34, 1, C_LINE_HI, 255);
    int cx = SCREEN_W / 2, cy = y + 140;
    float level = n_levels ? levels[n_levels - 1] : 0;
    if (phase == VOICE_RECORDING) {
        float pulse = 0.5f + 0.5f * sinf((float)time * 4.0f);
        circle(cx, cy, 78 + 26 * level + 6 * pulse, 0, C_ACCENT_HI, 40);
        circle(cx, cy, 66 + 12 * level, 0, C_ACCENT_HI, 60);
        circle(cx, cy, 58, 0, C_ACCENT, 255);
        mic_icon(cx, cy + 2, 1.0f, 0xFFFFFF);
        draw_center(&font_big, cx, y + 236, tr(S_LISTENING), 0xFFFFFF);
        char t[64];
        snprintf(t, sizeof(t), "%d:%02d / %d:%02d", (int)secs / 60, (int)secs % 60, (int)max_secs / 60, (int)max_secs % 60);
        draw_center(&font_small, cx, y + 298, t, C_TEXT_DIM);
        const int bars = 40, bw = 9, gap = 5, total = bars * (bw + gap) - gap;
        for (int i = 0; i < bars; i++) {
            int src = n_levels - bars + i;
            float lv = src >= 0 ? levels[src] : 0;
            int bh = 6 + (int)(lv * 46);
            int bx = cx - total / 2 + i * (bw + gap), by = y + 372 - bh / 2;
            rrect(bx, by, bw, bh, 4, i > bars - 4 ? C_ACCENT_HI : 0x3C78C8, 255);
        }
        const Hint hh[] = {{BTN_SQUARE, S_RELEASE_TO_END}};
        draw_footer(hh, 1, NULL);
    } else {
        for (int k = 0; k < 12; k++) {   // spinner
            float a = (float)k / 12 * 6.2831853f + (float)time * 6.0f;
            float fade = (float)k / 12;
            circle(cx + 44 * cosf(a), cy + 44 * sinf(a), 7, 0, C_ACCENT_HI, (int)(40 + 215 * fade));
        }
        draw_center(&font_big, cx, y + 236, tr(S_TRANSCRIBING), 0xFFFFFF);
        draw_footer(NULL, 0, NULL);
    }
}

void draw_confirm(const char* text) {
    static Layout* L;
    int w = COL_W, pad = 36;
    Layout* l = layout_get(&L, text, strlen(text), w - 2 * pad - 90, &PAL_FIELD, 1);
    int text_h = l->height > 4 * TEXT_LINE_H ? 4 * TEXT_LINE_H : l->height;
    int h = 64 + text_h + 30, x = COL_X, y = SCREEN_H - FOOTER_H - 28 - h;
    rrect(x, y, w, h, 28, C_PANEL, 255);
    rrect_line(x, y, w, h, 28, 2, C_ACCENT_HI, 150);
    circle(x + pad + 30, y + h / 2, 30, 0, C_ACCENT, 255);
    mic_icon(x + pad + 30, y + h / 2 + 1, 0.5f, 0xFFFFFF);
    int tx = x + pad + 90;
    draw_str(&font_small, tx, y + 22, tr(S_YOUR_MESSAGE), C_ACCENT_HI);
    set_clip(tx, y + 58, x + w - pad, y + 58 + text_h + 4);
    draw_layout(l, tx, y + 58 + (l->height > text_h ? text_h - l->height : 0));
    no_clip();
    const Hint hh[] = {{BTN_CROSS, S_SEND}, {BTN_TRIANGLE, S_EDIT}, {BTN_SQUARE, S_RETRY}, {BTN_CIRCLE, S_DISCARD}};
    draw_footer(hh, 4, NULL);
}

#include "ui_run.inc"
#include "ui_voice.inc"
