// Voice mode screen (included at the end of ui.c). A big orb in the middle that breathes when idle, follows your
// voice while you talk, turns while Garlic thinks and pulses while it speaks; under it, only Garlic's words.

void draw_voice_mode(int state, const char* reply, float level, double time) {
    draw_background();
    // header: logo, name, "voice mode" chip
    draw_image(&logo_s, 64, (HEADER_H - logo_s.h) / 2 - 2, 255);
    int tx = 64 + (logo_s.w ? logo_s.w + 14 : 0);
    tx = draw_str(&font_big, tx, (HEADER_H - font_big.line) / 2, "Garlic", 0xFFFFFF);
    tx = draw_str(&font_big, tx + font_big.space, (HEADER_H - font_big.line) / 2, "LLM", C_ACCENT_HI);
    chip(SCREEN_W - 64, HEADER_H / 2, tr(S_VOICE_MODE), 0xFFFFFF, C_ACCENT_HI, 1.0f, -1);

    float cx = SCREEN_W / 2.0f, cy = 360, t = (float)time;
    float base = 118;
    // rings
    if (state == VM_IDLE) {
        float b = 0.5f + 0.5f * sinf(t * 1.6f);
        circle(cx, cy, base + 26 + 8 * b, 0, C_ACCENT, (int)(26 + 20 * b));
        circle(cx, cy, base + 12, 0, C_ACCENT_HI, 34);
    } else if (state == VM_LISTEN || state == VM_SPEAK) {
        float l = level < 0 ? 0 : level > 1 ? 1 : level;
        for (int k = 3; k >= 1; k--) {
            float ph = t * (state == VM_SPEAK ? 5.0f : 3.0f) - k * 0.9f;
            float r = base + 8 + k * (11 + 20 * l) + 4 * sinf(ph);
            circle(cx, cy, r, 0, state == VM_SPEAK ? C_ACCENT_HI : C_GREEN, 22 + 16 * (4 - k));
        }
    } else {   // thinking: dots going round
        for (int k = 0; k < 16; k++) {
            float a = (float)k / 16 * 6.2831853f + t * 2.6f;
            float fade = (float)k / 16;
            circle(cx + (base + 34) * cosf(a), cy + (base + 34) * sinf(a), 7, 0, C_ACCENT_HI, (int)(30 + 200 * fade));
        }
        circle(cx, cy, base + 12, 0, C_ACCENT, 40);
    }
    circle(cx, cy, base, 0, 0x0E2145, 255);
    circle(cx, cy, base, 2, state == VM_LISTEN ? C_GREEN : C_ACCENT_HI, 200);
    draw_image(&logo_l, (int)cx - logo_l.w / 2, (int)cy - logo_l.h / 2 - 4, 255);

    // state line
    const char* line_ = state == VM_LISTEN ? tr(S_LISTENING_SHORT) : state == VM_THINK ? tr(S_THINKING) : NULL;
    if (state == VM_IDLE) {
        const char* s = tr(S_TALK_HOLD);
        int w = 34 + 12 + str_width(&font, s);
        int x = (SCREEN_W - w) / 2;
        int bw = draw_btn(BTN_SQUARE, x, (int)cy + (int)base + 62);
        draw_str(&font, x + bw + 12, (int)cy + (int)base + 62 - font.line / 2, s, C_TEXT_DIM);
    } else if (line_) {
        draw_center(&font, (int)cx, (int)cy + (int)base + 62 - font.line / 2, line_, C_TEXT_DIM);
    }

    // Garlic's words: the end of the reply, in a calm column
    if (reply && reply[0]) {
        static Layout* L;
        const int w = 1400, x = (SCREEN_W - w) / 2, top = (int)(cy + base + 120), bottom = SCREEN_H - FOOTER_H - 24;
        // Only the end shows (~9 lines): lay out the last ~2 KB from a line start, not the whole reply, so that
        // the cost of each new token does not grow with the length of the reply (laid out whole, a long reply
        // lowers the frame rate). A cut inside a ``` block opens it again, so the code still reads as code.
        static char tail[2048 + 8];
        size_t n = strlen(reply);
        const char* s = reply;
        if (n > 2048) {
            const char* p = reply + n - 2048;
            const char* nl = memchr(p, '\n', reply + n - p);
            if (nl && nl + 1 < reply + n) p = nl + 1;
            else while ((*p & 0xC0) == 0x80) p++;   // (no line break: at a character)
            int fences = 0;
            for (const char* q = reply; q + 2 < p; q++)
                if ((q == reply || q[-1] == '\n') && q[0] == '`' && q[1] == '`' && q[2] == '`') fences++, q += 2;
            size_t k = 0, m = reply + n - p;
            if (fences & 1) memcpy(tail, "```\n", 4), k = 4;
            memcpy(tail + k, p, m);
            tail[k + m] = 0;
            s = tail, n = k + m;
        }
        Layout* l = layout_get(&L, s, n, w, &PAL_AI, 0);
        int view = bottom - top;
        int oy = top + (l->height <= view ? 0 : view - l->height);
        set_clip(0, top, SCREEN_W, bottom);
        draw_layout(l, x, oy);
        no_clip();
        if (l->height > view) fade_rows(top, top + 60, 0, 255);
    }
    const Hint hh[] = {{BTN_SQUARE, S_TALK_HOLD}, {BTN_TRIANGLE, S_NEW_CHAT}, {BTN_CIRCLE, S_PY_EXIT}};
    draw_footer(hh, 3, NULL);
}
