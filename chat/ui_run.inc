// Python runner screens (included at the end of ui.c: uses its shapes, fonts and layouts). See ui.h.

static float cv_s = 1, cv_ox, cv_oy;
static int cv_x0, cv_y0, cv_x1 = SCREEN_W, cv_y1 = SCREEN_H, cv_w = 1, cv_h = 1;

void cv_begin(int w, int h, uint32_t bg_rgb) {
    cv_w = w > 0 ? w : 640, cv_h = h > 0 ? h : 480;
    const int ax0 = 40, ax1 = SCREEN_W - 40, ay0 = HEADER_H + 14, ay1 = SCREEN_H - FOOTER_H - 14;
    float sx = (float)(ax1 - ax0) / cv_w, sy = (float)(ay1 - ay0) / cv_h;
    cv_s = sx < sy ? sx : sy;
    cv_ox = (ax0 + ax1 - cv_w * cv_s) * 0.5f, cv_oy = (ay0 + ay1 - cv_h * cv_s) * 0.5f;
    cv_x0 = (int)floorf(cv_ox), cv_y0 = (int)floorf(cv_oy);
    cv_x1 = (int)ceilf(cv_ox + cv_w * cv_s), cv_y1 = (int)ceilf(cv_oy + cv_h * cv_s);
    draw_background();
    rrect(cv_x0 - 3, cv_y0 - 3, cv_x1 - cv_x0 + 6, cv_y1 - cv_y0 + 6, 6, C_LINE_HI, 255);   // a thin frame
    cv_clear(bg_rgb);
}

void cv_to_screen(float x, float y, float* sx, float* sy) { *sx = cv_ox + x * cv_s, *sy = cv_oy + y * cv_s; }
void cv_to_canvas(float sx, float sy, float* x, float* y) { *x = (sx - cv_ox) / cv_s, *y = (sy - cv_oy) / cv_s; }

void cv_clear(uint32_t rgb) {
    set_clip(cv_x0, cv_y0, cv_x1, cv_y1);
    fill(cv_x0, cv_y0, cv_x1 - cv_x0, cv_y1 - cv_y0, rgb & 0xFFFFFF);
    no_clip();
}

void cv_rect(float x, float y, float w, float h, int fillit, uint32_t frgb, float lw, uint32_t lrgb) {
    if (w < 0) x += w, w = -w;
    if (h < 0) y += h, h = -h;
    set_clip(cv_x0, cv_y0, cv_x1, cv_y1);
    int X0 = (int)lroundf(cv_ox + x * cv_s), Y0 = (int)lroundf(cv_oy + y * cv_s);
    int X1 = (int)lroundf(cv_ox + (x + w) * cv_s), Y1 = (int)lroundf(cv_oy + (y + h) * cv_s);
    if (X1 == X0) X1++;
    if (Y1 == Y0) Y1++;
    if (fillit) fill(X0, Y0, X1 - X0, Y1 - Y0, frgb & 0xFFFFFF);
    if (lw > 0) {
        int t = (int)lroundf(lw * cv_s);
        if (t < 1) t = 1;
        lrgb &= 0xFFFFFF;
        fill(X0, Y0, X1 - X0, t, lrgb), fill(X0, Y1 - t, X1 - X0, t, lrgb);
        fill(X0, Y0, t, Y1 - Y0, lrgb), fill(X1 - t, Y0, t, Y1 - Y0, lrgb);
    }
    no_clip();
}

void cv_oval(float x, float y, float w, float h, int fillit, uint32_t frgb, float lw, uint32_t lrgb) {
    if (w < 0) x += w, w = -w;
    if (h < 0) y += h, h = -h;
    set_clip(cv_x0, cv_y0, cv_x1, cv_y1);
    float cx = cv_ox + (x + w / 2) * cv_s, cy = cv_oy + (y + h / 2) * cv_s, rx = w / 2 * cv_s, ry = h / 2 * cv_s;
    float t = lw > 0 ? lw * cv_s : 0;
    if (t > 0 && t < 1) t = 1;
    if (fabsf(rx - ry) < 0.5f) {
        if (fillit) circle(cx, cy, rx, 0, frgb & 0xFFFFFF, 255);
        if (t > 0) circle(cx, cy, rx, t, lrgb & 0xFFFFFF, 255);
    } else if (rx > 0 && ry > 0) {
        int bx0 = (int)floorf(cx - rx - 1), bx1 = (int)ceilf(cx + rx + 1), by0 = (int)floorf(cy - ry - 1), by1 = (int)ceilf(cy + ry + 1);
        if (bx0 < clip_x0) bx0 = clip_x0;
        if (by0 < clip_y0) by0 = clip_y0;
        if (bx1 > clip_x1) bx1 = clip_x1;
        if (by1 > clip_y1) by1 = clip_y1;
        float rmin = rx < ry ? rx : ry;
        for (int py = by0; py < by1; py++)
            for (int px = bx0; px < bx1; px++) {
                float dx = (px + 0.5f - cx) / rx, dy = (py + 0.5f - cy) / ry;
                float d = (sqrtf(dx * dx + dy * dy) - 1) * rmin;   // approximate distance in pixels
                if (fillit) {
                    float c = clamp01(0.5f - d);
                    if (c > 0.004f) blend(back + py * SCREEN_W + px, frgb & 0xFFFFFF, (uint32_t)(c * 255));
                }
                if (t > 0) {
                    float c = clamp01(0.5f - d) - clamp01(0.5f - (d + t));
                    if (c > 0.004f) blend(back + py * SCREEN_W + px, lrgb & 0xFFFFFF, (uint32_t)(c * 255));
                }
            }
    }
    no_clip();
}

void cv_line(float x1, float y1, float x2, float y2, float lw, uint32_t rgb) {
    set_clip(cv_x0, cv_y0, cv_x1, cv_y1);
    float w = lw * cv_s;
    if (w < 1) w = 1;
    line(cv_ox + x1 * cv_s, cv_oy + y1 * cv_s, cv_ox + x2 * cv_s, cv_oy + y2 * cv_s, w, rgb & 0xFFFFFF, 255);
    no_clip();
}

void cv_polygon(const float* xy, int n, int fillit, uint32_t frgb, float lw, uint32_t lrgb) {
    if (n < 2) return;
    set_clip(cv_x0, cv_y0, cv_x1, cv_y1);
    if (fillit && n >= 3) {   // even-odd scanline fill at screen resolution
        float ymin = 1e9f, ymax = -1e9f;
        for (int i = 0; i < n; i++) {
            float sy = cv_oy + xy[2 * i + 1] * cv_s;
            if (sy < ymin) ymin = sy;
            if (sy > ymax) ymax = sy;
        }
        int y0 = (int)floorf(ymin), y1 = (int)ceilf(ymax);
        if (y0 < clip_y0) y0 = clip_y0;
        if (y1 > clip_y1) y1 = clip_y1;
        static float xs[512];
        for (int py = y0; py < y1; py++) {
            float fy = py + 0.5f;
            int m = 0;
            for (int i = 0; i < n && m < 512; i++) {
                float ax = cv_ox + xy[2 * i] * cv_s, ay = cv_oy + xy[2 * i + 1] * cv_s;
                int j = (i + 1) % n;
                float bx = cv_ox + xy[2 * j] * cv_s, by = cv_oy + xy[2 * j + 1] * cv_s;
                if ((ay <= fy && by > fy) || (by <= fy && ay > fy)) xs[m++] = ax + (fy - ay) / (by - ay) * (bx - ax);
            }
            for (int a = 1; a < m; a++)   // insertion sort: few crossings
                for (int b = a; b > 0 && xs[b - 1] > xs[b]; b--) {
                    float tmp = xs[b];
                    xs[b] = xs[b - 1], xs[b - 1] = tmp;
                }
            for (int k = 0; k + 1 < m; k += 2) {
                int a = (int)lroundf(xs[k]), b = (int)lroundf(xs[k + 1]);
                if (b > a) fill(a, py, b - a, 1, frgb & 0xFFFFFF);
            }
        }
        if (lw <= 0)   // a soft 1-pixel edge in the same colour smooths the outline
            for (int i = 0; i < n; i++) {
                int j = (i + 1) % n;
                line(cv_ox + xy[2 * i] * cv_s, cv_oy + xy[2 * i + 1] * cv_s, cv_ox + xy[2 * j] * cv_s,
                     cv_oy + xy[2 * j + 1] * cv_s, 1.0f, frgb & 0xFFFFFF, 160);
            }
    }
    if (lw > 0) {
        float w = lw * cv_s < 1 ? 1 : lw * cv_s;
        int segs = fillit || n > 2 ? n : n - 1;
        for (int i = 0; i < segs; i++) {
            int j = (i + 1) % n;
            line(cv_ox + xy[2 * i] * cv_s, cv_oy + xy[2 * i + 1] * cv_s, cv_ox + xy[2 * j] * cv_s,
                 cv_oy + xy[2 * j + 1] * cv_s, w, lrgb & 0xFFFFFF, 255);
        }
    }
    no_clip();
}

// Text at any size: glyphs of the nearest font, resampled (bilinear) when the size differs.
static const Font* pick_font(float px, int bold, float* scale) {
    const Font* f = bold ? (px > 34 ? &font_big : &font_bold) : (px < 27 ? &font_small : px > 34 ? &font_big : &font);
    int base = f == &font_small ? 24 : f == &font_big ? 40 : 30;
    *scale = px / base;
    return f;
}

static void draw_text_scaled(const Font* f, float x, float y, const char* s, uint32_t rgb, float scale) {
    if (fabsf(scale - 1) < 0.06f) {
        draw_str(f, (int)lroundf(x), (int)lroundf(y), s, rgb);
        return;
    }
    const char* e = s + strlen(s);
    Glyph g;
    while (next_glyph(f, &s, e, &g)) {
        float gx = x + g.bx * scale, gy = y + g.by * scale;
        int W = (int)ceilf(g.w * scale), H = (int)ceilf(g.h * scale);
        for (int j = 0; j < H; j++) {
            int py = (int)gy + j;
            if (py < clip_y0 || py >= clip_y1) continue;
            float v = (j + 0.5f) / scale - 0.5f;
            int v0 = (int)floorf(v);
            float fv = v - v0;
            for (int i = 0; i < W; i++) {
                int px = (int)gx + i;
                if (px < clip_x0 || px >= clip_x1) continue;
                float u = (i + 0.5f) / scale - 0.5f;
                int u0 = (int)floorf(u);
                float fu = u - u0;
                float r = 0, gg = 0, b = 0, a = 0;
                for (int k = 0; k < 4; k++) {
                    int uu = u0 + (k & 1), vv = v0 + (k >> 1);
                    if (uu < 0 || vv < 0 || uu >= g.w || vv >= g.h) continue;
                    float wgt = ((k & 1) ? fu : 1 - fu) * ((k >> 1) ? fv : 1 - fv);
                    if (g.kind == 1) {
                        const uint8_t* c = g.bm + (vv * g.w + uu) * 4;
                        r += c[0] * wgt * c[3], gg += c[1] * wgt * c[3], b += c[2] * wgt * c[3], a += c[3] * wgt;
                    } else a += g.bm[vv * g.w + uu] * wgt;
                }
                if (a < 2) continue;
                uint32_t* d = back + py * SCREEN_W + px;
                if (g.kind == 1) blend(d, (uint32_t)(r / a) << 16 | (uint32_t)(gg / a) << 8 | (uint32_t)(b / a), (uint32_t)a);
                else blend(d, rgb, (uint32_t)a);
            }
        }
        x += g.adv * scale;
    }
}

void cv_text_size(const char* s, float size, int bold, float* w, float* h) {
    float scale;
    const Font* f = pick_font(size * cv_s, bold, &scale);
    *w = str_width(f, s) * scale / cv_s;
    *h = f->line * scale / cv_s;
}

void cv_text(float x, float y, const char* s, float size, int bold, uint32_t rgb, int anchor) {
    float scale;
    const Font* f = pick_font(size * cv_s, bold, &scale);
    float w = str_width(f, s) * scale, h = f->line * scale;
    float sx = cv_ox + x * cv_s, sy = cv_oy + y * cv_s;
    int ax = anchor % 3, ay = anchor / 3;
    sx -= ax == 1 ? w / 2 : ax == 2 ? w : 0;
    sy -= ay == 1 ? h / 2 : ay == 2 ? h : 0;
    set_clip(cv_x0, cv_y0, cv_x1, cv_y1);
    draw_text_scaled(f, sx, sy, s, rgb & 0xFFFFFF, scale);
    no_clip();
}

static void run_header(const char* title) {
    memcpy(back, bg, HEADER_H * SCREEN_W * 4);
    draw_image(&logo_s, 64, (HEADER_H - logo_s.h) / 2 - 2, 255);
    int x = 64 + (logo_s.w ? logo_s.w + 14 : 0), cy = HEADER_H / 2;
    int w = 46 + str_width(&font_bold, "Python") + 20;
    rrect(x, cy - 22, w, 44, 22, C_ACCENT, 255);
    float tx = x + 20.0f;
    float tri[6];
    tri[0] = tx, tri[1] = cy - 9.0f, tri[2] = tx, tri[3] = cy + 9.0f, tri[4] = tx + 14, tri[5] = (float)cy;
    float ox = cv_ox, oy = cv_oy, s = cv_s;
    int c0 = cv_x0, c1 = cv_y0, c2 = cv_x1, c3 = cv_y1;
    cv_ox = 0, cv_oy = 0, cv_s = 1, cv_x0 = 0, cv_y0 = 0, cv_x1 = SCREEN_W, cv_y1 = SCREEN_H;
    cv_polygon(tri, 3, 1, 0xFFFFFF, 0, 0);
    cv_ox = ox, cv_oy = oy, cv_s = s, cv_x0 = c0, cv_y0 = c1, cv_x1 = c2, cv_y1 = c3;
    draw_str(&font_bold, x + 46, cy - font_bold.line / 2, "Python", 0xFFFFFF);
    if (title && title[0]) draw_str(&font, x + w + 20, cy - font.line / 2, title, C_TEXT_DIM);
}

// Footer with the program's own controls ("✕ Space  □ R ..."), then ○ to leave.
void draw_run_chrome(const char* title, int mouse, const int* btns, const char* const* labels, int n) {
    run_header(title);
    const int y = SCREEN_H - FOOTER_H, cy = y + FOOTER_H / 2;
    memcpy(back + y * SCREEN_W, bg + y * SCREEN_W, FOOTER_H * SCREEN_W * 4);
    int x = 72;
    if (mouse) {
        int w = draw_btn(BTN_CROSS, x, cy);
        x = draw_str(&font_small, x + w + 10, cy - font_small.line / 2, tr(S_PY_CLICK), C_TEXT_DIM) + 36;
        w = draw_btn(BTN_SQUARE, x, cy);
        x = draw_str(&font_small, x + w + 10, cy - font_small.line / 2, tr(S_PY_RCLICK), C_TEXT_DIM) + 36;
    }
    for (int i = 0; i < n && x < SCREEN_W - 400; i++) {
        int w = draw_btn(btns[i], x, cy);
        x = draw_str(&font_small, x + w + 10, cy - font_small.line / 2, labels[i], C_TEXT_DIM) + 36;
    }
    int w = draw_btn(BTN_CIRCLE, x, cy);
    draw_str(&font_small, x + w + 10, cy - font_small.line / 2, tr(S_PY_EXIT), C_TEXT_DIM);
}

void cv_fill_alpha(float x, float y, float w, float h, uint32_t rgb, int alpha) {
    if (w < 0) x += w, w = -w;
    if (h < 0) y += h, h = -h;
    set_clip(cv_x0, cv_y0, cv_x1, cv_y1);
    int X0 = (int)lroundf(cv_ox + x * cv_s), Y0 = (int)lroundf(cv_oy + y * cv_s);
    int X1 = (int)lroundf(cv_ox + (x + w) * cv_s), Y1 = (int)lroundf(cv_oy + (y + h) * cv_s);
    fill_a(X0, Y0, X1 - X0, Y1 - Y0, rgb & 0xFFFFFF, alpha);
    no_clip();
}

static uint32_t cursor_save[40 * 40];
static int cursor_x = -1000, cursor_y;

void cursor_draw(float sx, float sy) {
    cursor_x = (int)sx - 4, cursor_y = (int)sy - 4;
    for (int j = 0; j < 40; j++)
        for (int i = 0; i < 40; i++) {
            int px = cursor_x + i, py = cursor_y + j;
            cursor_save[j * 40 + i] = px >= 0 && py >= 0 && px < SCREEN_W && py < SCREEN_H ? back[py * SCREEN_W + px] : 0;
        }
    float xy[6] = {sx, sy, sx, sy + 26, sx + 18, sy + 19};   // an arrow, in screen coordinates
    float ox = cv_ox, oy = cv_oy, s = cv_s;
    int c0 = cv_x0, c1 = cv_y0, c2 = cv_x1, c3 = cv_y1;
    cv_ox = 0, cv_oy = 0, cv_s = 1, cv_x0 = 0, cv_y0 = 0, cv_x1 = SCREEN_W, cv_y1 = SCREEN_H;
    cv_polygon(xy, 3, 1, 0xFFFFFF, 2.0f, 0x0A1630);
    cv_ox = ox, cv_oy = oy, cv_s = s, cv_x0 = c0, cv_y0 = c1, cv_x1 = c2, cv_y1 = c3;
}

void cursor_undraw(void) {
    if (cursor_x < -100) return;
    for (int j = 0; j < 40; j++)
        for (int i = 0; i < 40; i++) {
            int px = cursor_x + i, py = cursor_y + j;
            if (px >= 0 && py >= 0 && px < SCREEN_W && py < SCREEN_H) back[py * SCREEN_W + px] = cursor_save[j * 40 + i];
        }
    cursor_x = -1000;
}

void draw_console(const char* text, size_t len, const char* title, int bottom, int scroll) {
    draw_background();
    run_header(title);
    if (bottom > SCREEN_H - FOOTER_H) bottom = SCREEN_H - FOOTER_H;
    const Hint hh[] = {{BTN_CIRCLE, S_PY_EXIT}};
    draw_footer(hh, 1, NULL);
    int x = COL_X - 60, y = HEADER_H + 24, w = COL_W + 120, h = bottom - y - 24;
    rrect(x, y, w, h, 22, 0x050E22, 255);
    rrect_line(x, y, w, h, 22, 1, C_LINE_HI, 255);
    int pad = 28, maxw = w - 2 * pad, lh = mono.line + 4, rows = (h - 2 * pad) / lh;
    static const char* ls[4096];
    static const char* le[4096];
    int n = 0;
    const char* end = text + len;
    for (const char* s = text; s < end && n < 4096;) {   // wrap into screen lines in the code font
        const char* nl = memchr(s, '\n', end - s);
        const char* e = nl ? nl : end;
        const char* a = s;
        do {
            const char* b = a;
            int wsum = 0;
            while (b < e) {
                const char* nx = b;
                utf8_next(&nx);
                int cw = text_width(&mono, b, nx);
                if (wsum + cw > maxw && b > a) break;
                wsum += cw, b = nx;
            }
            ls[n] = a, le[n] = b, n++;
            a = b;
        } while (a < e && n < 4096);
        s = nl ? nl + 1 : end;
    }
    int first = n - rows - scroll;
    if (first < 0) first = 0;
    set_clip(x, y + 8, x + w, y + h - 8);
    for (int i = first, row = 0; i < n && row < rows; i++, row++)
        draw_text(&mono, x + pad, y + pad + row * lh, ls[i], le[i], 0xD7E6FA);
    no_clip();
}

void draw_run_end(const char* title, const char* error, int stopped) {
    (void)title;
    int has_error = error && error[0];
    int w = 1200, x = (SCREEN_W - w) / 2;
    static Layout* L;
    Layout* l = has_error ? layout_get(&L, error, strlen(error), w - 80, &PAL_FIELD, 1) : NULL;
    int text_h = l ? (l->height > 6 * TEXT_LINE_H ? 6 * TEXT_LINE_H : l->height) : 0;
    int h = 104 + (l ? text_h + 24 : 0), y = SCREEN_H - FOOTER_H - 30 - h;
    ui_dim();
    rrect(x, y, w, h, 28, C_PANEL, 255);
    rrect_line(x, y, w, h, 28, 2, has_error ? C_RED : C_ACCENT_HI, 170);
    const char* head = has_error ? tr(S_PY_ERROR) : stopped ? tr(S_PY_STOPPED) : tr(S_PY_DONE);
    draw_str(&font_big, x + 40, y + 28, head, has_error ? 0xFF9AA2 : 0xFFFFFF);
    if (l) {
        set_clip(x + 40, y + 96, x + w - 40, y + 96 + text_h + 4);
        draw_layout(l, x + 40, y + 96 + (l->height > text_h ? text_h - l->height : 0));
        no_clip();
    }
    Hint hh[3];
    int n = 0;
    hh[n++] = (Hint){BTN_CROSS, S_PY_BACK};
    hh[n++] = (Hint){BTN_TRIANGLE, S_PY_AGAIN};
    if (has_error) hh[n++] = (Hint){BTN_SQUARE, S_PY_FIX};
    draw_footer(hh, n, NULL);
}

void draw_run_intro(const char* title, float t) {
    float e = 1 - (1 - t) * (1 - t);   // ease out
    char s[160];
    snprintf(s, sizeof(s), "%s %s…", tr(S_PY_STARTING), title && title[0] ? title : "Python");
    int w = (int)((str_width(&font_big, s) + 200) * (0.85f + 0.15f * e)), h = 150;
    int x = (SCREEN_W - w) / 2, y = (SCREEN_H - h) / 2 + (int)(30 * (1 - e));
    text_alpha = (int)(255 * (t < 0.3f ? t / 0.3f : 1));
    rrect(x, y, w, h, 30, C_PANEL, 255);
    rrect_line(x, y, w, h, 30, 2, C_ACCENT_HI, 200);
    circle(x + 75, y + h / 2, 34, 0, C_ACCENT, 255);
    float tri[6] = {x + 66.0f, y + h / 2 - 14.0f, x + 66.0f, y + h / 2 + 14.0f, x + 90.0f, (float)(y + h / 2)};
    float ox = cv_ox, oy = cv_oy, sc = cv_s;
    int c0 = cv_x0, c1 = cv_y0, c2 = cv_x1, c3 = cv_y1;
    cv_ox = 0, cv_oy = 0, cv_s = 1, cv_x0 = 0, cv_y0 = 0, cv_x1 = SCREEN_W, cv_y1 = SCREEN_H;
    cv_polygon(tri, 3, 1, 0xFFFFFF, 0, 0);
    cv_ox = ox, cv_oy = oy, cv_s = sc, cv_x0 = c0, cv_y0 = c1, cv_x1 = c2, cv_y1 = c3;
    draw_str(&font_big, x + 130, y + (h - font_big.line) / 2, s, 0xFFFFFF);
    text_alpha = 255;
}
