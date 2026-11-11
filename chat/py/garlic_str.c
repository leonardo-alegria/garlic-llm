// str methods CPython has and MicroPython leaves out, common in the programs Garlic writes: capitalize, title,
// swapcase, casefold, ljust, rjust, zfill, isalnum, isnumeric, isdecimal, istitle, removeprefix, removesuffix,
// expandtabs. They are added to the str method table (objstr.c) through garlic_str_*.h.
// Letters outside ASCII keep their case, as in MicroPython's own lower() and upper().
#include <string.h>

#include "py/objstr.h"
#include "py/runtime.h"

static mp_obj_t map_case(mp_obj_t self, int mode) {   // 0 capitalize, 1 title, 2 swapcase, 3 casefold
    GET_STR_DATA_LEN(self, s, n);
    vstr_t v;
    vstr_init_len(&v, n);
    int start = 1;
    for (size_t i = 0; i < n; i++) {
        byte c = s[i];
        byte o = c;
        if (c < 128) {
            if (mode == 0) o = i == 0 ? unichar_toupper(c) : unichar_tolower(c);
            else if (mode == 1) o = start ? unichar_toupper(c) : unichar_tolower(c);
            else if (mode == 2) o = unichar_isupper(c) ? unichar_tolower(c) : unichar_toupper(c);
            else o = unichar_tolower(c);
            start = !unichar_isalpha(c);
        } else if ((c & 0xC0) != 0x80) {
            start = 0;   // a non-ASCII letter starts a word too
        }
        v.buf[i] = (char)o;
    }
    return mp_obj_new_str_from_utf8_vstr(&v);
}
static mp_obj_t str_capitalize(mp_obj_t self) { return map_case(self, 0); }
static mp_obj_t str_title(mp_obj_t self) { return map_case(self, 1); }
static mp_obj_t str_swapcase(mp_obj_t self) { return map_case(self, 2); }
static mp_obj_t str_casefold(mp_obj_t self) { return map_case(self, 3); }
MP_DEFINE_CONST_FUN_OBJ_1(garlic_str_capitalize_obj, str_capitalize);
MP_DEFINE_CONST_FUN_OBJ_1(garlic_str_title_obj, str_title);
MP_DEFINE_CONST_FUN_OBJ_1(garlic_str_swapcase_obj, str_swapcase);
MP_DEFINE_CONST_FUN_OBJ_1(garlic_str_casefold_obj, str_casefold);

static size_t n_chars(const byte* s, size_t n) {
    size_t k = 0;
    for (size_t i = 0; i < n; i++) k += (s[i] & 0xC0) != 0x80;
    return k;
}

static mp_obj_t justify(size_t n_args, const mp_obj_t* args, int right) {
    GET_STR_DATA_LEN(args[0], s, n);
    mp_int_t width = mp_obj_get_int(args[1]);
    const char* fill = " ";
    size_t fill_len = 1;
    if (n_args > 2) fill = mp_obj_str_get_data(args[2], &fill_len);
    size_t have = n_chars(s, n);
    if (width <= (mp_int_t)have) return args[0];
    size_t pad = width - have;
    vstr_t v;
    vstr_init(&v, n + pad * fill_len);
    if (right)
        for (size_t i = 0; i < pad; i++) vstr_add_strn(&v, fill, fill_len);
    vstr_add_strn(&v, (const char*)s, n);
    if (!right)
        for (size_t i = 0; i < pad; i++) vstr_add_strn(&v, fill, fill_len);
    return mp_obj_new_str_from_utf8_vstr(&v);
}
static mp_obj_t str_ljust(size_t n, const mp_obj_t* a) { return justify(n, a, 0); }
static mp_obj_t str_rjust(size_t n, const mp_obj_t* a) { return justify(n, a, 1); }
MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(garlic_str_ljust_obj, 2, 3, str_ljust);
MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(garlic_str_rjust_obj, 2, 3, str_rjust);

static mp_obj_t str_zfill(mp_obj_t self, mp_obj_t width_in) {
    GET_STR_DATA_LEN(self, s, n);
    mp_int_t width = mp_obj_get_int(width_in);
    size_t have = n_chars(s, n);
    if (width <= (mp_int_t)have) return self;
    size_t pad = width - have, sign = n > 0 && (s[0] == '-' || s[0] == '+');
    vstr_t v;
    vstr_init(&v, n + pad);
    if (sign) vstr_add_byte(&v, s[0]);
    for (size_t i = 0; i < pad; i++) vstr_add_byte(&v, '0');
    vstr_add_strn(&v, (const char*)s + sign, n - sign);
    return mp_obj_new_str_from_utf8_vstr(&v);
}
MP_DEFINE_CONST_FUN_OBJ_2(garlic_str_zfill_obj, str_zfill);

static mp_obj_t test_chars(mp_obj_t self, int mode) {   // 0 alnum, 1 numeric / decimal, 2 title
    GET_STR_DATA_LEN(self, s, n);
    if (n == 0) return mp_const_false;
    int prev_cased = 0, any_cased = 0;
    for (size_t i = 0; i < n; i++) {
        byte c = s[i];
        if (mode == 0 && !(c >= 128 || unichar_isalpha(c) || unichar_isdigit(c))) return mp_const_false;
        if (mode == 1 && !unichar_isdigit(c)) return mp_const_false;
        if (mode == 2 && c < 128) {
            if (unichar_isupper(c)) {
                if (prev_cased) return mp_const_false;
                prev_cased = any_cased = 1;
            } else if (unichar_islower(c)) {
                if (!prev_cased) return mp_const_false;
                prev_cased = any_cased = 1;
            } else prev_cased = 0;
        }
    }
    return mode == 2 && !any_cased ? mp_const_false : mp_const_true;
}
static mp_obj_t str_isalnum(mp_obj_t self) { return test_chars(self, 0); }
static mp_obj_t str_isnumeric(mp_obj_t self) { return test_chars(self, 1); }
static mp_obj_t str_istitle(mp_obj_t self) { return test_chars(self, 2); }
MP_DEFINE_CONST_FUN_OBJ_1(garlic_str_isalnum_obj, str_isalnum);
MP_DEFINE_CONST_FUN_OBJ_1(garlic_str_isnumeric_obj, str_isnumeric);
MP_DEFINE_CONST_FUN_OBJ_1(garlic_str_istitle_obj, str_istitle);

static mp_obj_t str_removefix(mp_obj_t self, mp_obj_t fix_in, int suffix) {
    GET_STR_DATA_LEN(self, s, n);
    size_t m;
    const char* fix = mp_obj_str_get_data(fix_in, &m);
    if (m > n) return self;
    if (!suffix && !memcmp(s, fix, m)) return mp_obj_new_str((const char*)s + m, n - m);
    if (suffix && !memcmp(s + n - m, fix, m)) return mp_obj_new_str((const char*)s, n - m);
    return self;
}
static mp_obj_t str_removeprefix(mp_obj_t a, mp_obj_t b) { return str_removefix(a, b, 0); }
static mp_obj_t str_removesuffix(mp_obj_t a, mp_obj_t b) { return str_removefix(a, b, 1); }
MP_DEFINE_CONST_FUN_OBJ_2(garlic_str_removeprefix_obj, str_removeprefix);
MP_DEFINE_CONST_FUN_OBJ_2(garlic_str_removesuffix_obj, str_removesuffix);

static mp_obj_t str_expandtabs(size_t n_args, const mp_obj_t* args) {
    GET_STR_DATA_LEN(args[0], s, n);
    mp_int_t tab = n_args > 1 ? mp_obj_get_int(args[1]) : 8;
    vstr_t v;
    vstr_init(&v, n);
    size_t col = 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\t' && tab > 0) {
            do vstr_add_byte(&v, ' '), col++;
            while (col % tab);
        } else {
            vstr_add_byte(&v, s[i]);
            if (s[i] == '\n') col = 0;
            else if ((s[i] & 0xC0) != 0x80) col++;
        }
    }
    return mp_obj_new_str_from_utf8_vstr(&v);
}
MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(garlic_str_expandtabs_obj, 1, 2, str_expandtabs);
