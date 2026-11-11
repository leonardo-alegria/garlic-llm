// Garlic's reply as speech: where each piece to say ends, the text to read aloud, and its language.
#pragma once
#include <stdint.h>

// End (exclusive) of the next piece of text[from, len) to speak, or -1 if it is not complete yet. With final set
// (the reply is finished) the rest is one piece.
int st_next_piece(const char* text, int len, int from, int final);
// Bytes of text[0, n) as said: digits count as the words they become (st_next_piece's piece lengths).
int st_spoken_len(const char* text, int n);
// A natural point to cut text between words, from `from` on, between lo and hi bytes as said (st_spoken_len): the
// last space before a word that opens a phrase (que, y, pero, porque, that, and, dass, что...), else the last space
// that does not leave a word that needs what follows (an article, a preposition: "uno de los" | "primeros") at the
// end. Returns the byte after the space, or 0 if there is none.
int st_natural_cut(const char* text, int len, int from, int lo, int hi);
// The first space in [lo, hi) bytes as said before a word that opens a phrase ("el día | porque las..."), or 0.
int st_opener_cut(const char* text, int len, int from, int lo, int hi);
// Speech tokens T3 is expected to make for the words w (n bytes) in lang: 20 (silences) + a rate per character
// that depends on the language; 25 tokens = 1 s.
int st_speech_tokens(const char* w, int n, const char* lang);
// The words to read aloud from in[0, n): without markdown, code, URLs or emoji. Returns the length (0 = nothing).
int st_clean(const char* in, int n, char* out, int max);
// Language for Chatterbox ("es", "en", "pt", "fr", "de", "it", "ru", "el"): fallback unless another language is
// clearly more likely (character n-grams, speech_text.c).
const char* st_lang(const char* s, const char* fallback);
// The language of the words of text[0, n) (st_clean, then st_lang).
const char* st_text_lang(const char* text, int n, const char* fallback);
// The piece's words as Chatterbox should read them, in place (s[0, n), at most max bytes): numbers, times, dates,
// ordinals, percentages, prices and units as words of lang; abbreviations with their periods (a.C., v. Chr., н. э.)
// spelled out; all-caps acronyms letter by letter (LLM, GPU), except those said as words (NASA, OTAN). Returns the
// new length (speech_norm.c).
int st_speak(char* s, int n, int max, const char* lang);
