// The models folder: other weights for the app's language model, one folder each, chosen at start (main.c).
//
// MODELS_DIR/<name>/ holds the weights file under the built-in model's own file name (qwen3.5-9b-q4s.bin for the 9B,
// the name its model.idx gives) and may also hold:
//   model.idx            the converter's index (pc/convert_9b_q4s.py --out). It must equal the package's model.idx
//                        except for the weights path: same shapes, formats, offsets and sizes as the built-in model.
//   qwen3.5-9b-mtp.bin   an MTP head of its own (the name the package's mtp.idx gives), with or without its mtp.idx
//                        (which must then equal the package's except for the path). Without one the built-in head
//                        drafts.
// A folder whose files do not match is not loaded (model_check). No console calls here: the same code runs in the
// host tests (tests/models_test.c).
#pragma once

#define MODELS_DIR "/data/garlic/models"
#define MODEL_LAST_FILE "/data/garlic_model.txt"   // the folder chosen last; empty or missing: the built-in model
#define MODELS_MAX 32                              // folders listed
#define MODEL_NAME_MAX 33                          // a folder name of up to 32 bytes, and its 0

// The weights file an index names (model.idx or mtp.idx), without its folder: e.g. "qwen3.5-9b-q4s.bin". 0 = done.
int model_idx_file(const char* idx_path, char* out, int n);
// 1 if a folder name can be a model's: 1 to 32 bytes of UTF-8, no control characters, '/' or '\', not starting
// with '.'.
int model_name_ok(const char* name);
// The folders of root that hold a file named `weights`, sorted by name (at most max). Returns how many: 0 when root
// does not exist or has none.
int models_scan(const char* root, const char* weights, char (*names)[MODEL_NAME_MAX], int max);
// Whether the folder dir can stand in for the built-in model whose index is idx_path: its model.idx, if it has one,
// equals idx_path's except for the weights path; its weights file is exactly as long as the index says; its table
// of token ids holds only real tokens. 0 = it can; -1 = it cannot, and why says why (for the log), shown which file
// and what was found (for the screen: no words, only the file name and numbers).
int model_check(const char* idx_path, const char* dir, char* why, int why_n, char* shown, int shown_n);
// The folder's MTP head against the package's mtp.idx (mtp_idx_path): 1 = it has one and it is complete; 0 = it has
// none (or the package has no head); -1 = it has one that does not match (why says why): the built-in head drafts.
int model_check_mtp(const char* mtp_idx_path, const char* dir, char* why, int why_n);
// The folder chosen last ("" for the built-in model) as the file `path` keeps it (MODEL_LAST_FILE), and the choice
// for the next start.
void model_last_get(const char* path, char* name, int n);
void model_last_set(const char* path, const char* name);
