// time.time() for extmod/modtime.c (MICROPY_PY_TIME_INCLUDEFILE): seconds as a float.
static mp_obj_t mp_time_time_get(void) { return mp_obj_new_float((mp_float_t)(mp_hal_time_ns() / 1e9)); }
