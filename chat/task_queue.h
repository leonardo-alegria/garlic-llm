#ifndef GARLIC_TASK_QUEUE_H
#define GARLIC_TASK_QUEUE_H

#include <stddef.h>
#include <stdint.h>

/* Small, single-process durable queue for work owned by Garlic's main loop.
 * Version 1 intentionally serializes model work: one task may be RUNNING. */
#define TASK_QUEUE_CAPACITY 32
#define TASK_PROMPT_MAX 2048
#define TASK_RESULT_MAX 4096
#define TASK_ERROR_MAX 256
#define TASK_ID_MAX 24

typedef enum {
    TASK_QUEUED = 1,
    TASK_RUNNING = 2,
    TASK_COMPLETED = 3,
    TASK_FAILED = 4,
    TASK_CANCELED = 5
} TaskState;

typedef struct {
    uint32_t id;
    uint32_t attempts;
    uint32_t state;
    uint64_t created_at;
    uint64_t updated_at;
    char prompt[TASK_PROMPT_MAX];
    char result[TASK_RESULT_MAX];
    char error[TASK_ERROR_MAX];
} GarlicTask;

typedef struct {
    char journal_path[256];
    GarlicTask tasks[TASK_QUEUE_CAPACITY];
    size_t count;
    uint32_t next_id;
    int ready;
} TaskQueue;

/* Load/replay the journal. A torn final record is discarded during recovery. */
int task_queue_init(TaskQueue *q, const char *journal_path);
/* Requeue any TASK_RUNNING tasks interrupted by a prior crash/reboot. Returns count recovered, or -1 on I/O error. */
int task_queue_recover_running(TaskQueue *q);
/* Persist and enqueue. Returns 0 or -1; id_out receives task-XXXXXXXX on success. */
int task_queue_add(TaskQueue *q, const char *prompt, char *id_out, size_t id_cap);
/* Claim the oldest queued task, persisting RUNNING before returning it; 1=claimed, 0=empty, -1=I/O error. */
int task_queue_claim(TaskQueue *q, GarlicTask *out);
/* Transition a running task to a terminal state, persisted before memory is changed. */
int task_queue_complete(TaskQueue *q, uint32_t id, const char *result);
int task_queue_fail(TaskQueue *q, uint32_t id, const char *error);
int task_queue_cancel(TaskQueue *q, uint32_t id);
/* Find by numeric ID (the suffix in task-XXXXXXXX). */
const GarlicTask *task_queue_get(const TaskQueue *q, uint32_t id);
size_t task_queue_count_state(const TaskQueue *q, TaskState state);

#endif
