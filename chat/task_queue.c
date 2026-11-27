#include "task_queue.h"

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <io.h>
#define tq_fsync(fd) _commit(fd)
#define tq_ftruncate(fd, len) _chsize_s((fd), (__int64)(len))
static ssize_t tq_pread(int fd, void *buf, size_t n, off_t off) {
    if (_lseeki64(fd, (__int64)off, SEEK_SET) < 0) return -1;
    return _read(fd, buf, (unsigned)n);
}
#else
#include <unistd.h>
#define O_BINARY 0
#define tq_fsync(fd) fsync(fd)
#define tq_ftruncate(fd, len) ftruncate((fd), (len))
#define tq_pread(fd, buf, n, off) pread((fd), (buf), (n), (off))
#endif

#define JOURNAL_MAGIC 0x47545131u /* GTQ1 */
#define JOURNAL_VERSION 1u

typedef struct {
    uint32_t magic;
    uint32_t version;
    GarlicTask task;
    uint32_t checksum;
} JournalRecord;

static uint32_t checksum_bytes(const unsigned char *p, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 16777619u;
    return h;
}

static uint64_t task_now(void) {
    time_t t = time(NULL);
    return t < 0 ? 0 : (uint64_t)t;
}

static int record_valid(const JournalRecord *r) {
    return r->magic == JOURNAL_MAGIC && r->version == JOURNAL_VERSION &&
           r->task.id != 0 && r->task.state >= TASK_QUEUED && r->task.state <= TASK_CANCELED &&
           r->checksum == checksum_bytes((const unsigned char *)r, offsetof(JournalRecord, checksum));
}

static int write_all(int fd, const void *buf, size_t n) {
    const unsigned char *p = (const unsigned char *)buf;
    while (n) {
        ssize_t w = write(fd, p, n);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return -1;
        p += (size_t)w;
        n -= (size_t)w;
    }
    return 0;
}

static int read_all_at(int fd, void *buf, size_t n, off_t off) {
    unsigned char *p = (unsigned char *)buf;
    size_t done = 0;
    while (done < n) {
        ssize_t r = tq_pread(fd, p + done, n - done, off + (off_t)done);
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) return -1;
        if (r == 0) break;
        done += (size_t)r;
    }
    return (int)done;
}

static int persist(TaskQueue *q, const GarlicTask *task) {
    JournalRecord r;
    memset(&r, 0, sizeof(r));
    r.magic = JOURNAL_MAGIC;
    r.version = JOURNAL_VERSION;
    r.task = *task;
    r.checksum = checksum_bytes((const unsigned char *)&r, offsetof(JournalRecord, checksum));
    int fd = open(q->journal_path, O_WRONLY | O_APPEND | O_BINARY);
    if (fd < 0) return -1;
    int ok = write_all(fd, &r, sizeof(r)) == 0 && tq_fsync(fd) == 0;
    int saved = errno;
    close(fd);
    errno = saved;
    return ok ? 0 : -1;
}

static int task_index(const TaskQueue *q, uint32_t id) {
    for (size_t i = 0; i < q->count; i++) if (q->tasks[i].id == id) return (int)i;
    return -1;
}

static int replace_task(TaskQueue *q, const GarlicTask *task) {
    int i = task_index(q, task->id);
    if (i < 0) {
        if (q->count >= TASK_QUEUE_CAPACITY) return -1;
        i = (int)q->count++;
    }
    q->tasks[i] = *task;
    if (task->id >= q->next_id) q->next_id = task->id + 1;
    return 0;
}

int task_queue_init(TaskQueue *q, const char *path) {
    if (!q || !path || !*path || strlen(path) >= sizeof(q->journal_path)) return -1;
    memset(q, 0, sizeof(*q));
    strcpy(q->journal_path, path);
    q->next_id = 1;
    int fd = open(path, O_RDWR | O_CREAT | O_BINARY, 0666);
    if (fd < 0) return -1;
    off_t off = 0;
    for (;;) {
        JournalRecord r;
        int n = read_all_at(fd, &r, sizeof(r), off);
        if (n == 0) break;
        if (n < 0) { close(fd); return -1; }
        if ((size_t)n != sizeof(r) || !record_valid(&r)) {
            /* Only retain the verified prefix; this also removes a torn/corrupt tail. */
            if (tq_ftruncate(fd, off) != 0 || tq_fsync(fd) != 0) { close(fd); return -1; }
            break;
        }
        if (replace_task(q, &r.task) != 0) { close(fd); return -1; }
        off += (off_t)sizeof(r);
    }
    close(fd);
    q->ready = 1;
    return 0;
}

int task_queue_recover_running(TaskQueue *q) {
    if (!q || !q->ready) return -1;
    int recovered = 0;
    for (size_t i = 0; i < q->count; i++) {
        if (q->tasks[i].state != TASK_RUNNING) continue;
        GarlicTask t = q->tasks[i];
        t.state = TASK_QUEUED;
        t.updated_at = task_now();
        if (persist(q, &t) != 0) return -1;
        q->tasks[i] = t;
        recovered++;
    }
    return recovered;
}

int task_queue_add(TaskQueue *q, const char *prompt, char *id_out, size_t id_cap) {
    if (!q || !q->ready || !prompt || !*prompt || strlen(prompt) >= TASK_PROMPT_MAX ||
        q->count >= TASK_QUEUE_CAPACITY || !id_out || id_cap < TASK_ID_MAX) return -1;
    GarlicTask t;
    memset(&t, 0, sizeof(t));
    t.id = q->next_id;
    t.state = TASK_QUEUED;
    t.created_at = t.updated_at = task_now();
    strcpy(t.prompt, prompt);
    if (persist(q, &t) != 0) return -1;
    if (replace_task(q, &t) != 0) return -1;
    snprintf(id_out, id_cap, "task-%08u", t.id);
    return 0;
}

int task_queue_claim(TaskQueue *q, GarlicTask *out) {
    if (!q || !q->ready || !out) return -1;
    for (size_t i = 0; i < q->count; i++) {
        if (q->tasks[i].state != TASK_QUEUED) continue;
        GarlicTask t = q->tasks[i];
        t.state = TASK_RUNNING;
        t.attempts++;
        t.updated_at = task_now();
        if (persist(q, &t) != 0) return -1;
        q->tasks[i] = t;
        *out = t;
        return 1;
    }
    return 0;
}

static int terminal(TaskQueue *q, uint32_t id, TaskState state, const char *text) {
    if (!q || !q->ready || !text) return -1;
    int i = task_index(q, id);
    if (i < 0 || q->tasks[i].state != TASK_RUNNING) return -1;
    GarlicTask t = q->tasks[i];
    if (state == TASK_COMPLETED) {
        if (strlen(text) >= sizeof(t.result)) return -1;
        strcpy(t.result, text);
    } else {
        if (strlen(text) >= sizeof(t.error)) return -1;
        strcpy(t.error, text);
    }
    t.state = (uint32_t)state;
    t.updated_at = task_now();
    if (persist(q, &t) != 0) return -1;
    q->tasks[i] = t;
    return 0;
}

int task_queue_complete(TaskQueue *q, uint32_t id, const char *result) {
    return terminal(q, id, TASK_COMPLETED, result);
}

int task_queue_fail(TaskQueue *q, uint32_t id, const char *error) {
    return terminal(q, id, TASK_FAILED, error);
}

int task_queue_cancel(TaskQueue *q, uint32_t id) {
    if (!q || !q->ready) return -1;
    int i = task_index(q, id);
    if (i < 0 || (q->tasks[i].state != TASK_QUEUED && q->tasks[i].state != TASK_RUNNING)) return -1;
    GarlicTask t = q->tasks[i];
    t.state = TASK_CANCELED;
    t.updated_at = task_now();
    if (persist(q, &t) != 0) return -1;
    q->tasks[i] = t;
    return 0;
}

const GarlicTask *task_queue_get(const TaskQueue *q, uint32_t id) {
    if (!q || !q->ready) return NULL;
    int i = task_index(q, id);
    return i < 0 ? NULL : &q->tasks[i];
}

size_t task_queue_count_state(const TaskQueue *q, TaskState state) {
    size_t n = 0;
    if (q && q->ready) for (size_t i = 0; i < q->count; i++) n += q->tasks[i].state == (uint32_t)state;
    return n;
}
