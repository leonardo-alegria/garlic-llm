#include "task_queue.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <io.h>
#define TQ_TEST_PATH "build/garlic_task_queue_test.journal"
#else
#include <unistd.h>
#define O_BINARY 0
#define TQ_TEST_PATH "/tmp/garlic_task_queue_test.journal"
#endif
#include <fcntl.h>

static int fails, checks;
#define CHECK(x) do { checks++; if (!(x)) { fails++; printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); } } while (0)

int main(void) {
    const char *path = TQ_TEST_PATH;
    unlink(path);
    TaskQueue q;
    CHECK(task_queue_init(&q, path) == 0);
    char id[TASK_ID_MAX];
    CHECK(task_queue_add(&q, "summarize the project", id, sizeof(id)) == 0);
    CHECK(!strcmp(id, "task-00000001"));
    CHECK(task_queue_add(&q, "review the plan", id, sizeof(id)) == 0);
    CHECK(!strcmp(id, "task-00000002"));
    CHECK(task_queue_count_state(&q, TASK_QUEUED) == 2);

    GarlicTask t;
    CHECK(task_queue_claim(&q, &t) == 1 && t.id == 1 && t.attempts == 1 && t.state == TASK_RUNNING);
    CHECK(task_queue_claim(&q, &t) == 1 && t.id == 2); /* module offers claims; scheduler must serialize calls */
    CHECK(task_queue_complete(&q, 1, "summary artifact") == 0);
    CHECK(task_queue_fail(&q, 2, "mock failure") == 0);

    /* Restart replays latest snapshots and retains the terminal results. */
    TaskQueue recovered;
    CHECK(task_queue_init(&recovered, path) == 0);
    const GarlicTask *one = task_queue_get(&recovered, 1);
    const GarlicTask *two = task_queue_get(&recovered, 2);
    CHECK(one && one->state == TASK_COMPLETED && !strcmp(one->result, "summary artifact"));
    CHECK(two && two->state == TASK_FAILED && !strcmp(two->error, "mock failure"));
    CHECK(task_queue_add(&recovered, "after restart", id, sizeof(id)) == 0);
    CHECK(!strcmp(id, "task-00000003"));

    /* A torn append is discarded; prior valid tasks remain available. */
    int fd = open(path, O_WRONLY | O_APPEND | O_BINARY);
    CHECK(fd >= 0);
    if (fd >= 0) { CHECK(write(fd, "torn", 4) == 4); close(fd); }
    TaskQueue after_torn;
    CHECK(task_queue_init(&after_torn, path) == 0);
    CHECK(task_queue_get(&after_torn, 3) != NULL);
    CHECK(task_queue_count_state(&after_torn, TASK_QUEUED) == 1);

    unlink(path);
    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
