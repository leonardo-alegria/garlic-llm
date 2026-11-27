# Internal durable task queue: implementation status

This fork is building a PS4-native task manager. It does **not** embed RabbitMQ or Kafka: the console stores task state locally and serializes access to its single model engine. An external broker can later be a client/bridge, but is not required for on-console task durability.

## Current milestone: durable queue primitives

Implemented in `chat/task_queue.c` and `chat/task_queue.h`:

- Fixed-capacity journal (32 tasks), FIFO claim order, persistent IDs.
- Append-only snapshots with checksums and `fsync` before state changes become visible.
- Replay on startup; incomplete/corrupt final record truncation to the verified journal prefix.
- States: `queued`, `running`, `completed`, `failed`, `canceled`.
- Persisted prompt, result/error, attempt count, and timestamps.

Host test: `gcc -std=c99 -Wall -Wextra -g -fsanitize=address,undefined -I chat tests/task_queue_test.c chat/task_queue.c -o build/task_queue_test && build/task_queue_test`.

## Not integrated yet (follow-up milestones)

1. Add task endpoints to the HTTP API (`POST /v1/tasks`, status/list/cancel/result) and document them in OpenAPI.
2. Initialize the journal under `/data/garlic/tasks` and connect task claims to the main model loop, enforcing exactly one active inference.
3. Persist generated output/checkpoints as task artifacts and resume interrupted `running` tasks safely after reboot.
4. Add retry/backoff, idempotency keys, retention/compaction, queue limits, and UI task status/cancel controls.
5. Add optional OpenCode/RabbitMQ bridge documentation and integration tests.

The queue module is a foundation, not yet reachable from the installed app/API. The next integration step must add exclusive scheduler ownership: the current API still returns 503 when its single chat slot is busy. `task_queue_claim` itself persists a claim but does not provide cross-thread synchronization; callers must serialize queue operations on the main thread or guard them with a mutex.
