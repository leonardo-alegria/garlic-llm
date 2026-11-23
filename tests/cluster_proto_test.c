// Host tests for chat/cluster_proto.c (the DarkGarlic cluster's wire protocol and cluster.txt).
// Build and run: gcc -std=c99 -Wall -Wextra -g -fsanitize=address,undefined -I chat tests/cluster_proto_test.c chat/cluster_proto.c -o build/cluster_proto_test && build/cluster_proto_test
#include <stdio.h>
#include <string.h>

#include "cluster_proto.h"

static int fails, checks;
#define CHECK(c)                                                              \
    do {                                                                      \
        checks++;                                                             \
        if (!(c)) {                                                           \
            fails++;                                                          \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);               \
        }                                                                     \
    } while (0)

static void test_header(void) {
    ClusterMsg m = {0}, r;
    m.op = CL_RUN, m.seq = 77, m.pos0 = 1234, m.n = 32, m.flags = CL_F_SAMPLE, m.temperature = 0.7f, m.seed = 99;
    m.payload = 32 * 4096 * 4, m.check = 0xDEADBEEF;
    uint8_t b[CLUSTER_HDR];
    cluster_put_hdr(b, &m);
    CHECK(b[0] == 'D' && b[1] == 'G' && b[2] == 'C' && b[3] == 'L');
    CHECK(cluster_get_hdr(b, &r) == 0);
    CHECK(r.op == CL_RUN && r.seq == 77 && r.pos0 == 1234 && r.n == 32 && r.flags == CL_F_SAMPLE);
    CHECK(r.temperature == 0.7f && r.seed == 99 && r.payload == 32 * 4096 * 4 && r.check == 0xDEADBEEF);
    b[0] ^= 1;
    CHECK(cluster_get_hdr(b, &r) == -1);   // not our magic
    b[0] ^= 1;
    m.n = 0;
    cluster_put_hdr(b, &m);
    CHECK(cluster_get_hdr(b, &r) == -1);   // a RUN of no rows
    m.n = CLUSTER_MAX_ROWS + 1;
    cluster_put_hdr(b, &m);
    CHECK(cluster_get_hdr(b, &r) == -1);
    m.n = 1, m.payload = CLUSTER_MAX_PAYLOAD + 1;
    cluster_put_hdr(b, &m);
    CHECK(cluster_get_hdr(b, &r) == -1);   // too big
    m.op = 0, m.payload = 0;
    cluster_put_hdr(b, &m);
    CHECK(cluster_get_hdr(b, &r) == -1);   // unknown op
    m.op = CL_PING;
    cluster_put_hdr(b, &m);
    CHECK(cluster_get_hdr(b, &r) == 0);
}

static void test_fnv(void) {
    CHECK(cluster_fnv("", 0) == 0);
    CHECK(cluster_fnv("a", 1) == 0xE40C292Cu);        // the published FNV-1a 32 value of "a"
    CHECK(cluster_fnv("foobar", 6) == 0xBF9CF968u);
    CHECK(cluster_fnv_mix(cluster_fnv_init(), "a", 1) == cluster_fnv("a", 1));
    CHECK(cluster_fnv_mix(cluster_fnv_init(), "foobar", 6) == cluster_fnv("foobar", 6));
}

static void test_hello(void) {
    ClusterHello h = {1, 3, 23, 45, 64, 5120, 248070, 0, "Abliterated Qwen3.5-27B", 0}, r;
    uint8_t b[CLUSTER_HELLO_BYTES];
    cluster_put_hello(b, &h);
    cluster_get_hello(b, &r);
    CHECK(r.node == 1 && r.nodes == 3 && r.layer_first == 23 && r.layer_end == 45 && r.n_layers == 64);
    CHECK(r.hidden == 5120 && r.vocab == 248070 && r.fingerprint == 0 && !strcmp(r.name, h.name));
    h.fingerprint = 0xA1B2C3D4u;
    cluster_put_hello(b, &h);
    cluster_get_hello(b, &r);
    CHECK(r.fingerprint == 0xA1B2C3D4u && !strcmp(r.name, h.name));
    // a 64-byte hello (no fingerprint word) reads as fingerprint 0
    uint8_t old[CLUSTER_HELLO_BYTES];
    memset(old, 0, sizeof(old));
    cluster_put_hello(b, &h);
    memcpy(old, b, 64);
    cluster_get_hello(old, &r);
    CHECK(r.fingerprint == 0 && r.node == 1 && !strcmp(r.name, h.name));
}

static void test_config(void) {
    ClusterConfig c;
    char err[160];
    const char* ok = "# two workers\nname = Abliterated Qwen3.5-27B\r\n worker = 192.168.1.21   # node 1\n"
                     "worker=10.0.0.5:8091\ntimeout = 90\n\n";
    CHECK(cluster_parse_config(ok, strlen(ok), &c, err, sizeof(err)) == 0);
    CHECK(c.n_workers == 2 && c.timeout_s == 90 && !strcmp(c.name, "Abliterated Qwen3.5-27B"));
    CHECK(c.ip[0][0] == 192 && c.ip[0][3] == 21 && c.port[0] == CLUSTER_PORT);
    CHECK(c.ip[1][0] == 10 && c.ip[1][3] == 5 && c.port[1] == CLUSTER_PORT);
    const char* none = "# nothing\nname = x\n";
    CHECK(cluster_parse_config(none, strlen(none), &c, err, sizeof(err)) == -1);
    const char* bad_ip = "worker = 192.168.1\n";
    CHECK(cluster_parse_config(bad_ip, strlen(bad_ip), &c, err, sizeof(err)) == -1 && strstr(err, "line 1"));
    const char* bad_ip2 = "worker = 192.168.1.300\n";
    CHECK(cluster_parse_config(bad_ip2, strlen(bad_ip2), &c, err, sizeof(err)) == -1);
    const char* bad_key = "worker = 1.2.3.4\ncolour = red\n";
    CHECK(cluster_parse_config(bad_key, strlen(bad_key), &c, err, sizeof(err)) == -1 && strstr(err, "line 2"));
    const char* bad_port = "worker = 1.2.3.4:0\n";
    CHECK(cluster_parse_config(bad_port, strlen(bad_port), &c, err, sizeof(err)) == -1);
    const char* other_port = "worker = 1.2.3.4:9000\n";
    CHECK(cluster_parse_config(other_port, strlen(other_port), &c, err, sizeof(err)) == -1 && strstr(err, "8091"));
    const char* no_eq = "worker 1.2.3.4\n";
    CHECK(cluster_parse_config(no_eq, strlen(no_eq), &c, err, sizeof(err)) == -1);
    char many[512] = "";
    for (int i = 0; i < CLUSTER_MAX_WORKERS + 1; i++) strcat(many, "worker = 1.2.3.4\n");
    CHECK(cluster_parse_config(many, strlen(many), &c, err, sizeof(err)) == -1);
    const char* no_newline = "worker = 1.2.3.4";
    CHECK(cluster_parse_config(no_newline, strlen(no_newline), &c, err, sizeof(err)) == 0 && c.n_workers == 1);
}

static void test_chain(void) {
    char err[160];
    ClusterHello head = {0, 3, 0, 23, 64, 5120, 248070, 1, "M", 0};
    ClusterHello w[2] = {{1, 3, 23, 45, 64, 5120, 248070, 0, "M", 0}, {2, 3, 45, 64, 64, 5120, 248070, 2, "M", 0}};
    CHECK(cluster_check_chain(&head, w, 2, err, sizeof(err)) == 0);
    ClusterHello swapped[2] = {w[1], w[0]};
    CHECK(cluster_check_chain(&head, swapped, 2, err, sizeof(err)) == -1 && strstr(err, "order"));
    ClusterHello gap[2] = {w[0], w[1]};
    gap[1].layer_first = 46;
    CHECK(cluster_check_chain(&head, gap, 2, err, sizeof(err)) == -1);
    ClusterHello other[2] = {w[0], w[1]};
    strcpy(other[1].name, "N");
    CHECK(cluster_check_chain(&head, other, 2, err, sizeof(err)) == -1 && strstr(err, "different model"));
    ClusterHello nohead[2] = {w[0], w[1]};
    nohead[1].flags = 0;
    CHECK(cluster_check_chain(&head, nohead, 2, err, sizeof(err)) == -1);
    ClusterHello short_end[2] = {w[0], w[1]};
    short_end[1].layer_end = 63;
    CHECK(cluster_check_chain(&head, short_end, 2, err, sizeof(err)) == -1);
    CHECK(cluster_check_chain(&head, w, 1, err, sizeof(err)) == -1);   // 3-node shard, one worker listed
    ClusterHello head2 = {0, 2, 0, 19, 32, 4096, 248070, 1, "M", 0}, w2 = {1, 2, 19, 32, 32, 4096, 248070, 2, "M", 0};
    CHECK(cluster_check_chain(&head2, &w2, 1, err, sizeof(err)) == 0);
    head2.fingerprint = 0x11111111u;
    w2.fingerprint = 0x11111111u;
    CHECK(cluster_check_chain(&head2, &w2, 1, err, sizeof(err)) == 0);
    w2.fingerprint = 0x22222222u;
    CHECK(cluster_check_chain(&head2, &w2, 1, err, sizeof(err)) == -1 && strstr(err, "weights"));
    w2.fingerprint = 0;   // an older worker did not send a fingerprint: do not reject it
    CHECK(cluster_check_chain(&head2, &w2, 1, err, sizeof(err)) == 0);
}

int main(void) {
    test_header();
    test_fnv();
    test_hello();
    test_config();
    test_chain();
    printf("%d checks, %d failed\n", checks, fails);
    return fails != 0;
}
