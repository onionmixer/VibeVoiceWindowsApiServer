#pragma once
#include <cstdio>

inline int testsPassed = 0;
inline int testsFailed = 0;

inline void check(bool ok, const char* name) {
    if (ok) {
        fprintf(stderr, "  [PASS] %s\n", name);
        ++testsPassed;
    } else {
        fprintf(stderr, "  [FAIL] %s\n", name);
        ++testsFailed;
    }
}
