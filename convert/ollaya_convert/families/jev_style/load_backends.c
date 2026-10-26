/* Linked next to the author's jev_score_v2.cpp (unchanged) when it is built against ggml-org's stock release
 * libraries of llama.cpp b11146, the ones the runner loads. Those builds load the CPU backend variants at run
 * time (GGML_BACKEND_DL), and jev-score-v2 calls only llama_backend_init(), so the backends are loaded here,
 * before main(), from the directory that holds the libraries (JEV_GGML_BACKEND_DIR).
 *
 *   c++ -std=c++17 -O2 -I<b11146>/include -I<b11146>/ggml/include -I<b11146>/vendor \
 *       jev_score_v2.cpp -x c load_backends.c -o jev-score-v2 -L<libs> -lllama -lggml -lggml-base -Wl,-rpath,<libs>
 */
#include <stdlib.h>

#include "ggml-backend.h"

__attribute__((constructor)) static void jev_load_backends(void) {
    const char * dir = getenv("JEV_GGML_BACKEND_DIR");
    if (dir && *dir) {
        ggml_backend_load_all_from_path(dir);
    } else {
        ggml_backend_load_all();
    }
}
