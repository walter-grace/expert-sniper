# llama-cpp — Expert Sniper for GGUF models

The cross-platform path: an expert LRU/madvise cache that hooks llama.cpp's
eval callback so MoE GGUF models can run on machines that can't hold all
experts in RAM (measured: 0.57 tok/s on an 8 GB M2 Air where stock llama.cpp
produced no output, with ~1 MB overhead).

## Layout

- `patches/src/` — the expert-cache implementation (original Expert Sniper
  code, MIT): `llama-expert-cache.{cpp,h}`, `llama-expert-cache-ctx.{cpp,h}`
- `patches/apply.sh` — copies those sources into a stock llama.cpp checkout
  and anchor-patches the four integration points (CMake target, a
  `common_params` field, the `--expert-cache-size` CLI flag, and context
  init). Idempotent; warns instead of failing if upstream moved an anchor.
- `sniper.py` — a standalone agent/CLI that drives a patched `llama-server`
  over HTTP.

No llama.cpp source is vendored here; you patch your own checkout:

```bash
git clone https://github.com/ggml-org/llama.cpp
./patches/apply.sh ./llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_METAL=ON
cmake --build llama.cpp/build -j --target llama-server
llama-server -m model.gguf --expert-cache-size 512
```

Tested against llama.cpp master as of 2026-08.

> The Python direct-I/O expert reader that used to live here
> (`expert_io.py`) is superseded by the maintained implementation in
> `src/mlx_expert_sniper/expert_io.py`.

## Security note

`sniper.py` includes a `/shell` feature that executes model-proposed shell
commands on your machine. Use it only with prompts and models you trust,
and keep servers bound to `127.0.0.1` (the default).
