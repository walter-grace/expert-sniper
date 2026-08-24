# DFlash Benchmark

| suite | prompts | prompt tok avg | baseline tok/s | dflash tok/s | speedup | baseline score | dflash score | TTFT | peak memory | acceptance | prefix saved | baseline prefill tok/s | dflash prefill physical tok/s | dflash prefill apparent tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| smoke | 1 | 30.00 | 37.98 | 36.45 | 0.96x | n/a | n/a | 922.04 ms | 3.79 GB | 0.73 | n/a | 33.70 | 34.33 | 34.33 |

- mode: smoke
- suite: smoke
- model: mlx-community/Qwen3.5-4B-4bit
- draft: z-lab/Qwen3.5-4B-DFlash
- draft_quant: None
- git_hash: 09aecd1
- max_tokens: 200
- block_tokens: 16
- repeat: 2
- cooldown: 10
- prompt_count: 1
- prompt_ids: smoke-custom-write-a-python-function-that-reverses-a-linked-l-2e523c24
- prompt_source: smoke
- prompt_tokenization_mode: chat_template
- use_chat_template: True
- target_fa_window: 0
- draft_window: 64+1024
- verify_len_cap: 0
- verify_mode: adaptive
- only_dflash: False

## Per Prompt

| prompt id | prompt tokens | baseline tok/s | dflash tok/s | speedup | baseline score | dflash score | acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| smoke-custom-write-a-python-function-that-reverses-a-linked-l-2e523c24 | 30 | 37.98 | 36.45 | 0.96x | n/a | n/a | 0.73 |
