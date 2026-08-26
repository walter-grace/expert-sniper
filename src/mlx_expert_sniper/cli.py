"""
mlx-sniper CLI.

Usage:
    mlx-sniper download qwen3.5-35b [-o ~/models/qwen35-35b]
    mlx-sniper calibrate <model-dir> [--quick] [--force] [--ram N]
    mlx-sniper run <model-dir> -p "prompt" [-v] [--max-tokens N]
    mlx-sniper chat <model-dir> [--max-tokens 500]
    mlx-sniper serve <model-dir> [--port 11434] [--host 127.0.0.1]
    mlx-sniper eval <model-dir> [--bias N] [--text FILE] [--chunks N]
"""
import argparse
import sys
import os
import time


def cmd_download(args):
    from .download import download_model, list_models

    if args.model_name == "list":
        list_models()
        return

    output = args.output
    if output:
        output = os.path.expanduser(output)

    download_model(
        args.model_name,
        output_dir=output,
        calibrate_quick=not args.full_calibrate,
        keep_download=args.keep_download,
    )


def cmd_serve(args):
    from .server import run_server
    run_server(
        model_dir=args.model_dir,
        host=args.host,
        port=args.port,
    )


def cmd_calibrate(args):
    from .calibrate import calibrate, load_calibration

    if not args.force:
        existing = load_calibration(args.model_dir)
        if existing:
            print(f"Calibration exists: cache={existing['cache_size']}, "
                  f"bias={existing['routing_bias']}, "
                  f"dead={existing['reap_dead_pct']:.1%}")
            print(f"Use --force to overwrite.")
            return

    calibrate(args.model_dir, ram_gb=args.ram, quick=args.quick,
              ppl_tolerance=args.ppl_tolerance)


def cmd_run(args):
    from .generate import load_engine, generate_stream
    from .calibrate import load_calibration
    import mlx.core as mx

    cal = load_calibration(args.model_dir)
    if cal:
        bias = cal["routing_bias"]
        print(f"Loaded calibration: cache={cal['cache_size']}, bias={bias}, "
              f"dead={cal['reap_dead_pct']:.1%}")
    else:
        bias = 0.0
        print(f"No calibration found. Run 'mlx-sniper calibrate {args.model_dir}'")

    eng, bias_loaded, _ = load_engine(args.model_dir)
    bias = bias_loaded if args.bias is None else args.bias
    print(f"Model loaded. Metal: {mx.get_active_memory()/1e9:.2f} GB  bias={bias}")

    messages = [{"role": "user", "content": args.prompt}]

    spec_stats = {}
    if args.spec:
        from .speculative import spec_generate_stream, ModelDraft, RemoteDraft
        if args.dflash:
            from .speculative import DFlashDraft
            draft = DFlashDraft(args.dflash, eng)
            args.spec_k = min(args.spec_k, draft.block - 1)
        elif args.draft_url:
            draft = RemoteDraft(args.draft_url, args.draft_model,
                                eng.tokenizer)
        elif args.draft_model:
            draft = ModelDraft(args.draft_model, eng.tokenizer)
        else:
            draft = None
        stream = spec_generate_stream(eng, messages, bias=bias,
                                      max_tokens=args.max_tokens,
                                      k=args.spec_k, draft=draft,
                                      stats=spec_stats)
    else:
        stream = generate_stream(eng, messages, bias=bias,
                                 max_tokens=args.max_tokens)

    t0 = time.time()
    token_count = 0
    first_token_time = None

    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.time()
        sys.stdout.write(chunk)
        sys.stdout.flush()
        token_count += 1

    elapsed = time.time() - t0
    ttft = (first_token_time - t0) if first_token_time else elapsed
    tps = token_count / (elapsed - ttft) if elapsed > ttft and token_count > 0 else 0

    if args.verbose:
        print(f"\n\n  {token_count} tokens | {tps:.2f} tok/s | TTFT: {ttft:.2f}s | "
              f"Total: {elapsed:.2f}s")
        if spec_stats.get("forwards"):
            acc = (spec_stats["accepted"] / spec_stats["drafted"]
                   if spec_stats["drafted"] else 0)
            print(f"  Spec: {spec_stats['forwards']} forwards for "
                  f"{token_count} tokens ({token_count/spec_stats['forwards']:.2f} "
                  f"tok/forward), drafted={spec_stats['drafted']}, "
                  f"accepted={spec_stats['accepted']} ({acc:.0%})")
        print(f"  Cache: {eng.reader.stats()}")
        print(f"  Metal: {mx.get_active_memory()/1e9:.2f} GB")
    else:
        print()


def cmd_chat(args):
    from .generate import load_engine, generate_stream
    from .calibrate import load_calibration
    import mlx.core as mx

    cal = load_calibration(args.model_dir)
    bias = cal["routing_bias"] if cal else 0.0

    print("Loading model...", end=" ", flush=True)
    eng, bias_loaded, _ = load_engine(args.model_dir)
    bias = bias_loaded
    print(f"ready. ({mx.get_active_memory()/1e9:.1f} GB)")
    print(f"Type your message. /clear to reset, /stats for info, /quit to exit.\n")

    messages = []
    session_tokens = 0
    session_time = 0.0

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("bye.")
            break

        if user_input.lower() == "/clear":
            messages.clear()
            print("  (conversation cleared)\n")
            continue

        if user_input.lower() == "/stats":
            avg_tps = session_tokens / session_time if session_time > 0 else 0
            print(f"  Tokens: {session_tokens}")
            print(f"  Time:   {session_time:.1f}s")
            print(f"  Speed:  {avg_tps:.1f} tok/s")
            print(f"  Cache:  {eng.reader.stats()}")
            print(f"  Metal:  {mx.get_active_memory()/1e9:.1f} GB\n")
            continue

        messages.append({"role": "user", "content": user_input})

        t0 = time.time()
        token_count = 0
        full_response = ""

        print()
        for chunk in generate_stream(eng, messages, bias=bias, max_tokens=args.max_tokens):
            sys.stdout.write(chunk)
            sys.stdout.flush()
            full_response += chunk
            token_count += 1

        elapsed = time.time() - t0
        tps = token_count / elapsed if elapsed > 0 else 0
        session_tokens += token_count
        session_time += elapsed

        print(f"\n\n[{token_count} tok, {tps:.1f} tok/s]\n")

        messages.append({"role": "assistant", "content": full_response})


def cmd_publish(args):
    """Post a prepared model to the network so other machines can serve it.

    The network's models come from whoever has the disk to prepare them. This
    hashes every expert block, uploads the manifest under a name you claim,
    and prints the command other people run to take a slice. Your machine
    then seeds those blocks to them, which is itself paid work."""
    import hashlib, json, urllib.request
    api_key = args.key or os.environ.get("HERO_RUN_KEY")
    if not api_key:
        sys.exit("Need a key: --key hr_live_... (or HERO_RUN_KEY)")

    if not args.from_hf and not args.model_dir:
        sys.exit("publish needs a model dir, or --from-hf <repo>")
    PAGE = 16384
    if args.from_hf:
        # Stream the checkpoint through sha256 straight from HuggingFace: the
        # publisher never needs the disk to hold the model. Flash-Next is
        # ~75 GB of experts; at CDN speed that is under an hour.
        from expert_network.hf_source import HFCheckpoint
        hf = HFCheckpoint(args.from_hf, jobs=args.jobs)
        num_layers, num_experts = hf.num_layers, hf.num_experts
        lay0 = hf.public_layout(0)
        bsize = lay0["expert_block_size"]
        key = args.name or f"{num_layers}x{num_experts}"
        print(f"{args.from_hf}: {num_layers} layers x {num_experts} experts, "
              f"{bsize / 1e6:.2f} MB blocks — streaming "
              f"{num_layers * num_experts * bsize / 1e9:.0f} GB through sha256")
        blocks = hf.hash_all_blocks()
        total = len(blocks) * bsize
        manifest = {"model_type": hf.tc.get("model_type"), "num_layers": num_layers,
                    "num_experts": num_experts, "expert_block_size": bsize,
                    "tensors": lay0["tensors"], "source": args.from_hf, "blocks": blocks}
        model_dir = None
    else:
        model_dir = os.path.expanduser(args.model_dir)
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)
        num_layers = config["num_hidden_layers"]
        num_experts = (config.get("num_experts") or config.get("n_routed_experts"))
        key = args.name or f"{num_layers}x{num_experts}"
        blocks, total, tensors = {}, 0, None
        for li in range(num_layers):
            path = os.path.join(model_dir, "bin", f"layer_{li:02d}.bin")
            with open(path, "rb") as f:
                header = json.loads(f.read(PAGE).rstrip(b"\x00"))
                bsize = header["layout"]["expert_block_size"]
                tensors = tensors or header["layout"].get("tensors")
                for eid in range(num_experts):
                    f.seek(PAGE + eid * bsize)
                    blocks[f"{li}:{eid}"] = hashlib.sha256(f.read(bsize)).hexdigest()
                    total += bsize
            print(f"  hashing layer {li + 1}/{num_layers}", end="\r", flush=True)
        manifest = {"model_type": config.get("model_type"), "num_layers": num_layers,
                    "num_experts": num_experts, "expert_block_size": bsize,
                    "tensors": tensors, "blocks": blocks}
    print(f"\n{len(blocks)} blocks, {total / 1e9:.1f} GB")
    if args.save:
        with open(os.path.expanduser(args.save), "w") as f:
            json.dump(manifest, f)
        print(f"manifest saved to {args.save}")
    body = json.dumps({"model": key, "manifest": manifest}).encode()
    req = urllib.request.Request(f"{args.coordinator}/api/yield/manifest", data=body,
                                 headers={"Content-Type": "application/json",
                                          "x-api-key": api_key,
                                          "User-Agent": "mlx-sniper/0.3"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"publish failed: {e.read().decode()[:200]}")
    if out.get("error"):
        sys.exit(f"publish failed: {out['error']}")

    print(f"\npublished '{key}' — {out.get('blocks')} blocks")
    print(f"\nOthers join it with:")
    print(f"  expert-fetch --model {key} --partition <yours> -o ~/models/{key}")
    print(f"  expert-node --model-dir ~/models/{key} --roster auto --me <id> --join <key>")
    if args.from_hf:
        print(f"  or straight from the source, no peer needed:")
        print(f"  expert-fetch --from-hf {args.from_hf} --model {key} --partition <yours> -o ~/models/{key}")
    else:
        print(f"\nServe it yourself so there is something to pull from:")
        print(f"  expert-node --model-dir {model_dir} --roster auto --me $(hostname) --join {api_key[:12]}…")


def cmd_manifest(args):
    """Content-address every expert block: manifest.json of sha256 hashes.

    The Machine Yield coordinator verifies node challenge responses against
    this, so serving capability is provable, not self-reported."""
    import hashlib, json
    model_dir = os.path.expanduser(args.model_dir)
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    num_layers = config["num_hidden_layers"]
    num_experts = config["num_experts"]
    PAGE_SIZE = 16384
    manifest = {}
    for li in range(num_layers):
        path = os.path.join(model_dir, "bin", f"layer_{li:02d}.bin")
        with open(path, "rb") as f:
            header = json.loads(f.read(PAGE_SIZE).rstrip(b"\x00"))
            block = header["layout"]["expert_block_size"]
            for eid in range(num_experts):
                f.seek(PAGE_SIZE + eid * block)
                manifest[f"{li}:{eid}"] = hashlib.sha256(f.read(block)).hexdigest()
        print(f"  layer {li + 1}/{num_layers}", end="\r", flush=True)
    out = os.path.join(model_dir, "manifest.json")
    with open(out, "w") as f:
        json.dump({"model_type": config.get("model_type"),
                   "num_layers": num_layers, "num_experts": num_experts,
                   "expert_block_size": block, "blocks": manifest}, f)
    print(f"\n{len(manifest)} blocks hashed -> {out}")


def cmd_preprocess(args):
    from .preprocess import preprocess
    preprocess(args.src_dir, args.out_dir)


def cmd_eval(args):
    from .evaluate import evaluate_model
    evaluate_model(
        args.model_dir,
        bias=args.bias,  # None = use calibrated bias
        text_path=args.text,
        seq_len=args.seq_len,
        max_chunks=args.chunks,
        mode="decode" if args.decode else "prefill",
    )


def main():
    parser = argparse.ArgumentParser(
        prog="mlx-sniper",
        description="Run MoE models larger than RAM on Apple Silicon",
    )
    sub = parser.add_subparsers(dest="command")

    # download
    p = sub.add_parser("download", help="Download, preprocess, and calibrate a model")
    p.add_argument("model_name", help="Model name (e.g. qwen3.5-35b) or 'list'")
    p.add_argument("-o", "--output", default=None, help="Output directory")
    p.add_argument("--full-calibrate", action="store_true", help="Full calibration with bias sweep")
    p.add_argument("--keep-download", action="store_true", help="Keep raw HF download")

    # serve
    p = sub.add_parser("serve", help="Ollama-compatible HTTP server")
    p.add_argument("model_dir", help="Path to sniper model directory")
    p.add_argument("--port", type=int, default=11434, help="Port (default: 11434)")
    p.add_argument("--host", default="127.0.0.1", help="Host (use 0.0.0.0 for network)")

    # calibrate
    p = sub.add_parser("calibrate", help="One-time model calibration (~2-8 min)")
    p.add_argument("model_dir", help="Path to sniper model directory")
    p.add_argument("--ram", type=float, default=None, help="Override RAM (GB)")
    p.add_argument("--quick", action="store_true", help="Skip bias sweep (2 min)")
    p.add_argument("--force", action="store_true", help="Overwrite existing calibration")
    p.add_argument("--ppl-tolerance", type=float, default=None,
                   help="Bias passes if decode-ppl <= baseline * tolerance "
                        "(default 1.05; e.g. 1.10 trades ~6%% ppl for ~50%% speed "
                        "by admitting bias 0.5 on Qwen3-30B)")

    # run
    p = sub.add_parser("run", help="Generate text from a prompt")
    p.add_argument("model_dir", help="Path to sniper model directory")
    p.add_argument("--prompt", "-p", required=True, help="Text prompt")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--spec", action="store_true",
                   help="Speculative decoding (prompt-lookup drafts by default)")
    p.add_argument("--spec-k", type=int, default=8, help="Draft tokens per step")
    p.add_argument("--draft-model", default=None,
                   help="Tokenizer-compatible draft model path/repo (local), "
                        "or the model id when using --draft-url")
    p.add_argument("--bias", type=float, default=None,
                   help="Override the calibrated routing bias (0 = none)")
    p.add_argument("--dflash", default=None,
                   help="DFlash block-diffusion draft head for this target "
                        "(path or HF repo, e.g. z-lab/Qwen3-Coder-30B-A3B-DFlash)")
    p.add_argument("--draft-url", default=None,
                   help="OpenAI-compatible /v1 base URL of a remote draft "
                        "node (a GPU/DGX box on the network)")

    # chat
    p = sub.add_parser("chat", help="Interactive multi-turn chat")
    p.add_argument("model_dir", help="Path to sniper model directory")
    p.add_argument("--max-tokens", type=int, default=500)

    # manifest
    p = sub.add_parser("manifest", help="Write sha256 manifest of all expert blocks")
    p.add_argument("model_dir", help="Path to sniper model directory")

    # publish — post a prepared model so the rest of the network can serve it
    p = sub.add_parser("publish", help="Post a prepared model to the network")
    p.add_argument("model_dir", nargs="?", default=None,
                   help="Path to sniper model directory (omit with --from-hf)")
    p.add_argument("--from-hf", default=None, metavar="REPO",
                   help="hash the experts straight from a HuggingFace MLX checkpoint; "
                        "no local copy of the model needed")
    p.add_argument("--jobs", type=int, default=12, help="parallel range reads for --from-hf")
    p.add_argument("--save", default=None, help="also write the manifest JSON here")
    p.add_argument("--name", default=None, help="model key others will use (default: layersXexperts)")
    p.add_argument("--key", default=None, help="hr_live_ key (or HERO_RUN_KEY)")
    p.add_argument("--coordinator", default="https://herorunai.com")

    # preprocess
    p = sub.add_parser("preprocess", help="Split a downloaded MLX model into streaming format")
    p.add_argument("src_dir", help="Downloaded MLX model directory")
    p.add_argument("out_dir", help="Output streaming-format directory")

    # eval
    p = sub.add_parser("eval", help="Teacher-forced perplexity on held-out text")
    p.add_argument("model_dir", help="Path to sniper model directory")
    p.add_argument("--bias", type=float, default=None,
                   help="Routing bias (default: calibrated bias)")
    p.add_argument("--text", default=None, help="Eval text file (default: bundled)")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--chunks", type=int, default=8)
    p.add_argument("--decode", action="store_true",
                   help="Token-by-token decode-mode ppl (slow; engages the "
                        "routing bias like real serving — prefill mode doesn't)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    cmds = {
        "download": cmd_download,
        "serve": cmd_serve,
        "calibrate": cmd_calibrate,
        "run": cmd_run,
        "chat": cmd_chat,
        "eval": cmd_eval,
        "preprocess": cmd_preprocess,
        "manifest": cmd_manifest,
        "publish": cmd_publish,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
