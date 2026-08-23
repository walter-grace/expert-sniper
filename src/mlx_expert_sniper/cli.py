"""
mlx-sniper CLI.

Usage:
    mlx-sniper download qwen3.5-35b [-o ~/models/qwen35-35b]
    mlx-sniper calibrate <model-dir> [--quick] [--force] [--ram N]
    mlx-sniper run <model-dir> -p "prompt" [-v] [--max-tokens N]
    mlx-sniper chat <model-dir> [--max-tokens 500]
    mlx-sniper serve <model-dir> [--port 11434] [--host 127.0.0.1]
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

    calibrate(args.model_dir, ram_gb=args.ram, quick=args.quick)


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
    bias = bias_loaded
    print(f"Model loaded. Metal: {mx.get_active_memory()/1e9:.2f} GB")

    messages = [{"role": "user", "content": args.prompt}]

    t0 = time.time()
    token_count = 0
    first_token_time = None

    for chunk in generate_stream(eng, messages, bias=bias, max_tokens=args.max_tokens):
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

    # run
    p = sub.add_parser("run", help="Generate text from a prompt")
    p.add_argument("model_dir", help="Path to sniper model directory")
    p.add_argument("--prompt", "-p", required=True, help="Text prompt")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--verbose", "-v", action="store_true")

    # chat
    p = sub.add_parser("chat", help="Interactive multi-turn chat")
    p.add_argument("model_dir", help="Path to sniper model directory")
    p.add_argument("--max-tokens", type=int, default=500)

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
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
