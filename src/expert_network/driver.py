"""Expert Network driver — attention/routing local, experts remote.

Loads the pinned weights through the standard engine (pinned_only=True, so
no bin/ layer file is opened), then installs the distributed reader. The
SAME forward pass as single-machine serving (generate.make_forward) drives
the network via its remote-compute hook, so improvements and measurements
apply to both tiers.

A driver machine needs only pinned.safetensors + config.json + tokenizer
files (~1 GB for the 30B-class models); the experts live on the nodes. The
bin/ directory may be absent entirely.
"""
import argparse
import sys
import time


def run(model_dir, nodes, prompt=None, max_tokens=200, chat=False,
        spec=False, spec_k=8, draft_url=None, draft_model=None):
    from mlx_expert_sniper.generate import load_engine, generate_stream
    from .reader import DistributedExpertReader

    print(f"Loading pinned model from {model_dir}...")
    engine, _, model_type = load_engine(model_dir, pinned_only=True)

    # pinned_only left engine.reader = None; the network is the reader
    engine.reader = DistributedExpertReader(nodes)
    engine.predictor = "none"  # nodes hold everything; nothing to prefetch

    print(f"Nodes:")
    for url, h in engine.reader.health().items():
        print(f"  {url}: {h.get('status')} "
              f"({h.get('experts_per_layer', '?')} experts/layer, "
              f"{h.get('memory_gb', '?')} GB)")

    def generate(messages):
        spec_stats = {}
        if spec:
            from mlx_expert_sniper.speculative import (spec_generate_stream,
                                                       RemoteDraft)
            dr = (RemoteDraft(draft_url, draft_model,
                              engine.tokenizer) if draft_url else None)
            stream = spec_generate_stream(engine, messages, bias=0.0,
                                          max_tokens=max_tokens, k=spec_k,
                                          draft=dr, stats=spec_stats)
        else:
            stream = generate_stream(engine, messages, bias=0.0,
                                     max_tokens=max_tokens)
        t0 = time.time()
        n = 0
        first = None
        for chunk in stream:
            if first is None:
                first = time.time() - t0
            sys.stdout.write(chunk)
            sys.stdout.flush()
            n += 1
        dt = time.time() - t0
        tps = n / (dt - first) if first and dt > first and n else 0
        print(f"\n\n  [{n} tok, {tps:.2f} tok/s, TTFT {first:.1f}s]")
        if spec_stats.get("forwards"):
            print(f"  Spec: {spec_stats['forwards']} forwards for {n} tokens "
                  f"({n/spec_stats['forwards']:.2f} tok/forward), "
                  f"accepted {spec_stats['accepted']}/{spec_stats['drafted']}")
        print(f"  {engine.reader.stats()}\n")

    if chat:
        messages = []
        print("Expert Network chat. /quit to exit.\n")
        while True:
            try:
                user = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not user or user.lower() in ("/quit", "/q"):
                break
            messages.append({"role": "user", "content": user})
            generate(messages)
    else:
        generate([{"role": "user", "content": prompt}])
    engine.reader.close()


def main():
    parser = argparse.ArgumentParser(description="Expert Network driver")
    parser.add_argument("model_dir",
                        help="Model dir with pinned.safetensors + config.json "
                             "+ tokenizer (bin/ not needed)")
    parser.add_argument("--nodes", required=True,
                        help="Comma-separated node URLs, e.g. "
                             "http://127.0.0.1:8301,http://127.0.0.1:8302")
    parser.add_argument("--prompt", "-p", default=None)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--spec", action="store_true",
                        help="Speculative decoding (prompt-lookup drafts)")
    parser.add_argument("--spec-k", type=int, default=8)
    parser.add_argument("--draft-url", default=None,
                        help="OpenAI-compatible /v1 base of a remote draft node")
    parser.add_argument("--draft-model", default=None)
    args = parser.parse_args()
    if not args.chat and not args.prompt:
        parser.error("need --prompt or --chat")
    run(args.model_dir, args.nodes.split(","), prompt=args.prompt,
        max_tokens=args.max_tokens, chat=args.chat,
        spec=args.spec, spec_k=args.spec_k,
        draft_url=args.draft_url, draft_model=args.draft_model)


if __name__ == "__main__":
    main()
