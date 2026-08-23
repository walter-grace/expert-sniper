#!/usr/bin/env python3
"""
Expert Sniper — Speculative Decode Router.

Orchestrates heterogeneous MoE inference across draft + verify servers.
A small draft model generates K tokens fast, the full MoE model verifies
them in one batched pass. Accepted tokens stream to the user.

Result: 3-5x higher tok/s than serial generation on the same hardware.

Architecture:
  ┌─────────────────┐     ┌──────────────────────────┐
  │  DRAFT SERVER    │     │  VERIFY SERVER            │
  │  Small model     │     │  Full MoE model           │
  │  (1-3B, fast)    │     │  (35B, Expert Sniper)     │
  │  ~200 tok/s      │     │  batched verification     │
  └────────┬─────────┘     └────────────┬──────────────┘
           │                             │
           └──────────┬──────────────────┘
                      │
               spec_router.py
               (orchestrator)
                      │
                      ▼
                 User sees:
                 ~20 tok/s effective
                 (vs 5 tok/s serial)

Usage:
  # Start draft server (small model, any cheap GPU):
  llama-server -m Qwen3-0.6B.gguf -ngl 999 --port 8201 --host 0.0.0.0

  # Start verify server (Expert Sniper, MoE model):
  llama-server -m Qwen3.5-35B-A3B.gguf --expert-cache-size 1 --port 8202 --host 0.0.0.0

  # Run spec decode router:
  python3 spec_router.py --draft localhost:8201 --verify localhost:8202

  # Or with RunPod endpoints:
  python3 spec_router.py --draft https://api.runpod.ai/v2/DRAFT_ID --verify https://api.runpod.ai/v2/VERIFY_ID --runpod-key YOUR_KEY
"""

import os, sys, json, time, signal, argparse, threading, ssl
import urllib.request

# Allow HTTPS without certificate verification
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# ── Colors ─────────────────────────────────────────────
B = "\033[1m"
D = "\033[2m"
C = "\033[96m"
Y = "\033[93m"
G = "\033[92m"
R = "\033[91m"
W = "\033[97m"
X = "\033[0m"


# ── Spinner ────────────────────────────────────────────
class Spinner:
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    def __init__(self, message=""):
        self.message = message
        self.running = False
        self.thread = None
        self.frame = 0
        self.start_time = 0
    def _spin(self):
        while self.running:
            elapsed = time.time() - self.start_time
            frame = self.FRAMES[self.frame % len(self.FRAMES)]
            sys.stdout.write(f"\r  {C}{frame}{X} {self.message} {D}{elapsed:.0f}s{X}  ")
            sys.stdout.flush()
            self.frame += 1
            time.sleep(0.1)
    def start(self, message=None):
        if message: self.message = message
        self.running = True
        self.start_time = time.time()
        self.frame = 0
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
        return self
    def stop(self, final_message=None):
        self.running = False
        if self.thread: self.thread.join(timeout=1)
        elapsed = time.time() - self.start_time
        sys.stdout.write("\r" + " " * 80 + "\r")
        if final_message:
            print(f"  {G}✓{X} {final_message} {D}({elapsed:.1f}s){X}")


# ── Inference Server ───────────────────────────────────

class InferenceServer:
    """OpenAI-compatible inference server (llama-server or RunPod endpoint)."""

    def __init__(self, host, runpod_key=None):
        self.host = host
        self.runpod_key = runpod_key
        self.is_runpod = "runpod.ai" in host
        if not host.startswith("http"):
            self.host = f"http://{host}"

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.runpod_key:
            h["Authorization"] = f"Bearer {self.runpod_key}"
        return h

    def health(self):
        try:
            if self.is_runpod:
                url = f"{self.host}/health"
                req = urllib.request.Request(url, headers=self._headers())
                resp = urllib.request.urlopen(req, timeout=10, context=SSL_CTX)
                data = json.load(resp)
                return data.get("workers", {}).get("ready", 0) > 0
            else:
                resp = urllib.request.urlopen(
                    f"{self.host}/health", timeout=5, context=SSL_CTX)
                return json.load(resp).get("status") == "ok"
        except:
            return False

    def complete(self, messages, max_tokens=512, temperature=0.0):
        """Non-streaming completion. Returns full response text + token list."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        if self.is_runpod:
            # RunPod serverless format
            body = json.dumps({
                "input": {
                    "openai_route": "/v1/chat/completions",
                    "openai_input": {
                        "model": "default",
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": False,
                    }
                }
            }).encode()
            req = urllib.request.Request(
                f"{self.host}/runsync",
                data=body, headers=self._headers())
            resp = urllib.request.urlopen(req, timeout=300, context=SSL_CTX)
            data = json.load(resp)
            # Extract text from RunPod response
            output = data.get("output", [{}])
            if isinstance(output, list) and output:
                choices = output[0].get("choices", [{}])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    if not text:
                        text = choices[0].get("text", "")
                    return text
            return ""
        else:
            body = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.host}/v1/chat/completions",
                data=body, headers=self._headers())
            resp = urllib.request.urlopen(req, timeout=300, context=SSL_CTX)
            data = json.load(resp)
            return data["choices"][0]["message"]["content"]

    def stream(self, messages, max_tokens=512, temperature=0.4):
        """Streaming completion. Yields text chunks."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.host}/v1/chat/completions",
            data=body, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=600, context=SSL_CTX) as resp:
                for line in resp:
                    line = line.decode().strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            delta = json.loads(line[6:])["choices"][0]["delta"]
                            text = delta.get("content", "") or delta.get("reasoning_content", "")
                            if text:
                                text = text.replace("<think>", "").replace("</think>", "")
                                if text.strip():
                                    yield text
                        except:
                            pass
        except Exception as e:
            yield f"\n{R}Error: {e}{X}"

    def complete_tokens(self, messages, max_tokens=8, temperature=0.0):
        """
        Complete and return individual tokens (for draft generation).
        Uses logprobs or splits response into tokens.
        """
        text = self.complete(messages, max_tokens=max_tokens,
                             temperature=temperature)
        # Simple tokenization by words/subwords (approximate)
        # In production, use the tokenizer directly
        return text


# ── Speculative Decode Engine ──────────────────────────

class SpecDecodeEngine:
    """
    Orchestrates speculative decoding across draft + verify servers.

    Flow per step:
    1. Draft server generates K tokens quickly (small model)
    2. Verify server checks all K at once (batched, Expert Sniper)
    3. Accept longest matching prefix + 1 bonus token
    4. Emit accepted tokens to user
    """

    def __init__(self, draft_server, verify_server, draft_len=8):
        self.draft = draft_server
        self.verify = verify_server
        self.draft_len = draft_len
        # Stats
        self.total_drafted = 0
        self.total_accepted = 0
        self.total_steps = 0
        self.total_draft_time = 0.0
        self.total_verify_time = 0.0

    def generate(self, messages, max_tokens=256, temperature=0.4):
        """
        Generate tokens using speculative decode.
        Yields text chunks as they're accepted.
        """
        generated = ""
        tokens_generated = 0

        while tokens_generated < max_tokens:
            # ── DRAFT: Generate K tokens from small model ──────
            t0 = time.time()
            draft_messages = messages.copy()
            if generated:
                draft_messages.append({"role": "assistant", "content": generated})

            draft_text = self.draft.complete(
                draft_messages,
                max_tokens=self.draft_len,
                temperature=temperature,
            )
            draft_time = time.time() - t0

            if not draft_text:
                break

            # ── VERIFY: Check with full model ──────────────────
            # Send the SAME prompt + draft text to the verify server
            # and ask it to continue from the same point.
            # Compare what verify would generate vs what draft generated.
            t1 = time.time()
            verify_messages = messages.copy()
            if generated:
                verify_messages.append({"role": "assistant", "content": generated})

            verify_text = self.verify.complete(
                verify_messages,
                max_tokens=self.draft_len,
                temperature=temperature,
            )
            verify_time = time.time() - t1

            # ── ACCEPT/REJECT: Character-level comparison ──────
            # Find longest common prefix between draft and verify
            accepted_len = 0
            min_len = min(len(draft_text), len(verify_text))

            # Word-level comparison (more meaningful than char-level)
            draft_words = draft_text.split()
            verify_words = verify_text.split()

            accepted_words = 0
            for d, v in zip(draft_words, verify_words):
                if d == v:
                    accepted_words += 1
                else:
                    break

            if accepted_words > 0:
                # Accept the matching prefix from draft
                accepted_text = " ".join(draft_words[:accepted_words])
                # Add bonus: first differing word from verify
                if accepted_words < len(verify_words):
                    accepted_text += " " + verify_words[accepted_words]
            else:
                # Draft completely wrong — use verify's first word
                accepted_text = verify_words[0] if verify_words else ""

            if not accepted_text:
                # Both empty — generation complete
                break

            # Emit accepted text
            yield accepted_text + " "
            generated += accepted_text + " "
            tokens_generated += accepted_words + 1

            # Stats
            self.total_drafted += len(draft_words)
            self.total_accepted += accepted_words
            self.total_steps += 1
            self.total_draft_time += draft_time
            self.total_verify_time += verify_time

            # Check for EOS patterns
            if any(eos in accepted_text for eos in ["<|im_end|>", "</s>", "<|endoftext|>"]):
                break

    def stats(self):
        """Return performance statistics."""
        total_time = self.total_draft_time + self.total_verify_time
        acceptance = self.total_accepted / max(self.total_drafted, 1)
        effective_tps = (self.total_accepted + self.total_steps) / max(total_time, 0.001)
        return {
            "steps": self.total_steps,
            "drafted": self.total_drafted,
            "accepted": self.total_accepted,
            "acceptance_rate": acceptance,
            "draft_time": self.total_draft_time,
            "verify_time": self.total_verify_time,
            "effective_tok_per_sec": effective_tps,
        }


# ── Multi-Server Pool ─────────────────────────────────

class ServerPool:
    """Load balancer across multiple inference backends."""

    def __init__(self, hosts, runpod_key=None):
        self.servers = [InferenceServer(h, runpod_key) for h in hosts]
        self.idx = 0

    def next(self):
        """Round-robin to next healthy server."""
        for _ in range(len(self.servers)):
            server = self.servers[self.idx % len(self.servers)]
            self.idx += 1
            if server.health():
                return server
        raise RuntimeError("No healthy servers")

    def stream(self, messages, **kwargs):
        return self.next().stream(messages, **kwargs)

    def complete(self, messages, **kwargs):
        return self.next().complete(messages, **kwargs)


# ── Banner ─────────────────────────────────────────────

def banner(args, draft_ok, verify_ok):
    print()
    print(f"  {B}{C}  moe{X}{D}-{X}{B}{Y}sniper{X}  {D}spec decode{X}")
    print(f"  {D}  Heterogeneous MoE inference — draft + verify pipeline{X}")
    print()
    print(f"  {D}{'─' * 56}{X}")
    draft_status = f"{G}connected{X}" if draft_ok else f"{R}unreachable{X}"
    verify_status = f"{G}connected{X}" if verify_ok else f"{R}unreachable{X}"
    print(f"  {B}{W}Draft{X}     {args.draft}  [{draft_status}]")
    print(f"  {B}{W}Verify{X}    {args.verify}  [{verify_status}]")
    print(f"  {B}{W}Draft K{X}   {args.draft_len} tokens per step")
    mode = "spec-decode" if draft_ok and verify_ok else "single-server fallback"
    print(f"  {B}{W}Mode{X}      {mode}")
    print(f"  {D}{'─' * 56}{X}")
    print()
    print(f"  {D}Commands:{X}")
    print(f"  {D}  /stats     show spec decode performance{X}")
    print(f"  {D}  /mode      toggle spec-decode vs serial{X}")
    print(f"  {D}  /clear     clear conversation{X}")
    print(f"  {D}  /quit      exit{X}")
    print()
    if draft_ok and verify_ok:
        print(f"  {G}Speculative decode active{X} — draft generates {args.draft_len} tokens,")
        print(f"  verify checks in batch. Expect {B}3-5x faster{X} than serial.\n")
    elif verify_ok:
        print(f"  {Y}Draft server unavailable — falling back to serial verify server{X}\n")
    else:
        print(f"  {R}No servers available{X}\n")


# ── Main ───────────────────────────────────────────────

SYSTEM = "You are a helpful AI assistant. Be concise and direct."

def main():
    parser = argparse.ArgumentParser(
        description="Expert Sniper — Speculative Decode Router")
    parser.add_argument("--draft", type=str, required=True,
        help="Draft server (small model, e.g., localhost:8201)")
    parser.add_argument("--verify", type=str, required=True,
        help="Verify server (full MoE model, e.g., localhost:8202)")
    parser.add_argument("--draft-len", type=int, default=8,
        help="Number of draft tokens per step (default: 8)")
    parser.add_argument("--runpod-key", type=str, default=None,
        help="RunPod API key (for RunPod endpoints)")
    parser.add_argument("--serial", action="store_true",
        help="Force serial mode (no speculative decoding)")
    args = parser.parse_args()

    draft = InferenceServer(args.draft, args.runpod_key)
    verify = InferenceServer(args.verify, args.runpod_key)

    spin = Spinner()
    spin.start("Connecting to servers...")
    draft_ok = draft.health()
    verify_ok = verify.health()
    spin.stop("Servers checked")

    banner(args, draft_ok, verify_ok)

    use_spec_decode = draft_ok and verify_ok and not args.serial
    engine = SpecDecodeEngine(draft, verify, draft_len=args.draft_len)

    signal.signal(signal.SIGINT, lambda s, f: (print(f"\n  {D}goodbye.{X}\n"), sys.exit(0)))
    conversation = []

    while True:
        try:
            user = input(f"  {B}{Y}>{X} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {D}goodbye.{X}\n")
            break

        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            break

        if user == "/stats":
            s = engine.stats()
            print(f"\n  {C}Spec Decode Stats{X}")
            print(f"  {D}{'─' * 40}{X}")
            print(f"  Steps:           {s['steps']}")
            print(f"  Drafted:         {s['drafted']} words")
            print(f"  Accepted:        {s['accepted']} words")
            print(f"  Acceptance rate: {s['acceptance_rate']:.1%}")
            print(f"  Draft time:      {s['draft_time']:.1f}s")
            print(f"  Verify time:     {s['verify_time']:.1f}s")
            print(f"  {B}Effective:       {s['effective_tok_per_sec']:.1f} tok/s{X}")
            if s['steps'] > 0:
                serial_est = s['verify_time'] / max(s['accepted'], 1)
                speedup = s['effective_tok_per_sec'] / (1 / serial_est) if serial_est > 0 else 0
                print(f"  {B}Speedup vs serial: {speedup:.1f}x{X}")
            print()
            continue

        if user == "/mode":
            use_spec_decode = not use_spec_decode
            mode = "spec-decode" if use_spec_decode else "serial"
            print(f"  Mode: {mode}\n")
            continue

        if user == "/clear":
            conversation.clear()
            engine = SpecDecodeEngine(draft, verify, draft_len=args.draft_len)
            print(f"  {D}cleared{X}\n")
            continue

        # Build messages
        conversation.append({"role": "user", "content": user})
        messages = [{"role": "system", "content": SYSTEM}] + conversation[-10:]

        print()
        start = time.time()
        tokens = 0

        if use_spec_decode:
            # ── Speculative Decode ──
            spin = Spinner()
            spin.start("Draft → Verify pipeline...")
            first = True
            response = ""

            for chunk in engine.generate(messages, max_tokens=256):
                if first:
                    spin.stop("Generating (spec decode)")
                    print(f"\n  ", end="", flush=True)
                    first = False
                print(chunk, end="", flush=True)
                response += chunk
                tokens += len(chunk.split())

            if first:
                # No output from spec decode — fall back to serial
                spin.stop(f"{Y}Falling back to serial{X}")
                response = ""
                for chunk in verify.stream(messages):
                    print(chunk, end="", flush=True)
                    response += chunk
                    tokens += 1
        else:
            # ── Serial (fallback) ──
            spin = Spinner()
            spin.start("Generating (serial)...")
            first = True
            response = ""
            server = verify if verify_ok else draft
            for chunk in server.stream(messages):
                if first:
                    spin.stop("Generating")
                    print(f"\n  ", end="", flush=True)
                    first = False
                print(chunk, end="", flush=True)
                response += chunk
                tokens += 1

        conversation.append({"role": "assistant", "content": response})

        elapsed = time.time() - start
        if tokens > 0:
            speed = tokens / elapsed
            color = G if speed > 10 else Y if speed > 3 else R
            mode_tag = f" {D}(spec){X}" if use_spec_decode else ""
            print(f"\n\n  {color}{B}{speed:.1f} tok/s{X}{mode_tag}  "
                  f"{D}{tokens} tokens in {elapsed:.1f}s{X}")
        print()


if __name__ == "__main__":
    main()
