#!/usr/bin/env python3
"""One client for both engines: the same prompts, the same request, the same clock.

Fixed prompts of 256 / 1000 / 4096 tokens (made once with llama-server's own tokenizer -- the same GGUF as Strata's --
and saved, so every engine and every run reads exactly the same text), then for each: an OpenAI chat request,
streamed, temperature 0, presence_penalty 0, thinking off, 256 tokens out. Timed on the client: first token
(prompt reading) and the rate from the first token to the last (writing), so neither engine's own counters are
trusted over the other's.

    bench.py --url http://127.0.0.1:5851 --engine llama --label llama-24000 --out results/llama-24000.json
"""
import argparse, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "..", "results", "prompts.json")
ASK = "\n\nExplain what this module does, function by function."


def post(url, body, timeout=900):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def make_prompts(url, sizes):
    """Exact-length code text from llama-server's tokenizer: a long made-up module, cut at N tokens."""
    lines = []
    for i in range(3000):
        a, b, m = 7 + i * 13 % 97, 3 + i * 29 % 89, 101 + i * 17 % 997
        lines.append(f"def transform_{i:04d}(x: int) -> int:\n    \"\"\"Step {i}.\"\"\"\n    return (x * {a} + {b}) % {m}\n")
    text = "\n".join(lines)
    toks = json.load(post(url + "/tokenize", {"content": text}))["tokens"]
    out = {}
    for n in sizes:
        out[str(n)] = json.load(post(url + "/detokenize", {"tokens": toks[:n]}))["content"] + ASK
    return out


def one(url, engine, content, maximum):
    body = {"model": "x", "messages": [{"role": "user", "content": content}], "stream": True,
            "temperature": 0, "presence_penalty": 0, "max_tokens": maximum, "stream_options": {"include_usage": True}}
    if engine == "strata":
        body["reasoning_effort"] = "none"
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time(); first = last = None; text = []; usage = None; chunks = 0
    with post(url + "/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            j = json.loads(data)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece:
                    now = time.time()
                    first = first or now
                    last = now
                    chunks += 1
                    text.append(piece)
    n = (usage or {}).get("completion_tokens") or chunks
    return {"ttft_s": (first - t0) if first else None, "tokens": n, "chunks": chunks,
            "decode_tok_s": (n - 1) / (last - first) if first and last and last > first else None,
            "elapsed_s": time.time() - t0, "text": "".join(text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--engine", choices=["llama", "strata"], required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", default="256,1000,4096"); ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(",")]
    if not os.path.exists(PROMPTS):
        if a.engine != "llama":
            sys.exit("no prompts yet: the first run must be a llama-server one (it has /tokenize)")
        json.dump(make_prompts(a.url, sizes), open(PROMPTS, "w"), indent=1)
    prompts = json.load(open(PROMPTS))
    one(a.url, a.engine, "Say hello.", 8)   # warm-up, not counted
    rows = []
    for n in sizes:
        for k in range(a.runs):
            r = one(a.url, a.engine, prompts[str(n)], a.max_tokens)
            r.update(size=n, run=k + 1, label=a.label)
            rows.append(r)
            print(f"{a.label} {n:5d} run {k + 1}: first token {r['ttft_s']:.2f} s, writing {r['decode_tok_s'] or 0:.1f} tok/s, "
                  f"{r['tokens']} tokens", flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
