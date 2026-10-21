#!/usr/bin/env python3
"""Minimal BGE-M3 embedding endpoint for the v1 KoGovDoc scripts.

run_retrieval_eval.py and run_bccs_eval.py call an OpenAI-style /embeddings API
({"model": ..., "input": [...]} -> {"data": [{"index", "embedding"}]}). The v1 run used a
BGE-M3 server on the lab network; this script stands in for it with sentence-transformers.

Usage:
    python scripts/eval_kogovdoc/serve_bge_m3.py --port 8001 --device cuda
    # then pass --embedding-api-url http://localhost:8001/embeddings (the scripts' default)
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser(description="BGE-M3 /embeddings endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model = SentenceTransformer("BAAI/bge-m3", device=args.device)
    model.max_seq_length = 8192

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            emb = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True)
            out = json.dumps(
                {"data": [{"index": i, "embedding": e.tolist()} for i, e in enumerate(emb)]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    print(f"Serving BAAI/bge-m3 on http://{args.host}:{args.port}/embeddings ({args.device})")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
