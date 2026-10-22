"""
Lightweight HTTP API Server & Web Dashboard Backend for Mini AI Framework.
Built using standard Python http.server for zero-dependency execution.
Supports REST APIs and Server-Sent Events (SSE) for token streaming.
"""

import http.server
import socketserver
import json
import urllib.parse
import sys
import os
import time
from typing import Dict, Any

# Ensure parent path in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from miniai_framework import ModelConfig, RuntimeConfig, BackendType
from miniai_framework.tokenizer import BPETokenizer
from miniai_framework.models import ModelLoader, NanoTransformer
from miniai_framework.engine import NumPyInferenceEngine, MAXInferenceEngine, MojoInferenceEngine
from miniai_framework.benchmark import BackendComparator

# Global framework instances
tokenizer = BPETokenizer()
model = NanoTransformer(config=ModelConfig())
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class FrameworkRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path == "/api/info":
            self.send_json_response({
                "framework": "Mini AI Framework",
                "version": "0.1.0",
                "model": model.config.to_dict(),
                "vocab_size": tokenizer.vocab_size(),
                "parameters": model.get_num_parameters(),
                "memory_mb": round(model.get_memory_footprint_mb(), 2),
                "backends": [b.value for b in BackendType]
            })

        elif path == "/api/tokenize":
            query = urllib.parse.parse_qs(parsed_url.query)
            text = query.get("text", ["Building AI Framework with MAX and Mojo"])[0]
            pieces = tokenizer.tokenize_with_metadata(text)
            self.send_json_response({
                "text": text,
                "token_ids": [p.id for p in pieces],
                "pieces": [
                    {
                        "id": p.id,
                        "text": p.text,
                        "start": p.start_char,
                        "end": p.end_char,
                        "is_special": p.is_special
                    } for p in pieces
                ]
            })

        elif path == "/api/benchmark":
            query = urllib.parse.parse_qs(parsed_url.query)
            prompt = query.get("prompt", ["Comparative benchmark test"])[0]
            max_tokens = int(query.get("max_tokens", [40])[0])

            metrics_list = BackendComparator.compare_all(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_tokens
            )

            self.send_json_response({
                "results": [m.to_dict() for m in metrics_list]
            })

        elif path == "/api/generate_stream":
            self.handle_stream(parsed_url.query)

        else:
            # Default static file serving
            if path == "/":
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        data = json.loads(body) if body else {}

        if parsed_url.path == "/api/generate":
            prompt = data.get("prompt", "Artificial intelligence")
            backend_str = data.get("backend", "numpy")
            max_tokens = int(data.get("max_new_tokens", 40))
            temp = float(data.get("temperature", 0.7))

            backend_type = BackendType(backend_str)
            if backend_type == BackendType.MAX_ENGINE:
                engine = MAXInferenceEngine(model, tokenizer)
            elif backend_type == BackendType.MOJO_SIMD:
                engine = MojoInferenceEngine(model, tokenizer)
            else:
                engine = NumPyInferenceEngine(model, tokenizer)

            cfg = RuntimeConfig(backend=backend_type, max_new_tokens=max_tokens, temperature=temp)
            result = engine.generate(prompt, cfg)

            self.send_json_response({
                "text": result.text,
                "tokens": result.tokens,
                "total_time_ms": round(result.total_time_ms, 2),
                "ttft_ms": round(result.ttft_ms, 2),
                "tokens_per_sec": round(result.tokens_per_second, 2),
                "avg_latency_ms": round(result.avg_latency_per_token_ms, 2)
            })

        else:
            self.send_error(404, "Endpoint not found")

    def handle_stream(self, query_string: str):
        query = urllib.parse.parse_qs(query_string)
        prompt = query.get("prompt", ["The future of AI"])[0]
        backend_str = query.get("backend", ["numpy"])[0]
        max_tokens = int(query.get("max_tokens", [40])[0])
        temp = float(query.get("temperature", [0.7])[0])

        backend_type = BackendType(backend_str)
        if backend_type == BackendType.MAX_ENGINE:
            engine = MAXInferenceEngine(model, tokenizer)
        elif backend_type == BackendType.MOJO_SIMD:
            engine = MojoInferenceEngine(model, tokenizer)
        else:
            engine = NumPyInferenceEngine(model, tokenizer)

        cfg = RuntimeConfig(backend=backend_type, max_new_tokens=max_tokens, temperature=temp)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        try:
            for chunk in engine.generate_stream(prompt, cfg):
                payload = json.dumps({
                    "token_id": chunk.token_id,
                    "text": chunk.token_text,
                    "step": chunk.step,
                    "latency_ms": round(chunk.latency_ms, 2),
                    "total_time_ms": round(chunk.total_time_ms, 2),
                    "is_finished": chunk.is_finished
                })
                self.wfile.write(f"data: {payload}\n\n".encode('utf-8'))
                self.wfile.flush()
                time.sleep(0.02)
        except (ConnectionResetError, BrokenPipeError):
            pass

    def send_json_response(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


def start_server(host: str = "127.0.0.1", port: int = 8000):
    with socketserver.TCPServer((host, port), FrameworkRequestHandler) as httpd:
        print(f"Mini AI Framework Server running on http://{host}:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    start_server()
