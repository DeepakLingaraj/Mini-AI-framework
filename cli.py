"""
Mini AI Framework Command-Line Interface (CLI).
Supports model loading, tokenization, real-time generation, benchmarking, and web server startup.
"""

import argparse
import sys
import os
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from miniai_framework import ModelConfig, RuntimeConfig, BackendType
from miniai_framework.tokenizer import BPETokenizer
from miniai_framework.models import ModelLoader, NanoTransformer
from miniai_framework.engine import NumPyInferenceEngine, MAXInferenceEngine, MojoInferenceEngine
from miniai_framework.benchmark import BackendComparator, BenchmarkReporter


def handle_tokenize(args):
    print(f"\n--- Tokenizer Visual Inspector ---")
    tokenizer = BPETokenizer()
    text = args.text
    pieces = tokenizer.tokenize_with_metadata(text)
    token_ids = [p.id for p in pieces]

    print(f"Input Text: {text!r}")
    print(f"Token IDs:  {token_ids}")
    print(f"Vocab Size: {tokenizer.vocab_size()}")
    print("\nDetailed Token Breakdown:")
    for i, p in enumerate(pieces):
        print(f"  [{i:02d}] ID: {p.id:<5} Text: {p.text!r:<15} Char Span: [{p.start_char}:{p.end_char}]")


def handle_generate(args):
    print(f"\n=== Mini AI Framework Generation ===")
    print(f"Backend: {args.backend.upper()} | Model: {args.model}")
    print(f"Prompt:  {args.prompt!r}\n")

    tokenizer = BPETokenizer()
    config = ModelConfig(name=args.model)
    model = NanoTransformer(config=config)

    backend_type = BackendType(args.backend.lower())
    if backend_type == BackendType.MAX_ENGINE:
        engine = MAXInferenceEngine(model, tokenizer)
    elif backend_type == BackendType.MOJO_SIMD:
        engine = MojoInferenceEngine(model, tokenizer)
    else:
        engine = NumPyInferenceEngine(model, tokenizer)

    runtime_cfg = RuntimeConfig(
        backend=backend_type,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    sys.stdout.write("Output: ")
    sys.stdout.flush()

    start_time = time.perf_counter()
    chunks = []
    for chunk in engine.generate_stream(args.prompt, runtime_cfg):
        chunks.append(chunk)
        sys.stdout.write(chunk.token_text)
        sys.stdout.flush()
    sys.stdout.write("\n\n")

    if chunks:
        total_time = chunks[-1].total_time_ms
        tps = len(chunks) / (total_time / 1000.0) if total_time > 0 else 0
        print(f"[Stats] Generated {len(chunks)} tokens in {total_time:.2f}ms ({tps:.2f} tokens/sec) | TTFT: {chunks[0].latency_ms:.2f}ms")


def handle_benchmark(args):
    print("\n=== Mini AI Framework Multi-Backend Benchmark ===")
    print("Benchmarking backends: NumPy SIMD, Modular MAX Engine, Mojo SIMD Kernels...\n")

    tokenizer = BPETokenizer()
    model = NanoTransformer(config=ModelConfig())

    metrics_list = BackendComparator.compare_all(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens
    )

    print(BenchmarkReporter.generate_console_summary(metrics_list))


def handle_serve(args):
    print(f"\n=== Launching Mini AI Framework Web Server ===")
    print(f"Server starting on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.\n")

    from web_dashboard.server import start_server
    start_server(host=args.host, port=args.port)


def main():
    parser = argparse.ArgumentParser(description="Mini AI Framework CLI (ApexMini MAX-Mojo Engine)")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")

    # Tokenize command
    token_parser = subparsers.add_parser("tokenize", help="Tokenize text and inspect subword chunks")
    token_parser.add_argument("--text", type=str, default="Building lightweight AI framework with MAX and Mojo", help="Text to tokenize")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Run LLM model inference and stream tokens")
    gen_parser.add_argument("--prompt", type=str, default="The future of high performance AI framework is", help="Prompt text")
    gen_parser.add_argument("--model", type=str, default="MiniLlama-15M", help="Model name")
    gen_parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "max_engine", "mojo_simd"], help="Execution backend")
    gen_parser.add_argument("--max-tokens", type=int, default=40, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_parser.add_argument("--top-k", type=int, default=40, help="Top-K sampling parameter")
    gen_parser.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus sampling parameter")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run multi-backend benchmark comparison")
    bench_parser.add_argument("--prompt", type=str, default="Benchmarking framework execution speed", help="Benchmark prompt")
    bench_parser.add_argument("--max-tokens", type=int, default=50, help="Tokens per run")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Launch interactive Web Dashboard & REST API")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")

    args = parser.parse_args()

    if args.command == "tokenize":
        handle_tokenize(args)
    elif args.command == "generate":
        handle_generate(args)
    elif args.command == "benchmark":
        handle_benchmark(args)
    elif args.command == "serve":
        handle_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
