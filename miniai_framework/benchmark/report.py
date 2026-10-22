"""
Report Exporter to Markdown, Console, and JSON tables.
"""

from typing import List
from miniai_framework.benchmark.profiler import BenchmarkMetrics


class BenchmarkReporter:
    @staticmethod
    def generate_console_summary(metrics_list: List[BenchmarkMetrics]) -> str:
        header = f"{'Backend':<15} | {'Tok/sec':<10} | {'TTFT (ms)':<10} | {'Avg Lat (ms)':<12} | {'P99 (ms)':<10} | {'GFLOPS':<8}"
        divider = "-" * len(header)
        lines = [header, divider]

        for m in metrics_list:
            lines.append(
                f"{m.backend_name:<15} | {m.tokens_per_second:<10.2f} | {m.ttft_ms:<10.2f} | "
                f"{m.avg_latency_ms:<12.2f} | {m.p99_latency_ms:<10.2f} | {m.estimated_gflops:<8.2f}"
            )
        lines.append(divider)
        return "\n".join(lines)

    @staticmethod
    def generate_markdown(metrics_list: List[BenchmarkMetrics]) -> str:
        md = ["# Mini AI Framework - Benchmark Results\n"]
        md.append("| Backend | Tokens / Sec | TTFT (ms) | Avg Latency (ms) | P99 Latency (ms) | GFLOPS | Memory (MB) |")
        md.append("|---|---|---|---|---|---|---|")

        for m in metrics_list:
            md.append(
                f"| `{m.backend_name}` | **{m.tokens_per_second:.2f}** | {m.ttft_ms:.2f} | "
                f"{m.avg_latency_ms:.2f} | {m.p99_latency_ms:.2f} | {m.estimated_gflops:.2f} | {m.memory_footprint_mb:.2f} |"
            )

        return "\n".join(md)
