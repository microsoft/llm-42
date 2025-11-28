#!/usr/bin/env python3
"""Add rollback metrics to SGLang. Usage: python add_rollback_metrics.py"""
import os, sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_FILE = f"{REPO_ROOT}/python/sglang/srt/metrics/collector.py"
WORKER_FILE = f"{REPO_ROOT}/python/sglang/srt/detinfer/det_verify_worker.py"

# Add metrics to collector.py
with open(METRICS_FILE, 'r') as f:
    content = f.read()

if 'num_rollbacks_total' not in content:
    marker = '        self.num_aborted_requests_total = Counter(\n            name="sglang:num_aborted_requests_total",\n            documentation="Number of requests aborted.",\n            labelnames=labels.keys(),\n        )'
    
    metrics = '''
        self.num_rollbacks_total = Counter(
            name="sglang:num_rollbacks_total",
            documentation="Total rollback events.",
            labelnames=labels.keys(),
        )
        self.tokens_rolled_back_total = Counter(
            name="sglang:tokens_rolled_back_total",
            documentation="Total tokens rolled back.",
            labelnames=labels.keys(),
        )'''
    
    content = content.replace(marker, marker + metrics)
    with open(METRICS_FILE, 'w') as f:
        f.write(content)
    print("✓ Added metrics to collector.py")
else:
    print("✓ Metrics already exist")

# Add tracking to det_verify_worker.py
with open(WORKER_FILE, 'r') as f:
    content = f.read()

if 'num_rollbacks_total' not in content:
    # Add metrics_collector attribute
    content = content.replace(
        '        self.target_worker = target_worker',
        '        self.target_worker = target_worker\n        self.metrics_collector = getattr(target_worker, "metrics_collector", None)'
    )
    
    # Add tracking in _handle_kv_cache_rollback
    marker = '            if info is None or info[1] == 0:\n                continue\n            \n            mismatch_pos, tokens_rolled_back = info'
    
    tracking = '''            if info is None or info[1] == 0:
                continue
            
            # Track metrics
            if self.metrics_collector:
                self.metrics_collector.num_rollbacks_total.labels(**self.metrics_collector.labels).inc()
                self.metrics_collector.tokens_rolled_back_total.labels(**self.metrics_collector.labels).inc(info[1])
            
            mismatch_pos, tokens_rolled_back = info'''
    
    content = content.replace(marker, tracking)
    with open(WORKER_FILE, 'w') as f:
        f.write(content)
    print("✓ Added tracking to det_verify_worker.py")
else:
    print("✓ Tracking already exists")

print("\nUsage: Start server with --enable-metrics, then:")
print("  curl http://localhost:30000/metrics | grep rollback")
