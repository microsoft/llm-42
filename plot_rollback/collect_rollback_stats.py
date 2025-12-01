#!/usr/bin/env python3
"""Collect rollback stats from server. Usage: python collect_rollback_stats.py"""
import argparse, json, re, time, requests
from datetime import datetime


def parse_metrics(text):
    metrics = {}
    for name in ["sglang:num_rollbacks_total", "sglang:tokens_rolled_back_total", "sglang:num_requests_total"]:
        if m := re.search(rf'{re.escape(name)}\{{[^}}]*\}}\s+([\d.]+)', text):
            metrics[name] = float(m.group(1))
    return metrics


def collect_stats(url, interval, duration, output):
    print(f"Collecting from {url}/metrics every {interval}s")
    stats, start = [], time.time()
    
    try:
        while True:
            elapsed = time.time() - start
            if duration > 0 and elapsed > duration:
                break
            
            try:
                r = requests.get(f"{url}/metrics", timeout=5)
                if r.status_code == 200:
                    m = parse_metrics(r.text)
                    rollback_rate = m['sglang:num_rollbacks_total'] / m['sglang:num_requests_total'] if m.get('sglang:num_requests_total', 0) > 0 else 0
                    
                    record = {'elapsed': elapsed, 'rollback_rate': rollback_rate, **m}
                    stats.append(record)
                    
                    print(f"[{elapsed:.0f}s] Rollbacks: {m.get('sglang:num_rollbacks_total', 0):.0f}, Rate: {rollback_rate:.4f}")
            except Exception as e:
                print(f"Error: {e}")
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped")
    
    if stats:
        with open(output, 'w') as f:
            json.dump(stats, f)
        print(f"✓ Saved {len(stats)} points to {output}")
        print(f"Final rollbacks: {stats[-1].get('sglang:num_rollbacks_total', 0):.0f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:30000")
    p.add_argument("--interval", type=float, default=5)
    p.add_argument("--duration", type=float, default=0)
    p.add_argument("--output", default="rollback_stats.json")
    args = p.parse_args()
    collect_stats(args.url, args.interval, args.duration, args.output)
