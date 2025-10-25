import torch
import flashinferbench as fibench
#import flashattentionbench as fabench
#import vllmbench
import sys
import utils
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

context_lens = [1024, 2048, 4096, 8192, 16384, 32768]

def get_batch_sizes(model, num_heads, num_kv_heads):
    batch_sizes = []
    return [1, 2, 4, 8, 16] if num_heads == num_kv_heads else [1, 2, 4, 8, 16, 32, 64]

# Data storage for plotting
results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

print("model;cl;bs;fi_default;fi_deterministic;fi_deterministic_no_split;overhead")
for model in utils.attn_configs:
    num_heads = utils.attn_configs[model]['num_heads']
    num_kv_heads = utils.attn_configs[model]['num_kv_heads']
    head_dim = utils.attn_configs[model]['head_dim']
    batch_sizes = get_batch_sizes(model, num_heads, num_kv_heads)
    fa_latency, fa_paged_latency, fi_latency, fi_paged_latency = -1, -1, -1, -1
    for cl in context_lens:
        for bs in batch_sizes:
            #fa_latency = fabench.do_flashattention_decode(bs, cl, num_heads, num_kv_heads, head_dim)
            #fa_paged_latency = fabench.do_flashattention_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, 256)
            #fi_latency = fibench.do_flashinfer_decode(bs, cl, num_heads, num_kv_heads, head_dim)
            fi_default = fibench.do_flashinfer_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, 16, use_fixed_split=False, disable_split=False)
            fi_deterministic = fibench.do_flashinfer_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, 16, use_fixed_split=True, disable_split=False)
            fi_deterministic_no_split = fibench.do_flashinfer_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, 16, use_fixed_split=True, disable_split=True)
            overhead = round(fi_deterministic / fi_default, 3) if fi_default > 0 and fi_deterministic > 0 else -1
            print(f"{model};{cl};{bs};{fi_default};{fi_deterministic};{fi_deterministic_no_split};{overhead}")
            
            # Store results for plotting
            results[model][cl][bs] = {
                'fi_default': fi_default,
                'fi_deterministic': fi_deterministic,
                'fi_deterministic_no_split': fi_deterministic_no_split,
                'overhead': overhead
            }
    print()

# Generate plots
for model in utils.attn_configs:
    num_heads = utils.attn_configs[model]['num_heads']
    num_kv_heads = utils.attn_configs[model]['num_kv_heads']
    batch_sizes = get_batch_sizes(model, num_heads, num_kv_heads)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{model} Benchmark Results', fontsize=16)
    
    for idx, cl in enumerate(context_lens):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        bs_list = []
        default_list = []
        deterministic_list = []
        deterministic_no_split_list = []
        
        for bs in batch_sizes:
            if bs in results[model][cl]:
                bs_list.append(bs)
                default_list.append(results[model][cl][bs]['fi_default'])
                deterministic_list.append(results[model][cl][bs]['fi_deterministic'])
                deterministic_no_split_list.append(results[model][cl][bs]['fi_deterministic_no_split'])
        
        x = np.arange(len(bs_list))
        width = 0.25
        
        ax.bar(x - width, default_list, width, label='Default', alpha=0.8)
        ax.bar(x, deterministic_list, width, label='Deterministic', alpha=0.8)
        ax.bar(x + width, deterministic_no_split_list, width, label='Deterministic No Split', alpha=0.8)
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'Context Length: {cl}')
        ax.set_xticks(x)
        ax.set_xticklabels(bs_list)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model}_latency_comparison.pdf', dpi=1200, bbox_inches='tight')
    print(f"Saved plot: {model}_latency_comparison.pdf")
    plt.close()
    
    # Create overhead plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for cl in context_lens:
        bs_list = []
        overhead_list = []
        
        for bs in batch_sizes:
            if bs in results[model][cl] and results[model][cl][bs]['overhead'] > 0:
                bs_list.append(bs)
                overhead_list.append(results[model][cl][bs]['overhead'])
        
        if bs_list:
            ax.plot(bs_list, overhead_list, marker='o', label=f'CL={cl}', linewidth=2)
    
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Overhead (Deterministic / Default)', fontsize=12)
    ax.set_title(f'{model} - Deterministic Overhead', fontsize=14)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No Overhead')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model}_overhead.pdf', dpi=1200, bbox_inches='tight')
    print(f"Saved plot: {model}_overhead.pdf")
    plt.close()

print("\nAll plots generated successfully!")