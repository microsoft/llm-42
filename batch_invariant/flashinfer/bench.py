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
split_sizes = [32, 64, 128, 256, 512, 1024, 2048]

def get_batch_sizes(model, num_heads, num_kv_heads):
    batch_sizes = []
    return [1, 2, 4, 8, 16] if num_heads == num_kv_heads else [1, 2, 4, 8, 16, 32, 64]

# Data storage for plotting
results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
optimal_splits = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

print("model;cl;bs;split_size;fi_default;fi_deterministic;latency;overhead")
for model in utils.attn_configs:
    num_heads = utils.attn_configs[model]['num_heads']
    num_kv_heads = utils.attn_configs[model]['num_kv_heads']
    head_dim = utils.attn_configs[model]['head_dim']
    batch_sizes = get_batch_sizes(model, num_heads, num_kv_heads)
    
    for cl in context_lens:
        for bs in batch_sizes:
            # Test default (non-deterministic) once
            try:
                fi_default = fibench.do_flashinfer_decode_paged(bs, cl, num_heads, num_kv_heads, head_dim, 16, use_fixed_split=False, disable_split=False)
            except Exception as e:
                print(f"# Error testing default for {model} cl={cl} bs={bs}: {e}")
                fi_default = -1
            
            # Test different split sizes
            best_latency = float('inf')
            best_split_size = None
            
            for split_size in split_sizes:
                try:
                    # Modify fibench to accept split_size parameter
                    fi_deterministic = fibench.do_flashinfer_decode_paged_with_split(bs, cl, num_heads, num_kv_heads, head_dim, 16, split_size)
                except Exception as e:
                    print(f"# Error testing {model} cl={cl} bs={bs} split={split_size}: {e}")
                    fi_deterministic = -1
                    
                overhead = round(fi_deterministic / fi_default, 3) if fi_default > 0 and fi_deterministic > 0 else -1
                print(f"{model};{cl};{bs};{split_size};{fi_default};{fi_deterministic};{fi_deterministic};{overhead}")
                
                # Store results for plotting
                results[model][cl][bs][split_size] = {
                    'fi_default': fi_default,
                    'fi_deterministic': fi_deterministic,
                    'overhead': overhead
                }
                
                # Track best split size
                if fi_deterministic > 0 and fi_deterministic < best_latency:
                    best_latency = fi_deterministic
                    best_split_size = split_size
            
            # Store optimal split size
            optimal_splits[model][cl][bs] = {
                'split_size': best_split_size,
                'latency': best_latency,
                'fi_default': fi_default,
                'overhead': round(best_latency / fi_default, 3) if fi_default > 0 else -1
            }
    print()

# Generate plots
for model in utils.attn_configs:
    num_heads = utils.attn_configs[model]['num_heads']
    num_kv_heads = utils.attn_configs[model]['num_kv_heads']
    batch_sizes = get_batch_sizes(model, num_heads, num_kv_heads)
    
    # Plot 1: Latency comparison across split sizes for each context length
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{model} - Latency vs Split Size', fontsize=16)
    
    for idx, cl in enumerate(context_lens):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        for bs in batch_sizes:
            split_list = []
            latency_list = []
            
            for split_size in split_sizes:
                if split_size in results[model][cl][bs]:
                    split_list.append(split_size)
                    latency_list.append(results[model][cl][bs][split_size]['fi_deterministic'])
            
            if split_list:
                ax.plot(split_list, latency_list, marker='o', label=f'BS={bs}', linewidth=2)
        
        # Add baseline (non-deterministic)
        if batch_sizes and split_sizes[0] in results[model][cl][batch_sizes[0]]:
            baseline = results[model][cl][batch_sizes[0]][split_sizes[0]]['fi_default']
            ax.axhline(y=baseline, color='r', linestyle='--', alpha=0.5, label='Non-det baseline')
        
        ax.set_xlabel('Split Size', fontsize=10)
        ax.set_ylabel('Latency (ms)', fontsize=10)
        ax.set_title(f'Context Length: {cl}', fontsize=12)
        ax.set_xscale('log', base=2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'fi_split_analysis_{model}_latency_vs_split.pdf', dpi=1200, bbox_inches='tight')
    print(f"Saved plot: fi_split_analysis_{model}_latency_vs_split.pdf")
    plt.close()
    
    # Plot 2: Overhead for different split sizes
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{model} - Overhead vs Split Size', fontsize=16)
    
    for idx, cl in enumerate(context_lens):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        for bs in batch_sizes:
            split_list = []
            overhead_list = []
            
            for split_size in split_sizes:
                if split_size in results[model][cl][bs] and results[model][cl][bs][split_size]['overhead'] > 0:
                    split_list.append(split_size)
                    overhead_list.append(results[model][cl][bs][split_size]['overhead'])
            
            if split_list:
                ax.plot(split_list, overhead_list, marker='o', label=f'BS={bs}', linewidth=2)
        
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='No Overhead')
        ax.set_xlabel('Split Size', fontsize=10)
        ax.set_ylabel('Overhead (Det / Non-det)', fontsize=10)
        ax.set_title(f'Context Length: {cl}', fontsize=12)
        ax.set_xscale('log', base=2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'fi_split_analysis_{model}_overhead_vs_split.pdf', dpi=1200, bbox_inches='tight')
    print(f"Saved plot: fi_split_analysis_{model}_overhead_vs_split.pdf")
    plt.close()
    
    # Plot 3: Optimal split size heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model} - Optimal Configuration Analysis', fontsize=16)
    
    # Heatmap for optimal split sizes
    split_matrix = np.zeros((len(context_lens), len(batch_sizes)))
    overhead_matrix = np.zeros((len(context_lens), len(batch_sizes)))
    
    for i, cl in enumerate(context_lens):
        for j, bs in enumerate(batch_sizes):
            if bs in optimal_splits[model][cl]:
                split_matrix[i, j] = optimal_splits[model][cl][bs]['split_size'] or 0
                overhead_matrix[i, j] = optimal_splits[model][cl][bs]['overhead'] or 0
    
    im1 = ax1.imshow(split_matrix, cmap='viridis', aspect='auto')
    ax1.set_xticks(np.arange(len(batch_sizes)))
    ax1.set_yticks(np.arange(len(context_lens)))
    ax1.set_xticklabels(batch_sizes)
    ax1.set_yticklabels(context_lens)
    ax1.set_xlabel('Batch Size', fontsize=12)
    ax1.set_ylabel('Context Length', fontsize=12)
    ax1.set_title('Optimal Split Size', fontsize=14)
    
    # Add text annotations
    for i in range(len(context_lens)):
        for j in range(len(batch_sizes)):
            val = split_matrix[i, j]
            if val > 0:
                text = ax1.text(j, i, int(val), ha="center", va="center", color="w", fontsize=8)
    
    plt.colorbar(im1, ax=ax1, label='Split Size')
    
    # Heatmap for overhead with optimal split
    im2 = ax2.imshow(overhead_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0.9, vmax=1.5)
    ax2.set_xticks(np.arange(len(batch_sizes)))
    ax2.set_yticks(np.arange(len(context_lens)))
    ax2.set_xticklabels(batch_sizes)
    ax2.set_yticklabels(context_lens)
    ax2.set_xlabel('Batch Size', fontsize=12)
    ax2.set_ylabel('Context Length', fontsize=12)
    ax2.set_title('Overhead with Optimal Split', fontsize=14)
    
    # Add text annotations
    for i in range(len(context_lens)):
        for j in range(len(batch_sizes)):
            val = overhead_matrix[i, j]
            if val > 0:
                text = ax2.text(j, i, f"{val:.2f}", ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im2, ax=ax2, label='Overhead Ratio')
    
    plt.tight_layout()
    plt.savefig(f'fi_split_analysis_{model}_optimal_heatmap.pdf', dpi=1200, bbox_inches='tight')
    print(f"Saved plot: fi_split_analysis_{model}_optimal_heatmap.pdf")
    plt.close()
    
    # Plot 4: Bar chart comparing best split size performance
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{model} - Performance with Optimal Split Size', fontsize=16)
    
    for idx, cl in enumerate(context_lens):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        bs_list = []
        default_list = []
        optimal_list = []
        
        for bs in batch_sizes:
            if bs in optimal_splits[model][cl]:
                bs_list.append(bs)
                default_list.append(optimal_splits[model][cl][bs]['fi_default'])
                optimal_list.append(optimal_splits[model][cl][bs]['latency'])
        
        x = np.arange(len(bs_list))
        width = 0.35
        
        ax.bar(x - width/2, default_list, width, label='Non-deterministic', alpha=0.8)
        ax.bar(x + width/2, optimal_list, width, label='Deterministic (Optimal Split)', alpha=0.8)
        
        ax.set_xlabel('Batch Size', fontsize=10)
        ax.set_ylabel('Latency (ms)', fontsize=10)
        ax.set_title(f'Context Length: {cl}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(bs_list)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'fi_split_analysis_{model}_optimal_comparison.pdf', dpi=1200, bbox_inches='tight')
    print(f"Saved plot: fi_split_analysis_{model}_optimal_comparison.pdf")
    plt.close()

print("\nAll plots generated successfully!")

# Print summary of optimal split sizes
print("\n" + "="*80)
print("OPTIMAL SPLIT SIZE SUMMARY")
print("="*80)
for model in utils.attn_configs:
    num_heads = utils.attn_configs[model]['num_heads']
    num_kv_heads = utils.attn_configs[model]['num_kv_heads']
    batch_sizes = get_batch_sizes(model, num_heads, num_kv_heads)
    
    print(f"\n{model}:")
    print(f"{'Context Length':<15} {'Batch Size':<12} {'Optimal Split':<15} {'Overhead':<10}")
    print("-" * 60)
    
    for cl in context_lens:
        for bs in batch_sizes:
            if bs in optimal_splits[model][cl]:
                opt = optimal_splits[model][cl][bs]
                print(f"{cl:<15} {bs:<12} {opt['split_size']:<15} {opt['overhead']:<10.3f}")
