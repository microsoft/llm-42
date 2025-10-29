import unittest

import torch
import os
os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = "1"

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ServerArgs
from sglang.test.test_utils import CustomTestCase
from sglang.srt.layers import dp_attention

import time


class MockModelRunner:
    def __init__(
        self,
        num_heads,
        num_heads_kv,
        head_dim,
        hidden_size,
        deterministic=0,
        max_batch_size=512, # Max batch size for the test.
        max_context_len=32768, # Total tokens(prefix + extend + decode) in the test should not exceed this length.
        page_size=1,
    ):
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.kv_cache_dtype = torch.bfloat16
        attention_arch = AttentionArch.MHA
        self.is_hybrid = False
        hf_config = type("HFConfig", (), {"architectures": ["Qwen3ForCausalLM"]})
        self.model_config = type(
            "ModelConfig",
            (),
            {
                "context_len": max_context_len,
                "is_multimodal": False,
                "attention_arch": attention_arch,
                "num_attention_heads": num_heads,
                "hidden_size": hidden_size,
                "get_num_kv_heads": lambda self: num_heads_kv,
                "is_encoder_decoder": False,
                "hf_config": hf_config,
                "head_dim": head_dim,
            },
        )
        self.sliding_window_size = None
        self.device = self.device
        # Create a large enough req_to_token_pool to fit the test usage.
        self.req_to_token_pool = type(
            "TokenPool",
            (),
            {
                # A typical max_bs * max_context_len for cuda graph decode
                "size": max_batch_size,
                # Add req_to_token attribute
                "req_to_token": torch.zeros(
                    max_batch_size,
                    max_context_len,
                    dtype=torch.int32,
                    device=self.device,
                ),
            },
        )
        self.token_to_kv_pool_allocator = None
        self.page_size = page_size
        max_total_num_tokens = max_context_len * 2
        self.token_to_kv_pool = MHATokenToKVPool(
            size=max_total_num_tokens,
            page_size=page_size,
            dtype=self.dtype,
            head_num=num_heads_kv,
            head_dim=head_dim,
            layer_num=1,  # only consider layer=1 for unit test
            device=self.device,
            enable_memory_saver=False,
        )
        self.server_args = type("ServerArgs", (), {})()
        self.server_args.device = self.device
        self.server_args.kv_cache_dtype = "bfloat16"
        self.server_args.enable_deterministic_inference = deterministic
        self.server_args.speculative_eagle_topk = None
        self.server_args.speculative_num_draft_tokens = None
        # DP stuff
        self.server_args.enable_dp_attention = False
        self.server_args.tp_size = 1
        self.server_args.dp_size = 1
        self.server_args.moe_dense_tp_size = 1
        self.server_args.pp_size = 1

        dp_attention._ATTN_TP_SIZE = 1

        #initialize_dp_attention(self.server_args, self.model_config)
        # Required by torch native backend
        #self.server_args = ServerArgs(model_path="fake_model_path")

# Llama3-8B dimensions
INTERMEDIATE_SIZE = 14336
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = HEAD_DIM * NUM_HEADS

class BenchFlashAttentionBackend:
    def __init__(self, deterministic=0):
        self.deterministic = deterministic

    def setUp(self):
        # Test parameters
        self.num_heads = NUM_HEADS
        self.head_dim = HEAD_DIM
        self.num_heads_kv = NUM_KV_HEADS
        self.hidden_size = HIDDEN_SIZE
        self.device = "cuda"
        self.dtype = torch.bfloat16

    def _init_model_runner(self, page_size=1):
        self.model_runner = MockModelRunner(
            page_size=page_size,
            num_heads=self.num_heads,
            num_heads_kv=self.num_heads_kv,
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            deterministic=self.deterministic,
        )

        self.backend = FlashInferAttnBackend(self.model_runner)
        self.ref_backend = TorchNativeAttnBackend(self.model_runner)
        self.model_runner.model_config.num_attention_heads = self.num_heads

    def _mock_write_to_req_to_token_pool(self, batch_size, seq_len, page_size):
        # if page_size > 1, the token pool stores the index to the page.
        # so we need to multiply the index by page_size.
        self.req_to_token = (
            (torch.arange(0, batch_size, dtype=torch.int32, device=self.device)[:, None]
            * seq_len
            + torch.arange(0, seq_len, dtype=torch.int32, device=self.device)[None, :])
            * page_size
        )
        self.model_runner.req_to_token_pool.req_to_token[:batch_size, :seq_len] = (
            self.req_to_token
        )

    def _create_attention_layer(self):
        """Create attention layer for testing."""
        return RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=1.0,
            num_kv_heads=self.num_heads_kv,
            layer_id=0,
        )

    def _create_qkv_tensors(self, tokens_len):
        """Create q, k, v tensors for testing."""
        shape = (tokens_len, self.num_heads, self.head_dim)
        shape_kv = (tokens_len, self.num_heads_kv, self.head_dim)
        return (
            torch.randn(shape, dtype=self.dtype, device=self.device),
            torch.randn(shape_kv, dtype=self.dtype, device=self.device),
            torch.randn(shape_kv, dtype=self.dtype, device=self.device),
        )

    def _run_reference_forward(
        self, mode, q, k, v, layer, forward_batch, expected_shape
    ):
        """Run reference forward pass using native backend."""
        if mode == ForwardMode.EXTEND:
            output = self.ref_backend.forward_extend(q, k, v, layer, forward_batch)
        else:  # ForwardMode.DECODE
            output = self.ref_backend.forward_decode(q, k, v, layer, forward_batch)
        return output.view(expected_shape)

    def _verify_output(self, output, expected_shape, output_ref=None):
        """Verify output tensor shape, dtype, and values."""
        #assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        assert output.dtype == self.dtype
        assert output.device.type == "cuda"
        assert torch.isnan(output).sum().item() == 0, "Output contains NaN values"

        if output_ref is not None:
            if not torch.allclose(output, output_ref[:output.shape[0]], atol=1e-1, rtol=0.0):
                # Check where the values differ beyond the given tolerances
                diff_mask = ~torch.isclose(output, output_ref[:output.shape[0]], atol=1e-1, rtol=0.0)

                # Find the first index where the difference occurs
                if diff_mask.any():
                    first_mismatch_idx = diff_mask.nonzero()[0]
                    print(
                        "First mismatch at index:", tuple(first_mismatch_idx.tolist()), flush=True
                    )
                    print("output:", output[tuple(first_mismatch_idx.tolist())], flush=True)
                    print("output_ref:", output_ref[tuple(first_mismatch_idx.tolist())], flush=True)
                    print(output, output_ref, flush=True)
                raise AssertionError(
                    "Attention output is not close to the torch native backend output"
                )

    def _create_forward_batch(self, mode, q_len, seq_len, batch_size, prefix_len=0, page_size=1):
        """Create a forward batch for testing based on mode and lengths."""
        self._init_model_runner(page_size=page_size)

        # Default to self.seq_len if not specified
        q_len = q_len or seq_len

        if mode == ForwardMode.EXTEND:
            total_len = prefix_len + q_len
            out_cache_start = prefix_len * batch_size
            out_cache_end = total_len * batch_size

            forward_batch = ForwardBatch(
                batch_size=batch_size,
                input_ids=torch.randint(
                    0, 100, (batch_size, q_len), device=self.device
                ),
                out_cache_loc=torch.arange(
                    out_cache_start, out_cache_end, device=self.device
                ),
                seq_lens_sum=batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(batch_size, device=self.device),
                seq_lens=torch.tensor(
                    [total_len] * batch_size, device=self.device
                ),
                seq_lens_cpu=torch.tensor([total_len] * batch_size, device="cpu"),
                extend_prefix_lens=torch.tensor(
                    [prefix_len] * batch_size, device=self.device
                ),
                extend_prefix_lens_cpu=torch.tensor(
                    [prefix_len] * batch_size, device="cpu"
                ),
                extend_seq_lens=torch.tensor(
                    [q_len] * batch_size, device=self.device
                ),
                extend_seq_lens_cpu=torch.tensor(
                    [q_len] * batch_size, device="cpu"
                ),
                attn_backend=self.backend,
            )
        else:  # ForwardMode.DECODE
            decode_len = q_len  # Assuming 1 for decode testing
            total_len = seq_len + decode_len
            if mode == ForwardMode.DECODE and page_size > 1:
                # Get next page_size multiple of self.seq_len
                out_cache_start = (
                    batch_size * seq_len // page_size + 1
                ) * page_size
                # out_cache_end is the start of the next block
                out_cache_end = out_cache_start + decode_len * page_size
            else:
                out_cache_start = batch_size * seq_len
                out_cache_end = batch_size * total_len

            forward_batch = ForwardBatch(
                batch_size=batch_size,
                input_ids=torch.randint(
                    0, 100, (batch_size, decode_len), device=self.device
                ),
                out_cache_loc=torch.arange(
                    out_cache_start, out_cache_end, device=self.device
                ),
                seq_lens_sum=batch_size * total_len,
                forward_mode=mode,
                req_pool_indices=torch.arange(batch_size, device=self.device),
                seq_lens=torch.tensor(
                    [total_len] * batch_size, device=self.device
                ),
                seq_lens_cpu=torch.tensor([total_len] * batch_size, device="cpu"),
                attn_backend=self.backend,
            )

        # Add token pool
        forward_batch.req_to_token_pool = self.model_runner.req_to_token_pool

        # Write current batch's req_to_token to req_to_token_pool
        self._mock_write_to_req_to_token_pool(batch_size, total_len, page_size)
        # Add kv pool for this forward batch
        forward_batch.token_to_kv_pool = self.model_runner.token_to_kv_pool

        return forward_batch

    def _setup_kv_cache(self, batch_size, forward_batch, layer, cache_len):
        # Create constant values for the prefix cache for easy debugging
        cache_k = torch.ones(
            batch_size * cache_len,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        cache_v = (
            torch.ones(
                batch_size * cache_len,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            * 2
        )

        # Set the prefix KV cache
        forward_batch.token_to_kv_pool.set_kv_buffer(
            layer,
            torch.arange(batch_size * cache_len, device=self.device),
            cache_k,
            cache_v,
            layer.k_scale,
            layer.v_scale,
        )

    def _run_attention_test(self, mode, q_len, seq_len, batch_size, prefix_len=0, page_size=1, warmup=20, iters=100):
        """
            Run an attention test with the specified parameters.
        Args:
            mode: ForwardMode.EXTEND or ForwardMode.DECODE
            q_len: Length of the query sequence. For decode mode, q_len is 1.
            prefix_len: Length of the prefix sequence for extend mode
            page_size: Page size for the KV cache
        """
        layer = self._create_attention_layer()

        # Create forward batch and set up
        forward_batch = self._create_forward_batch(mode, q_len, seq_len, batch_size, prefix_len, page_size)

        # Create QKV tensors for the input
        q, k, v = self._create_qkv_tensors(batch_size * q_len)

        # KV cache for prefixed extend is prefix_len
        # KV cache for decode is same as seq_len
        # No KV cache for extend without prefix
        if mode == ForwardMode.EXTEND:
            if prefix_len > 0:
                self._setup_kv_cache(batch_size, forward_batch, layer, prefix_len)
        else:
            self._setup_kv_cache(batch_size, forward_batch, layer, seq_len)

        self.backend.init_forward_metadata(forward_batch)

        start = end = 0

        if mode == ForwardMode.EXTEND:
            expected_shape = (
                batch_size * q_len,
                self.num_heads * self.head_dim,
            )
            # Warmup iterations
            torch.cuda.synchronize()
            for _ in range(warmup):
                output = self.backend.forward_extend(q, k, v, layer, forward_batch)
            torch.cuda.synchronize()

            # Run iterations
            start = time.time()
            for _ in range(iters):
                output = self.backend.forward_extend(q, k, v, layer, forward_batch)
            torch.cuda.synchronize()
            end = time.time()
        else:
            expected_shape = (batch_size, self.num_heads * self.head_dim)
            torch.cuda.synchronize()
            start = time.time()
            output = self.backend.forward_decode(q, k, v, layer, forward_batch)
            torch.cuda.synchronize()
            end = time.time()

        output_ref = self._run_reference_forward(
            mode, q, k, v, layer, forward_batch, expected_shape
        )

        self._verify_output(output, expected_shape, output_ref)

        return output, end - start

def test_forward_extend(bench, seq_len=2048):
    """Test the standard extend operation."""
    output, time_taken = bench._run_attention_test(ForwardMode.EXTEND, seq_len, seq_len, batch_size=1)
    return time_taken

def test_forward_decode(bench, batch_size=128, seq_len=2048):
    """Test the decode operation with cached tokens."""
    output, time_taken = bench._run_attention_test(ForwardMode.DECODE, q_len=1, seq_len=seq_len, batch_size=batch_size)
    return time_taken

def test_forward_extend_with_page_size_greater_than_1(bench, seq_len=2048):
    """Test extending from cached prefix tokens with page size greater than 1."""
    batch_size = 1
    output, time_taken = bench._run_attention_test(ForwardMode.EXTEND, seq_len, seq_len, batch_size=batch_size, page_size=64)
    return time_taken

def test_forward_decode_with_page_size_greater_than_1(bench, batch_size=128, seq_len=2048):
    """Test decode operation with page size greater than 1."""
    output, time_taken = bench._run_attention_test(ForwardMode.DECODE, q_len=1, seq_len=seq_len, batch_size=batch_size, page_size=64)
    return time_taken


bench_nondet = BenchFlashAttentionBackend()
bench_det = BenchFlashAttentionBackend(deterministic=1)
bench_mixed = BenchFlashAttentionBackend(deterministic=8)
bench_nondet.setUp()
bench_det.setUp()
bench_mixed.setUp()

SEQ_LENS = [1024, 2048, 4096, 8192]
BATCH_SIZES = [1, 2, 4, 16, 64, 128, 256]
SPLIT_TILES = [1024, 2048, 4096, 8192]

pref_non_det = {}
pref_det = {}

dec_non_det = {}
dec_det = {}

for seq_len in SEQ_LENS:
    print(seq_len, flush=True)
    pref_non_det[seq_len] = test_forward_extend(bench_nondet, seq_len)
    pref_det[seq_len] = {}
    for split in SPLIT_TILES:
        os.environ["SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE"] = str(split)
        pref_det[seq_len][split] = test_forward_extend(bench_det, seq_len)


for seq_len in SEQ_LENS:
    print(f"Extend seq_len={seq_len}: Non-deterministic time = {pref_non_det[seq_len]:.4f}s, "
          f"{' '.join([f'Deterministic (split size {split}) time = {pref_det[seq_len][split]:.4f}s' for split in SPLIT_TILES])}"
          )

'''
for batch_size in BATCH_SIZES:
    dec_non_det[batch_size] = test_forward_decode(bench_nondet, batch_size=batch_size, seq_len=8192)
    dec_det[batch_size] = test_forward_decode(bench_det, batch_size=batch_size, seq_len=8192)

for batch_size in BATCH_SIZES:
    print(f"Decode batch_size={batch_size}: Non-deterministic time = {dec_non_det[batch_size]:.4f}s, Deterministic time = {dec_det[batch_size]:.4f}s")
'''