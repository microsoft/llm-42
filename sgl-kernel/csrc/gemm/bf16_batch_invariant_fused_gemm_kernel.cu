#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cudaTypedefs.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/arch/mma.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/thread/mma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

#include "math.hpp"
#include "utils.h"

using namespace cute;

#define SWITCH_TYPE(type, name, ...) \
  {                                      \
    if (type == torch::kHalf) {  \
      using name = cutlass::half_t;      \
      { __VA_ARGS__ }                    \
    } else if (type == torch::kBFloat16) { \
      using name = cutlass::bfloat16_t;   \
      { __VA_ARGS__ }                    \
    } else {                             \
      TORCH_CHECK(false, "Unsupported type"); \
    }                                    \
  }

#define SWITCH_MAJOR(row_condition, name, ...) \
  if (row_condition) {                         \
    using name = cutlass::layout::RowMajor;    \
    { __VA_ARGS__ }                            \
  } else {                                     \
    using name = cutlass::layout::ColumnMajor; \
    { __VA_ARGS__ }                            \
  }

#define SWITCH_OUT_TYPE(type, name, ...) \
  {                                      \
    if (type == torch::kHalf) {  \
      using name = cutlass::half_t;      \
      { __VA_ARGS__ }                    \
    } else if (type == torch::kBFloat16) { \
      using name = cutlass::bfloat16_t;   \
      { __VA_ARGS__ }                    \
    } else {                             \
      TORCH_CHECK(false, "Unsupported out type"); \
    }                                    \
  }

#if defined CUDA_VERSION && CUDA_VERSION >= 12000
template <
    typename ElementType,
    typename OutElementType,
    typename AccumElementType,
    typename LayoutA,
    typename LayoutB,
    typename CTAShape,
    typename ClusterShape,
    typename MainloopScheduleType,
    typename EpilogueScheduleType,
    typename TileSchedulerType = void,
    bool WithBias = false>
struct DeviceGemmbf16RowwiseSm90 {
  //static_assert(std::is_same_v<ElementType, cutlass::bfloat16_t>, "ElementType must be BF16");

  // A matrix configuration
  using ElementA = ElementType;               // Element type for A matrix operand
  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<ElementA>::value;  // Memory access granularity/alignment of A
                                                    // matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = ElementType;                  // Element type for B matrix operand
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<ElementB>::value;  // Memory access granularity/alignment of B
                                                    // matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using ElementC = OutElementType;                      // Element type for C matrix operands
  using LayoutC = cutlass::layout::RowMajor;  // Layout type for C matrix operands
  static constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<OutElementType>::value;  // Memory access granularity/alignment of C matrices in
                                                          // units of elements (up to 16 bytes)

  // Output matrix configuration
  using ElementOutput = OutElementType;            // Element type for output matrix operands
  using LayoutOutput = cutlass::layout::RowMajor;  // Layout type for output matrix operands
  static constexpr int AlignmentOutput = 128 / cutlass::sizeof_bits<ElementOutput>::value;

  // Multiply-accumulate blocking/pipelining details
  using ElementAccumulator = AccumElementType;  // Element type for internal accumulation
  using ElementCompute = float;                 // Element type for compute
  using ElementComputeEpilogue = float;
  using ArchTag = cutlass::arch::Sm90;  // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
  using TileShape = CTAShape;                            // Threadblock-level tile size

  using StageCountType = cutlass::gemm::collective::StageCountAuto;      // Stage count maximized
                                                                         // based on the tile size
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;  // Kernel to launch based on the default
                                                                         // setting in the Collective Builder

  using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0,
      TileShape,
      ElementOutput,
      ElementOutput,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

  // With bias
  using AddBias = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::plus,
      ElementOutput,
      ElementComputeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest>;

  using EVTComputeWithBias = cutlass::epilogue::fusion::Sm90EVT<AddBias, Accum, Bias>;

  using EpilogueEVT = typename cutlass::platform::conditional<WithBias, EVTComputeWithBias, Accum>::type;

  using CollectiveEpilogueBias = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementComputeEpilogue,
      void,
      LayoutC,
      AlignmentC,
      ElementOutput,
      LayoutOutput,
      AlignmentOutput,
      cutlass::epilogue::TmaWarpSpecialized,
      EpilogueEVT>::CollectiveOp;

  using CollectiveEpilogueDefault = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    EpilogueScheduleType>::CollectiveOp;

  using CollectiveEpilogue = typename cutlass::platform::conditional<WithBias, CollectiveEpilogueBias, CollectiveEpilogueDefault>::type;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      ElementA,
      LayoutA,
      AlignmentA,
      ElementB,
      LayoutB,
      AlignmentB,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopScheduleType>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,  // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue,
      TileSchedulerType>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};


template <typename Gemm, bool WithBias>
typename Gemm::Arguments prepare_sm90_bf16_batch_invariant_fused_args(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const int32_t m1,
    const c10::optional<torch::Tensor>& bias) {
  using ElementT = typename Gemm::ElementA;
  using ElementOutput = typename Gemm::ElementD;
  using ElementComputeEpilogue = float;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  int32_t n = b.size(1);
  int32_t k = a.size(1);

  //int32_t m1 = static_cast<int32_t>(m * split_frac);
  //int32_t m2 = m - m1;

  ElementT const* ptr_a = reinterpret_cast<ElementT const*>(a.data_ptr());
  ElementT const* ptr_a1, *ptr_a2;
  if (a.stride(0) == 1) {
    // column major
    //TORCH_CHECK(a.stride(1) == m, "mat_a is not a column major or row major tensor");
    ptr_a1 = ptr_a;
    //ptr_a2 = ptr_a + m1;
  } else {
    // row major
    TORCH_CHECK(a.stride(1) == 1, "mat_a is not a column major or row major tensor");
    ptr_a1 = ptr_a;
    //ptr_a2 = ptr_a + m1 * k;
  }

  ElementT const* ptr_b = reinterpret_cast<ElementT const*>(b.data_ptr());

  ElementOutput const* ptr_bias1 = nullptr, *ptr_bias2 = nullptr;
  if constexpr (WithBias) {
    TORCH_CHECK(bias.has_value())
    ElementOutput const* ptr_bias = reinterpret_cast<ElementOutput const*>(bias.value().data_ptr());
    if (bias->stride(0) == 1) {
      // column major
      TORCH_CHECK(false, "column-major bias is not supported");
    } else {
      // row major
      TORCH_CHECK(bias->stride(1) == 1, "bias is not a row major tensor");
      ptr_bias1 = ptr_bias;
      //ptr_bias2 = ptr_bias + m1 * n;
    }
  }
  ElementOutput* ptr_d = reinterpret_cast<ElementOutput*>(out.data_ptr());
  ElementOutput* ptr_d1 = ptr_d;
  //ElementOutput* ptr_d2 = ptr_d + m1 * n;

  StrideA stride_a1 = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m1, k, 1));
  //StrideA stride_a2 = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m2, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
  StrideC stride_c;
  StrideD stride_d1 = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m1, n, 1));
  //StrideD stride_d2 = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m2, n, 1));
  typename Gemm::Arguments args1 = {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m1, n, k, 1},
      {ptr_a1, stride_a1, ptr_b, stride_b},
      {{},  // epilogue.thread
       nullptr,
       stride_c,
       ptr_d1,
       stride_d1}};
  /*typename Gemm::Arguments args2 = {
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m2, n, k, 1},
      {ptr_a2, stride_a2, ptr_b, stride_b},
      {{},  // epilogue.thread
       nullptr,
       stride_c,
       ptr_d2,
       stride_d2}};*/
  if constexpr (WithBias) {
    args1.epilogue.thread = {
        {}, // Accum
        {ptr_bias1},
        {}, // Add
    };
    /*args2.epilogue.thread = {
        {}, // Accum
        {ptr_bias2},
        {}, // Add
    };*/
  }

  return args1;
}

template <typename Gemm, bool WithBias>
void launch_sm90_bf16_batch_invariant_fused_mm(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const double split_frac,
    const c10::optional<torch::Tensor>& bias) {

  int32_t m = a.size(0);
  int32_t m1 = static_cast<int32_t>(m * split_frac);
  int32_t m2 = m - m1;

  auto args1 = prepare_sm90_bf16_batch_invariant_fused_args<Gemm, WithBias>(out, a, b, m1, bias);

  Gemm gemm_op;
  auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt16).device(a.device());

  torch::Tensor workspace1, workspace2;

  if (m1 > 0) {
    //size_t workspace_size1 = gemm_op.get_workspace_size(args1);
    //workspace1 = torch::empty(workspace_size1, workspace_options);
    //fprintf(stderr, "workspace_size1: %zu\n", workspace_size1);
  }
  /*if (m2 > 0) {
    size_t workspace_size2 = gemm_op.get_workspace_size(args2);
    workspace2 = torch::empty(workspace_size2, workspace_options);
  }*/

  auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

  static cudaStream_t stream1 = nullptr;
  // stream2 = nullptr;
  static at::cuda::CUDAStream stream2 = at::cuda::getStreamFromPool();
  static cudaEvent_t event1 = nullptr, event2 = nullptr;
  if (stream1 == nullptr) {
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&event1, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&event2, cudaEventDisableTiming);
  }

  if (m1 > 0) {
    gemm_op.initialize(args1, nullptr/*workspace1.data_ptr()*/, stream1);
  }

  // Wait for previous kernels to finish
  cudaEventRecord(event1, stream);
  cudaStreamWaitEvent(stream1, event1, 0);
  cudaStreamWaitEvent(stream2.stream(), event1, 0);

  //auto can_implement = gemm_op.can_implement(args1);
  //TORCH_CHECK(can_implement == cutlass::Status::kSuccess)

  if (m2 > 0) {
    {
      c10::cuda::CUDAStreamGuard guard(stream2);
      torch::Tensor a2 = torch::from_blob(
          static_cast<void*>(
              reinterpret_cast<typename Gemm::ElementA*>(a.data_ptr()) + m1 * a.size(1)),
          {m2, a.size(1)},
          a.strides(),
          a.options());
      torch::Tensor out2 = torch::from_blob(
          static_cast<void*>(
              reinterpret_cast<typename Gemm::ElementD*>(out.data_ptr()) + m1 * out.size(1)),
          {m2, out.size(1)},
          out.strides(),
          out.options());
      torch::mm_out(
          out2,
          a2,
          b);
      //auto status = gemm_op.run(args2, workspace2.data_ptr(), stream2);
      //TORCH_CHECK(status == cutlass::Status::kSuccess)
    }
  }

  if (m1 > 0) {
    auto status = gemm_op.run(stream1);
    TORCH_CHECK(status == cutlass::Status::kSuccess)
  }

  cudaEventRecord(event1, stream1);
  cudaEventRecord(event2, stream2.stream());
  cudaStreamWaitEvent(stream, event1, 0);
  cudaStreamWaitEvent(stream, event2, 0);

}

template <
    typename OutType,
    typename LayoutA,
    typename LayoutB,
    typename CTAShape,
    typename ClusterShape,
    typename MainloopScheduleType,
    typename EpilogueScheduleType,
    typename TileSchedulerType,
    bool BIAS>
void sm90_bf16_batch_invariant_fused_dispatch_bias(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    const double split_frac,
    const c10::optional<torch::Tensor>& bias,
    bool fast_accum = true,
    bool use_persistent = false) {
  using ElementOutput = OutType;
  using AccumElementType = float;

  SWITCH_TYPE(a.dtype(), ElementInput, {
    using Gemm = typename DeviceGemmbf16RowwiseSm90<
        ElementInput,
        ElementOutput,
        AccumElementType,
        LayoutA,
        LayoutB,
        CTAShape,
        ClusterShape,
        MainloopScheduleType,
        EpilogueScheduleType,
        TileSchedulerType,
        BIAS>::Gemm;
    return launch_sm90_bf16_batch_invariant_fused_mm<Gemm, BIAS>(out, a, b, split_frac, bias);
  });
}

template <typename OutType, typename LayoutA, typename LayoutB>
void sm90_bf16_batch_invariant_fused_dispatch_shape(
    torch::Tensor& out,
    const torch::Tensor& a,
    const torch::Tensor& b,
    double split_frac,
    const c10::optional<torch::Tensor>& bias) {
  // Disable splitting to avoid zero-padding tile for det kernel.
  if (a.size(0) <= 64 && split_frac != 0.0) {
    split_frac = 1.0;
  }
  uint32_t const m = static_cast<int32_t>(a.size(0) * split_frac);
  using PingpongScheduler = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using BasicScheduler = cutlass::gemm::KernelTmaWarpSpecialized;
  using CooperativeScheduler = cutlass::gemm::KernelTmaWarpSpecializedCooperative;

  using PersistentTileScheduler = cutlass::gemm::PersistentScheduler;
  using BasicTileScheduler = void;
  using StreamScheduler = cutlass::gemm::StreamKScheduler;

  using ChosenSmallScheduler = PingpongScheduler;
  using ChosenBigScheduler = PingpongScheduler;
  using ChosenTileScheduler = PersistentTileScheduler;

  using ChosenSmallEpilogueScheduler = cutlass::epilogue::TmaWarpSpecialized;
  using ChosenEpilogueScheduler = cutlass::epilogue::TmaWarpSpecializedCooperative;
  if (bias) {
    if (m <= 128) {
      // m in [1, 64]
      return sm90_bf16_batch_invariant_fused_dispatch_bias<
          OutType,
          LayoutA,
          LayoutB,
          Shape<_64, _64, _128>,
          Shape<_1, _1, _1>,
          ChosenSmallScheduler,
          ChosenSmallEpilogueScheduler,
          BasicTileScheduler,
          true>(out, a, b, split_frac, bias);
    } else if (m <= 512) {
      // m in (64, 256]
      return sm90_bf16_batch_invariant_fused_dispatch_bias<
          OutType,
          LayoutA,
          LayoutB,
          Shape<_128, _64, _64>,
          Shape<_2, _1, _1>,
          ChosenSmallScheduler,
          ChosenSmallEpilogueScheduler,
          ChosenTileScheduler,
          true>(out, a, b, split_frac, bias);
    } else {
      // m in (512, inf)
      return sm90_bf16_batch_invariant_fused_dispatch_bias<
          OutType,
          LayoutA,
          LayoutB,
          Shape<_128, _128, _64>,
          Shape<_2, _1, _1>,
          ChosenSmallScheduler,
          ChosenSmallEpilogueScheduler,
          ChosenTileScheduler,
          true>(out, a, b, split_frac, bias);
    }
  } else {
    if (m <= 128) {
      // m in [1, 128]
      return sm90_bf16_batch_invariant_fused_dispatch_bias<
          OutType,
          LayoutA,
          LayoutB,
          Shape<_64, _64, _128>,
          Shape<_1, _1, _1>,
          ChosenSmallScheduler,
          ChosenSmallEpilogueScheduler,
          BasicTileScheduler,
          false>(out, a, b, split_frac, bias);
    } else if (m < 256) {
      // m in (128, 256)
      return sm90_bf16_batch_invariant_fused_dispatch_bias<
          OutType,
          LayoutA,
          LayoutB,
          Shape<_256, _128, _64>,
          Shape<_2, _1, _1>,
          ChosenBigScheduler,
          ChosenEpilogueScheduler,
          ChosenTileScheduler,
          false>(out, a, b, split_frac, bias);
    } else if (m < 512) {
      // m in [256, 512)
      return sm90_bf16_batch_invariant_fused_dispatch_bias<
          OutType,
          LayoutA,
          LayoutB,
          Shape<_128, _64, _64>,
          Shape<_2, _1, _1>,
          ChosenSmallScheduler,
          ChosenSmallEpilogueScheduler,
          BasicTileScheduler,
          false>(out, a, b, split_frac, bias);
    } else if (m < 1024) {
      // m in [512, 1024)
      return sm90_bf16_batch_invariant_fused_dispatch_bias<
          OutType,
          LayoutA,
          LayoutB,
          Shape<_128, _128, _64>,
          Shape<_2, _1, _1>,
          ChosenSmallScheduler,
          ChosenSmallEpilogueScheduler,
          BasicTileScheduler,
          false>(out, a, b, split_frac, bias);
    } else {
      // m in [1024, inf)
      return sm90_bf16_batch_invariant_fused_dispatch_bias<
          OutType,
          LayoutA,
          LayoutB,
          Shape<_256, _128, _64>,
          Shape<_2, _1, _1>,
          ChosenBigScheduler,
          ChosenEpilogueScheduler,
          ChosenTileScheduler,
          false>(out, a, b, split_frac, bias);
    }
  }
}
#endif

torch::Tensor bf16_batch_invariant_fused_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Dtype& out_dtype,
    const double split_frac,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& out_tensor) {
  TORCH_CHECK(mat_a.is_cuda(), "mat_a must be a CUDA tensor");
  TORCH_CHECK(mat_b.is_cuda(), "mat_b must be a CUDA tensor");
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "mat_a and mat_b shapes cannot be multiplied");
  TORCH_CHECK(split_frac <= 1.0 && split_frac >= 0.0, "split_frac must be in [0.0, 1.0]");

  TORCH_CHECK(
      (mat_a.size(1) * mat_a.element_size()) % 16 == 0, "mat_a must be multiple of 16 bytes for memory alignment");
  TORCH_CHECK(
      (mat_b.size(0) * mat_b.element_size()) % 16 == 0, "mat_b must be multiple of 16 bytes for memory alignment");
  TORCH_CHECK(out_dtype == torch::kHalf || out_dtype == torch::kBFloat16, "out_dtype must be Half or BFloat16");

  if (bias) {
    TORCH_CHECK(bias->numel() == mat_b.size(1), "size of bias is not matched");
    TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias->dtype() == out_dtype, "bias dtype must match output dtype");
  }

  torch::Tensor out;

  if (out_tensor.has_value()) {
    out = out_tensor.value();
    TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
    TORCH_CHECK(out.dim() == 2, "out must be a 2D tensor");
    TORCH_CHECK(out.size(0) == mat_a.size(0) && out.size(1) == mat_b.size(1), "out shape is not matched");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
    TORCH_CHECK(out.dtype() == out_dtype, "out dtype must match out_dtype");
  } else {
    out = torch::empty({mat_a.size(0), mat_b.size(1)}, mat_a.options().dtype(out_dtype));
  }

  TORCH_CHECK((out.size(1) * out.element_size()) % 16 == 0, "out must be multiple of 16 bytes for memory alignment");

  auto sm_version = getSMVersion();
  // TODO: add support for sm_100
/*
#if defined CUDA_VERSION && CUDA_VERSION >= 12080
  if (sm_version == 100
#if CUDA_VERSION >= 12090
      || sm_version == 103
#endif
  ) {
    if (out_dtype == torch::kBFloat16) {
      sm100_bf16_batch_invariant_fused_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, bias);
    } else {
      sm100_bf16_batch_invariant_fused_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, bias);
    }
    return out;
  }
#endif
*/

#if defined CUDA_VERSION && CUDA_VERSION >= 12000
  if (sm_version >= 90) {
    if (mat_a.stride(0) != 1 && mat_a.stride(1) != 1) {
      TORCH_CHECK(false, "mat_a must be a row major or column major tensor");
    }
    if (mat_b.stride(0) != 1 && mat_b.stride(1) != 1) {
      TORCH_CHECK(false, "mat_b must be a row major or column major tensor");
    }

    SWITCH_OUT_TYPE(out_dtype, outType, {
      SWITCH_MAJOR(mat_a.stride(1) == 1, LayoutA, {
        SWITCH_MAJOR(mat_b.stride(1) == 1, LayoutB, {
          sm90_bf16_batch_invariant_fused_dispatch_shape<outType, LayoutA, LayoutB>(
            out, mat_a, mat_b, split_frac, bias);
        });
      });
      return out;
    });
  }
#endif
  // TODO: add support for sm_100
/*
#if defined CUDA_VERSION && CUDA_VERSION >= 12040
  if (sm_version == 89) {
    if (out_dtype == torch::kBFloat16) {
      sm89_bf16_batch_invariant_fused_dispatch_shape<cutlass::bfloat16_t>(out, mat_a, mat_b, bias);
    } else {
      sm89_bf16_batch_invariant_fused_dispatch_shape<cutlass::half_t>(out, mat_a, mat_b, bias);
    }
    return out;
  }
#endif
*/
  TORCH_CHECK_NOT_IMPLEMENTED(false, "No implemented bf16_batch_invariant_fused_mm for current compute capability: ", sm_version);
}
