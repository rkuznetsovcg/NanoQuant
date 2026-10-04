<h1>NanoQuant: Efficient Sub-1-Bit Quantization<br>of Large Language Models</h1>
<h4>Authors: <a href="mailto:c42.hyochan@samsung.com">Hyochan Chong</a><sup>*</sup>, <a href="mailto:dongkyu.k@samsung.com">Dongkyu Kim</a><sup>*,&dagger;</sup>, <a href="mailto:c046385.kim@samsung.com">Changdong Kim</a>, <a href="mailto:manner.choi@samsung.com">Minseop Choi</a></h4>
<p><sup>*</sup>Equal Contribution, <sup>&dagger;</sup>Corresponding Author</p>

---

[![arXiv](https://img.shields.io/badge/arXiv-2602.06694-b31b1b.svg)](https://arxiv.org/abs/2602.06694)
[![Hugging Face – paper](https://img.shields.io/badge/Hugging%20Face-Paper-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/papers/2602.06694)
[![ICML](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/virtual/2026/poster/61392)


**NanoQuant** is a **post-training quantization** algorithm that enables **sub-1-bit** LLM weight quantization.

<div align="center">
<img src="media/ppl_bpw.png" width="600">
</div>

---

## Abstract

> **NanoQuant** is a state-of-the-art post-training quantization method that can compress LLMs to **sub-1-bit levels**. It introduces a novel non-factored/factored decomposition of weight matrices which, combined with learnable mixture-of-rank binary bases, achieves extreme compression while preserving model accuracy. By iteratively optimizing bases via ADMM and compensating per-layer quantization error, NanoQuant enables up to **over 3x faster decoding** and **10x smaller memory footprints**, pushing the frontier of practical ultra-low-bit LLM deployment.

---

## Key Features

### Method & Efficiency
* **Sub-1-bit Compression:** Pushes quantization below 1 bit per weight via mixture-of-rank binary bases.
* **Post-Training Quantization (PTQ):** No retraining required; calibration-only compression.
* **State-of-the-art Decoding Speed:** Up to **3x faster** inference via optimized binary GEMV/GEMM kernels.
* **Extreme Memory Reduction:** Up to **10x smaller** model footprints compared to FP16.

### Supported Models
The codebase currently supports the following architectures:
* **OPT**
* **Llama** (Llama-1, Llama-2, Llama-3)
* **Qwen** (Qwen-2.5, Qwen-3, Qwen-3.5, Qwen-3.8)
* **Gemma** (Gemma-2, Gemma-3)
* **Rnj-1**

Qwen-3.5 and Qwen-3.8 use a hybrid text decoder with both full-attention and
Gated DeltaNet (linear-attention) blocks. NanoQuant quantizes the text backbone
and saves it as a causal language model; the multimodal vision tower and
image/video inputs are not included in the quantized checkpoint.

Qwen-3.5/Qwen-3.8 support requires Python 3.10 or newer and Transformers 5.13.0
or newer (and before 6.0). The install command below resolves a compatible version.

### Custom GPU Kernels
* **GEMM (prefill):** CUDA
* **GEMV (decode):** CUDA

---

## 📦 Installation

```bash
# create conda environment
conda create -n nanoquant python=3.12 -y
conda activate nanoquant

# Install dependencies
pip install .

# Compile CUDA kernels
cd src/nanoquant/kernel
bash compile_kernel.sh
```

---

## 🎯 Usage

### Basic Quantization

Compress a model using the main NanoQuant script. Below is an example for Llama-2-7b:

```bash
python -m nanoquant.main \
    --model_id meta-llama/Llama-2-7b-hf/ \
    --qmodel_path "Llama-2-7b-hf-1bit.pt" \
    --num_calib_samples 128 \
    --nonfact_epochs 8 \
    --fact_epochs 8 \
    --admm_outer_iters 400 \
    --ppl_task "wikitext2"
```

### Very Large Models (>70B)

For models that may not fit in CPU memory, use `--device_map auto` to enable GPU+CPU offloading:

```bash
python -m nanoquant.main \
    --model_id meta-llama/Llama-3-70B-Instruct \
    --device_map auto \
    --num_calib_samples 128 \
    --nonfact_epochs 8 \
    --fact_epochs 8 \
    --admm_outer_iters 400 \
    --ppl_task "wikitext2"
```

### Qwen3.8-27B

Qwen3.8 uses the Qwen3.5 text architecture. The following command quantizes its
text backbone:

```bash
python -m nanoquant.main \
    --model_id Qwen/Qwen3.8-27B \
    --qmodel_path "Qwen3.8-27B-NQ-1bit.pt" \
    --num_calib_samples 128 \
    --nonfact_epochs 8 \
    --fact_epochs 8 \
    --admm_outer_iters 400 \
    --ppl_task "wikitext2"
```

### Multilingual and agentic calibration data

To calibrate Qwen3.8 with a small mixture closer to coding and agent use, prepare
128 windows of 2,048 tokens (the default NanoQuant calibration budget):

```bash
python scripts/prepare_calibration_dataset.py --overwrite
```

The prepared Hugging Face dataset is saved at
`data/calibration/generated/qwen3.8-27b-multidomain-128x2048/dataset` and can be
selected with `--calib_dataset`:

```bash
python -m nanoquant.main \
    --model_id Qwen/Qwen3.8-27B \
    --calib_dataset data/calibration/generated/qwen3.8-27b-multidomain-128x2048/dataset \
    --num_calib_samples 128
```

The mix contains English, Russian, Chinese, Spanish, German, French, and Korean
instruction data; Korean multi-turn tool-agent trajectories; code-review prompts
in Korean and English over permissively licensed Python, JavaScript, TypeScript,
Java, C++, Go, Rust, and Shell source; additional agentic coding tasks; and
Hermes function calls/structured JSON. Sources are streamed and only 128 token
windows are retained; Stack v3 scans repositories until each language quota is
filled. Terminal-Bench, DeepSWE, and Toolathlon evaluation examples are
excluded. The manifest records source, content-language, and prompt-language
counts, plus CC BY 4.0 attribution for the small Korean-agent slice. This
calibrates the supported text backbone; it does not add image or video
calibration to the vision tower. `--overwrite` preserves the previous dataset
in a sibling `.previous` folder when rebuilding.

To use the experimental KronQ-inspired trace allocator at the same 0.55 bpw,
run `python -m nanoquant.main configs/qwen38-0.55-kronq-trace.json`. The default
`sensitivity` allocator remains available as the control profile
`configs/qwen38-0.55-balanced.json`.

### Qwen calibration runtime

The loader defaults to `--attn_implementation auto`: on Ampere/Ada it tries
FlashAttention-2; on Hopper it tries FlashAttention-3 then FlashAttention-2.
Import/build failures are reported and auto falls back to SDPA, which can already
select PyTorch's Flash kernel. Explicit backend requests fail before weights load
if the kernel cannot be imported. CPU checkpoint inference keeps SDPA.

Install `pip install -e '.[kernels]'` to let Transformers fetch compatible compiled
FlashAttention builds from the Hugging Face Kernel Hub. The native package is used
when available. The FA3 Hub path explicitly selects `kernels-community/flash-attn3`
with backward, rather than relying on a version-dependent vLLM fallback. Available
choices: `auto`, `sdpa`, `flash_attention_2`, `flash_attention_3`, `flash_attention_4`.
FA4 is an explicit beta option for Hopper / datacenter Blackwell (SM90/100/110);
auto retains SDPA on other GPU families. Selection imports code but does not run
a GPU kernel. All paths retain BF16 and Transformers' causal/padding mask handling.
See [HF kernel loading](https://huggingface.co/docs/transformers/main/en/kernel_doc/loading_kernels).

Qwen3.5/3.8 Gated DeltaNet uses separate
optional FLA and causal-conv1d kernels. Install them on the CUDA host after CUDA
PyTorch and the build tools:

```bash
pip install -e '.[qwen-fast]' --no-build-isolation
```

`--require_fast_linear_attention true` checks CUDA availability and imports of
the required package functions before downloading model weights. Missing or
broken imports stop the run. The check does not execute a GPU kernel. The default
warns and permits the PyTorch fallback; Transformers selects the installed package
implementations itself. See the [Qwen3.5 usage notes](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5).

For Hopper / compatible Blackwell, an optional installation profile enables
[Qwen FlashQLA](https://github.com/QwenLM/FlashQLA) through FLA's own dispatcher:

```bash
pip install -e '.[qwen-flashqla]' --no-build-isolation
```

This includes FLA >=0.5.2, FlashQLA >=0.1.3, causal-conv1d and HF kernels. FlashQLA
requires CUDA >=12.8 and PyTorch >=2.8. FLA checks dtype, head dimensions and gradient
requirements per call, and falls back to Triton for unsupported calls. Some FLA
versions still use Triton for SM120 backward despite FlashQLA 0.1.3 adding that
implementation. `FLA_FLASH_QLA=0` selects the FLA Triton path. The loader checks
the optional FlashQLA APIs/dispatcher before weights load on eligible devices.
FLA and FlashQLA wheels contain Python kernel definitions: first use can still JIT
compile Triton/TileLang code. HF FlashAttention-2/3 builds are compiled binaries;
FA4 uses CuTe DSL.

```bash
python -m nanoquant.main \
    --model_id Qwen/Qwen3.8-27B \
    --bits 0.55 \
    --calib_dataset data/calibration/generated/qwen3.8-27b-multidomain-128x2048/dataset \
    --num_calib_samples 128 \
    --require_fast_linear_attention true \
    --attn_implementation auto \
    --qmodel_path Qwen3.8-27B-NQ-0.55.pt
```

Rank planning prints selected-linear bpw, whole-model bpw, packed weight size,
and the size with BF16 factors. `--bits` budgets selected linear weights and
scales. Embeddings, `lm_head`, and other unchanged weights retain their precision.
Qwen3.8-27B's two untied vocabulary matrices alone use about 4.74 GiB in BF16.
The regular checkpoint loader unpacks binary factors into BF16. Compact GPU
inference requires preparing NanoQuant kernels through `NanoQuantLinear._prepare_kernel`.
Estimates exclude activations, gradients, optimizer state, metadata and kernel padding.

The balanced profile and experimental KronQ-inspired rank allocator are available
as separate recipes. They keep the same target budget; the second changes only
how ranks are assigned:

```bash
python -m nanoquant.main configs/qwen38-0.55-balanced.json
python -m nanoquant.main configs/qwen38-0.55-kronq-trace.json
```

### Kernel Benchmarking

Run custom CUDA kernels for GEMV and GEMM decode-stage benchmarking. Save a 1-bit model (e.g., `Llama-3.2-1B-NQ-1bit.pt`) in the repo root, then:

```bash
# Run benchmark
cd src/nanoquant/kernel
bash bench_decode.sh
```

---

## Results

### Pareto Accuracy

We compare results across pretrained models in the Qwen3 family (0.6B, 1.7B, 4B, 8B, 14B):

<div align="center">
<img src="media/qwen3_pareto.png" width="600">
</div>

## GPU Kernel Performance

NanoQuant implements matmul-free GEMV kernels that do not require NVIDIA Tensor Cores. All tests are conducted with 128 input tokens.

### NanoQuant vs <a href="https://github.com/dropbox/gemlite">GemLite</a>

NanoQuant GEMV kernels outperform state-of-the-art binary Triton kernels from GemLite.

<div align="center">
<img src="media/h100_qwen3_gemlite_gemv.png" width="600">
</div>

### NanoQuant vs. Vector Quantization

NanoQuant GEMV kernels also outperform vector quantization kernels in both speed and memory efficiency.

<div align="center">
<img src="media/h100_gemv_nq_vs_vq.png" width="600">
</div>

---

## Citation

If you find NanoQuant useful or relevant to your research, please kindly cite our paper:

```bibtex
@article{chong2026nanoquant,
  title={NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models},
  author={Chong, Hyochan and Kim, Dongkyu and Kim, Changdong and Choi, Minseop},
  journal={arXiv preprint arXiv:2602.06694},
  year={2026}
}
```

## License

This project is licensed under the [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.

## Заметка на потом: попробовать батчинг сбора статистик

На H100 проверить, ускорит ли обработка нескольких calibration-окон за один проход сбор статистик. Начать с батча 2 и сравнить с текущей обработкой по одному окну. При реализации отдельно считать отсечение выбросов и вклад каждого окна, сохранить нынешний масштаб градиентов и только затем складывать статистики. Не менять основной режим, пока не сравним скорость и качество.
