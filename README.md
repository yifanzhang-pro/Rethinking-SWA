# [Yifan's Blog] Rethinking SWA

### Why Short Sliding Window Attention Will Replace ShortConv in Modern Architectures 

**Author:** Yifan Zhang  
**Date:** December 16, 2025  

In the rapid evolution of efficient sequence modeling, whether modern RNNs, state space models (SSMs), linear attention, or even standard full attention, optimization efforts have largely focused on the global mixing mechanism (the global mixer). Yet across these architectures, there remains an inconspicuous legacy artifact: ShortConv (short convolution).

**Core claim:**  
In an era where hardware-efficient algorithms (for example, FlashAttention and SSD) effectively mandate chunk-based computation, continuing to use convolutions with kernel sizes of 3 or 4 is an architectural mismatch. Since our models already split sequences into chunks for efficiency (for example, 64 or 128 tokens), we should directly replace ShortConv with Short Sliding Window Attention (ShortSWA). This aligns the local mixing range with the hardware chunk size and can substantially improve local modeling capability without materially increasing cost.

**A useful precedent from vision:**  
This shift mirrors a transition we already saw in vision backbones. Small, fixed-kernel convolutions were once the default local mixer, but ViT-style and SwinT-style designs replaced that assumption with attention-based mixing, and in many scaling regimes they matched or surpassed strong convolutional baselines. Even as convnets were modernized into ConvNeXt (convolutions redesigned in the image of Transformers), the broader trend remained: once hardware and scale make it practical, dynamic, content-adaptive mixing becomes structurally advantaged.

ShortSWA is the direct analog on sequences. It upgrades the local mixer from fixed, tiny receptive fields (ShortConv) to data-dependent, chunk-aligned receptive fields (ShortSWA). Put simply, just as ViT or Swin displaced modern conv variants in vision, ShortSWA will displace ShortConv once practitioners benchmark it seriously in the right compute regime.

## 1. Current architecture: the parallel projection paradigm

Most modern efficient architectures (RetNet, Lightning Attention, Mamba-2, GLA, GDN, etc.) as well as Transformer blocks adopt a similar parallel structure to maximize GPU throughput.

A typical pipeline looks like this:

1. Parallel input projection: The input u passes through a large Linear layer that produces all required branches at once (Gate, Value, Key, etc.).
2. Local mixing layer (the bottleneck): Before entering the heavy global mixer, each branch typically goes through a Conv1d(k=2 or 4) to provide translation invariance or local smoothing.
3. Global mixing layer (chunked): The core mechanism (SSM, linear attention, or masked attention) processes data in blocks or chunks using Tensor Cores.

### The misalignment
The global mixer operates on chunks (typically Q=64 or 128). However, within these large chunks brought into memory, we restrict the local receptive field to a convolution kernel spanning only 4 tokens. This is a mismatch. We already paid the memory cost to load a 128-token chunk into SRAM, so limiting local mixing to 4 tokens wastes that opportunity.

## 2. Why ShortSWA is the inevitable upgrade

The case for replacing ShortConv with ShortSWA rests on two pillars: hardware efficiency and modeling capacity, plus a familiar empirical pattern from other domains.

### A. Hardware: aligning with chunks
Modern GPUs (H100 or A100) depend on Tensor Cores, which excel at block-structured computation. Algorithms like SSD (state-space duality) and FlashAttention effectively enforce chunk sizes.

- ShortConv (k=4) is memory-bound. It loads data but performs very little arithmetic, yielding low arithmetic intensity.
- ShortSWA (window size 128): if the window size matches the chunk size, we run dense local attention inside each chunk. Because the data has already been loaded into SRAM or registers for chunk operations, the incremental cost of intra-chunk attention is small relative to the expressive gains.

### B. Modeling: filling the mid-range gap
- ShortConv captures only very short local dependencies (tokens t-1 to t-3).
- Linear attention and SSMs excel at global recurrence, but may exhibit attenuation at medium distances.
- Full attention is excellent globally, but expensive at full scale.

ShortSWA fills this gap. With a 128-token window aligned to the chunk, the model gains a sentence-level or clause-level receptive field. It captures precise local context that Conv(4) misses, reducing the burden on the global mixer to handle mid-range dependencies.

### C. Empirical intuition: the ConvNeXt moment for sequence local mixing
Why expect ShortSWA to consistently beat ShortConv rather than merely match it?

- ShortConv is fixed and stencil-like. With k in {2,4}, the operation is constrained to a tiny, position-local pattern with shared weights. It is excellent at smoothing, but weak at selective routing, for example mixing with the relevant token within the last sentence.
- ShortSWA is content-adaptive by construction. Within each chunk or window, attention can express conditional local interactions such as copying, alignment, delimiter-aware routing, or syntax-triggered mixing that a fixed kernel cannot emulate without additional depth or width.
- Scaling favors dynamic mixers. Once the implementation is fused and chunk-aligned, the marginal cost of richer local mixing is low. Optimization pressure shifts from saving FLOPs at all costs to spending similar compute more expressively. This is the regime where attention-style mixers historically gained an edge over conv-style mixers.

The practical prediction is straightforward. If you hold compute and memory bandwidth roughly fixed because you already chunk, ShortSWA will dominate ShortConv on accuracy per token and robustness, in the same way windowed attention backbones outcompeted modernized conv baselines in vision when implementations matured.

## 3. Generality: not limited to SSMs

While the discussion often centers on RNNs and SSMs, the same logic applies to full-attention Transformers.

### 1) For linear attention and SSMs (Mamba, GLA, RWKV)
These models process effectively unbounded context with linear-time recurrence. Adding a ShortSWA layer creates a powerful local–global hybrid. SWA handles high-frequency, precise local interactions, while linear recurrence handles long-range retrieval. This matches the state-space duality principle: combining local attention with global state-space recurrence.

### 2) For full attention (Transformers)
Even in standard Transformers, replacing early positional or convolutional mixing (as seen in Conformer-style designs or early ViT practices) with ShortSWA can help.

- Query and key conditioning: before the expensive global QK^T multiplication, ShortSWA mixes information so tokens can form locally enhanced representations.
- Sliding window attention: many efficient Transformers already use sliding window attention. Explicitly decoupling it as a lightweight local mixing layer ensures that even if global attention is sparse, the local region still has dense attention coverage.

## 4. Implementation sketch

The replacement is straightforward: move from fixed-weight convolution to a dynamic window-attention mechanism matched to the chunk dimension.

Conceptual code (PyTorch-like):

```python
class ModernBlockWithSWA(nn.Module):
    def forward(self, u):
        # 1. Parallel projection (standard in Mamba-2, GLA, Transformers)
        # Generate all branches at once: Gates, Values, Keys, etc.
        z, x, B, C, ... = self.in_proj(u).split(...)

        # 2. Local mixing: upgraded
        # Old: x = conv1d(x, k=4)
        # New: chunk-aligned ShortSWA
        # Window size = 128 (matching the chunk length used by the global mixer)
        x = self.short_swa(x, window_size=128)

        # Note: ShortSWA can be implemented efficiently using FlashAttention's
        # sliding-window or block-masking capabilities.

        # 3. Global mixing (the core heavy lifting)
        # Could be SSM (SSD), linear attention, or full attention
        y = self.global_mixer(x, B, C, ...)

        # 4. Gating and output
        y = y * F.silu(z)
        return self.out_proj(y)
````

## 5. Conclusion

ShortConv was a necessary bridge when transitioning from standard Transformers to efficient RNN-style models. But in the era of chunk-based computation, it has become a bottleneck for expressivity.

Whether you are building the next-generation Mamba, designing a linear Transformer, or optimizing a full-attention model, the logic is clear: if your hardware loads a 128-token chunk at a time, your local mixer should exploit all 128 tokens.

It is time to retire (k=4) convolutions and embrace chunk-aligned ShortSWA as the new standard for local modeling.

If you want a one-line mental model: ShortConv is to ShortSWA what modern conv (ConvNeXt-style) is to ViT or Swin. It is a strong baseline that becomes structurally disadvantaged once dynamic, windowed attention is implemented efficiently at scale. In the chunked era, the winning local mixer is the one that is both hardware-aligned and content-adaptive, and that is precisely what ShortSWA provides.

## Citation

```bibtex
@article{zhang2025rethink,
  title = {Rethinking SWA},
  author = {Zhang, Yifan},
  journal = {yifanzhang-pro.github.io},
  year = {2025},
  month = {December},
  url = "https://github.com/yifanzhang-pro/Rethinking-SWA"
}
```
