---
layout: default
title: Architecture
permalink: /architecture/
---

<section class="page-header-section">
  <h1>System Architecture</h1>
  <p>How SiLens implements a vision-language model in hardwired silicon.</p>
</section>

<div class="content-wrapper">

<h2>Overview</h2>

<p>SiLens implements SmolVLM-256M as a single ASIC with three main components:</p>

<ul>
  <li><strong>Vision Encoder</strong> (SigLIP-B/16) — Processes images into visual tokens</li>
  <li><strong>Multimodal Projector</strong> — Maps vision space to language space</li>
  <li><strong>Language Model</strong> (SmolLM2-135M) — Generates text responses</li>
</ul>

<hr>

<h2>Model Components</h2>

<h3>Vision Encoder (SigLIP-B/16)</h3>

<ul>
  <li><strong>Parameters:</strong> 93M</li>
  <li><strong>Patch size:</strong> 16×16 pixels</li>
  <li><strong>Input size:</strong> 384×384</li>
  <li><strong>Hidden dimension:</strong> 768</li>
  <li><strong>Layers:</strong> 12 transformer blocks</li>
  <li><strong>Attention heads:</strong> 12</li>
  <li><strong>Output:</strong> 576 tokens</li>
</ul>

<h3>Multimodal Projector</h3>

<ul>
  <li><strong>Parameters:</strong> 18M</li>
  <li><strong>Input dimension:</strong> 768</li>
  <li><strong>Output dimension:</strong> 576</li>
  <li><strong>Type:</strong> Linear projection</li>
</ul>

<h3>Language Model (SmolLM2-135M)</h3>

<ul>
  <li><strong>Parameters:</strong> 135M</li>
  <li><strong>Hidden dimension:</strong> 576</li>
  <li><strong>Layers:</strong> 30 transformer blocks</li>
  <li><strong>Attention heads:</strong> 9</li>
  <li><strong>Vocabulary:</strong> 49,152 tokens</li>
  <li><strong>Max context:</strong> 8,192 tokens</li>
</ul>

<hr>

<h2>Hardwired Weight Encoding</h2>

<p>The key innovation: model weights become physical wire connections.</p>

<h3>Ternary Quantization</h3>

<p>SmolVLM-256M uses ternary weights {-1, 0, +1}:</p>

<ul>
  <li><strong>Weight = +1</strong> → Metal trace to VDD (power)</li>
  <li><strong>Weight = -1</strong> → Metal trace to GND (ground)</li>
  <li><strong>Weight = 0</strong> → No connection (implicit zero)</li>
</ul>

<h3>Why This Works</h3>

<p><strong>Traditional approach:</strong> Weights stored in memory, loaded for each computation. Memory bandwidth becomes the bottleneck.</p>

<p><strong>SiLens approach:</strong> Weights ARE the circuit. No memory access needed. Computation happens at wire speed (nanoseconds).</p>

<hr>

<h2>Data Flow</h2>

<ol>
  <li><strong>Image Input</strong> — 384×384×3 RGB image</li>
  <li><strong>Patch Extraction</strong> — Split into 24×24 patches of 16×16 pixels</li>
  <li><strong>Patch Embedding</strong> — Convert to 576 tokens × 768 dimensions</li>
  <li><strong>Vision Transformer</strong> — Process through 12 layers</li>
  <li><strong>Projection</strong> — Map from 768 to 576 dimensions</li>
  <li><strong>Concatenation</strong> — Combine with text tokens</li>
  <li><strong>Language Model</strong> — Process through 30 layers</li>
  <li><strong>Token Generation</strong> — Autoregressive output</li>
  <li><strong>Output Text</strong> — Final response</li>
</ol>

<hr>

<h2>Physical Design</h2>

<h3>Die Specifications</h3>

<ul>
  <li><strong>Process:</strong> SkyWater SKY130 (130nm)</li>
  <li><strong>Die size:</strong> ~800mm²</li>
  <li><strong>Metal layers:</strong> 5</li>
  <li><strong>Core voltage:</strong> 1.8V</li>
  <li><strong>I/O voltage:</strong> 3.3V</li>
  <li><strong>Clock:</strong> 100-200 MHz</li>
</ul>

<h3>Area Breakdown</h3>

<ul>
  <li><strong>Vision encoder:</strong> 280mm² (35%)</li>
  <li><strong>Language model:</strong> 400mm² (50%)</li>
  <li><strong>Projector:</strong> 55mm² (7%)</li>
  <li><strong>PCIe + I/O:</strong> 40mm² (5%)</li>
  <li><strong>Power/clocking:</strong> 25mm² (3%)</li>
</ul>

<h3>Power Budget</h3>

<ul>
  <li><strong>Vision encoder:</strong> 8W</li>
  <li><strong>Language model:</strong> 12W</li>
  <li><strong>PCIe PHY:</strong> 2W</li>
  <li><strong>Clock/control:</strong> 2W</li>
  <li><strong>Margin:</strong> 1W</li>
  <li><strong>Total:</strong> 25W</li>
</ul>

<hr>

<h2>PCIe Interface</h2>

<h3>Specifications</h3>

<ul>
  <li><strong>Standard:</strong> PCIe 3.0</li>
  <li><strong>Lanes:</strong> x4</li>
  <li><strong>Bandwidth:</strong> 4 GB/s bidirectional</li>
  <li><strong>Power:</strong> Slot-powered (75W max available)</li>
</ul>

<h3>Register Map</h3>

<ul>
  <li><strong>0x000 CTRL</strong> — Control register</li>
  <li><strong>0x004 STATUS</strong> — Status/interrupt register</li>
  <li><strong>0x008 IMG_ADDR</strong> — Image buffer DMA address</li>
  <li><strong>0x00C IMG_SIZE</strong> — Image dimensions</li>
  <li><strong>0x010 OUT_ADDR</strong> — Output buffer DMA address</li>
  <li><strong>0x014 OUT_LEN</strong> — Output length</li>
  <li><strong>0x100 DMA_CTRL</strong> — DMA control</li>
  <li><strong>0x200+ DEBUG</strong> — Debug registers</li>
</ul>

<hr>

<h2>Performance Targets</h2>

<ul>
  <li><strong>Single-image latency:</strong> &lt;5ms</li>
  <li><strong>Throughput (pipelined):</strong> 200+ images/sec</li>
  <li><strong>Token generation:</strong> 50+ tokens/sec</li>
  <li><strong>Power efficiency:</strong> 8+ images/joule</li>
</ul>

<hr>

<div style="text-align: center; margin-top: 3rem;">
<a href="{{ site.baseurl }}/getting-started/" class="btn btn-primary">Get Started →</a>
<a href="{{ site.baseurl }}/docs/" class="btn btn-outline">View Documentation →</a>
</div>

</div>
