---
layout: default
title: Getting Started
permalink: /getting-started/
---

<section class="page-header-section">
  <h1>Getting Started</h1>
  <p>Set up your development environment and start exploring SiLens.</p>
</section>

<div class="content-wrapper">

<div class="alert alert-info">
  <span class="alert-icon">ℹ️</span>
  <div>
    <strong>Note:</strong> SiLens hardware is not yet available. These instructions help you work with the simulation and model conversion tools.
  </div>
</div>

<h2>Quick Start</h2>

<p>Clone the repository and run the demo:</p>

<pre><code>git clone https://github.com/loreii/SiLens.git
cd SiLens
pip install numpy
python demo.py</code></pre>

<h3>Demo Features</h3>

<ul>
  <li><strong>Ternary Quantization</strong> — Convert FP32 weights to 2-bit (16× compression)</li>
  <li><strong>Hardware Simulation</strong> — Interact with simulated accelerator</li>
  <li><strong>Performance Profiling</strong> — Timing and throughput analysis</li>
  <li><strong>Multi-Device Inference</strong> — Distributed batch processing</li>
  <li><strong>Sparse Attention</strong> — Attention pattern optimization</li>
  <li><strong>End-to-End Pipeline</strong> — Complete inference demo</li>
</ul>

<hr>

<h2>Prerequisites</h2>

<ul>
  <li><strong>Python 3.8+</strong> with pip</li>
  <li><strong>Git</strong> for cloning</li>
  <li><strong>8GB+ RAM</strong> recommended</li>
  <li><strong>NVIDIA GPU</strong> (optional, for faster analysis)</li>
</ul>

<hr>

<h2>Full Installation</h2>

<h3>1. Clone Repository</h3>

<pre><code>git clone https://github.com/loreii/SiLens.git
cd SiLens
git submodule update --init --recursive</code></pre>

<h3>2. Create Python Environment</h3>

<pre><code>python -m venv venv
source venv/bin/activate
pip install -r requirements.txt</code></pre>

<h3>3. Download Model</h3>

<pre><code>python tools/download_model.py</code></pre>

<hr>

<h2>Model Quantization</h2>

<p>Convert FP32 model to ternary weights:</p>

<h3>Step 1: Analyze</h3>

<pre><code>python model/conversion/analyze_model.py --model HuggingFaceTB/SmolVLM-256M-Instruct</code></pre>

<h3>Step 2: Sensitivity Analysis</h3>

<pre><code>python model/conversion/sensitivity_analysis.py --model HuggingFaceTB/SmolVLM-256M-Instruct</code></pre>

<h3>Step 3: Quantize</h3>

<pre><code>python model/conversion/quantize_ternary.py --model HuggingFaceTB/SmolVLM-256M-Instruct --alpha 0.7 --mode per_tensor --export --output ./model/weights/quantized</code></pre>

<h3>Step 4: Validate</h3>

<pre><code>python model/conversion/validate_quantization.py --model HuggingFaceTB/SmolVLM-256M-Instruct --quantized ./model/weights/quantized</code></pre>

<hr>

<h2>Expected Results</h2>

<p>With default settings (α=0.7):</p>

<ul>
  <li><strong>Memory:</strong> 1024 MB → 64 MB (16× smaller)</li>
  <li><strong>VQA Accuracy:</strong> ~71% → ~67% (~4% drop)</li>
  <li><strong>Perplexity:</strong> ~15 → ~17 (~13% increase)</li>
  <li><strong>Cosine Similarity:</strong> 0.92</li>
</ul>

<hr>

<h2>Repository Structure</h2>

<ul>
  <li><strong>model/</strong> — Quantization and validation tools</li>
  <li><strong>rtl/</strong> — Verilog source (coming soon)</li>
  <li><strong>fpga/</strong> — FPGA prototypes</li>
  <li><strong>drivers/</strong> — Linux kernel driver</li>
  <li><strong>sdk/</strong> — Python SDK</li>
  <li><strong>firmware/</strong> — Card firmware</li>
  <li><strong>docs/</strong> — Documentation</li>
</ul>

<hr>

<h2>Next Steps</h2>

<div class="features-grid">
<div class="feature-card">
<div class="feature-icon">📐</div>
<h3>Architecture</h3>
<p>Understand the hardware design.</p>
<a href="{{ site.baseurl }}/architecture/">Learn more →</a>
</div>

<div class="feature-card">
<div class="feature-icon">📖</div>
<h3>Documentation</h3>
<p>API reference and tools.</p>
<a href="{{ site.baseurl }}/docs/">View docs →</a>
</div>

<div class="feature-card">
<div class="feature-icon">🤝</div>
<h3>Contributing</h3>
<p>Help build SiLens.</p>
<a href="https://github.com/loreii/SiLens/blob/main/CONTRIBUTING.md" target="_blank">Contribute →</a>
</div>
</div>

</div>
