---
layout: default
title: Documentation
permalink: /docs/
---

<section class="page-header-section">
  <h1>Documentation</h1>
  <p>Technical guides for SiLens hardware and software.</p>
</section>

<div class="content-wrapper">

<h2>Core Documentation</h2>

<div class="features-grid">
<div class="feature-card">
<div class="feature-icon">📐</div>
<h3>Architecture</h3>
<p>System design and specifications.</p>
<a href="{{ site.baseurl }}/architecture/">Read more →</a>
</div>

<div class="feature-card">
<div class="feature-icon">🚀</div>
<h3>Getting Started</h3>
<p>Installation and setup.</p>
<a href="{{ site.baseurl }}/getting-started/">Read more →</a>
</div>

<div class="feature-card">
<div class="feature-icon">📖</div>
<h3>Quantization Guide</h3>
<p>Ternary quantization for SiLens.</p>
<a href="https://github.com/loreii/SiLens/blob/main/model/QUANTIZATION_GUIDE.md" target="_blank">Read more →</a>
</div>
</div>

<hr>

<h2>Model Conversion Tools</h2>

<p>Located in <code>model/conversion/</code>:</p>

<h3>Quantization Pipeline</h3>

<ul>
  <li><strong>analyze_model.py</strong> — Architecture analysis, weight statistics</li>
  <li><strong>extract_weights.py</strong> — Extract and organize weights</li>
  <li><strong>quantize_ternary.py</strong> — Ternary quantization</li>
  <li><strong>calibration.py</strong> — Calibration-aware quantization</li>
  <li><strong>mixed_precision.py</strong> — Keep critical layers higher precision</li>
  <li><strong>sensitivity_analysis.py</strong> — Layer sensitivity scoring</li>
</ul>

<h3>Validation Tools</h3>

<ul>
  <li><strong>validate_quantization.py</strong> — Quality validation</li>
  <li><strong>benchmark_suite.py</strong> — VQA, TextVQA benchmarks</li>
  <li><strong>perplexity_test.py</strong> — Language model perplexity</li>
  <li><strong>visual_qa_test.py</strong> — Visual QA accuracy</li>
  <li><strong>compare_outputs.py</strong> — Side-by-side comparison</li>
</ul>

<h3>Analysis Tools</h3>

<ul>
  <li><strong>weight_visualizer.py</strong> — Distribution plots</li>
  <li><strong>sparsity_analyzer.py</strong> — Sparsity patterns</li>
  <li><strong>outlier_detector.py</strong> — Outlier detection</li>
</ul>

<hr>

<h2>Hardware Documentation</h2>

<h3>FPGA Prototyping</h3>

<p>Located in <code>fpga/</code>:</p>

<p><strong>Xilinx:</strong></p>
<ul>
  <li>silens_fpga_wrapper.v</li>
  <li>silens_artix7.xdc</li>
  <li>silens_kintex7.xdc</li>
  <li>synth_vivado.tcl</li>
</ul>

<p><strong>Intel:</strong></p>
<ul>
  <li>silens_fpga_wrapper_intel.v</li>
  <li>silens_arria10.sdc</li>
  <li>silens_cyclone10.sdc</li>
</ul>

<h3>PCB Design</h3>

<p>Located in <code>pcb/</code> (coming soon):</p>
<ul>
  <li>Schematics</li>
  <li>Layout files</li>
  <li>Bill of Materials</li>
  <li>Assembly instructions</li>
</ul>

<hr>

<h2>Software Documentation</h2>

<h3>Linux Driver</h3>

<p>Located in <code>drivers/</code>:</p>
<ul>
  <li><strong>silens_drv.c</strong> — Main driver</li>
  <li><strong>silens_ioctl.h</strong> — IOCTL definitions</li>
  <li><strong>Makefile</strong> — Build instructions</li>
</ul>

<h3>Python SDK</h3>

<p>Install:</p>
<pre><code>pip install silens</code></pre>

<p>Usage:</p>
<pre><code>import silens
from PIL import Image

device = silens.Device()
image = Image.open("photo.jpg")

# Describe image
result = device.describe(image)

# Visual QA
answer = device.ask(image, "What color is the car?")</code></pre>

<h3>Firmware</h3>

<p>Located in <code>firmware/</code>:</p>
<ul>
  <li><strong>main.c</strong> — Main application</li>
  <li><strong>startup.S</strong> — Startup code</li>
  <li><strong>linker.ld</strong> — Linker script</li>
</ul>

<hr>

<h2>Specifications</h2>

<h3>ASIC</h3>

<ul>
  <li><strong>Model:</strong> SmolVLM-256M (246M parameters)</li>
  <li><strong>Vision:</strong> SigLIP-B/16 (93M)</li>
  <li><strong>Language:</strong> SmolLM2-135M (135M)</li>
  <li><strong>Process:</strong> SkyWater SKY130 (130nm)</li>
  <li><strong>Die Size:</strong> ~800mm²</li>
  <li><strong>Clock:</strong> 100-200 MHz</li>
</ul>

<h3>Card</h3>

<ul>
  <li><strong>Interface:</strong> PCIe 3.0 x4</li>
  <li><strong>Form Factor:</strong> Half-height, half-length</li>
  <li><strong>Dimensions:</strong> 168mm × 69mm</li>
  <li><strong>Power:</strong> 25W TDP (slot-powered)</li>
  <li><strong>Cooling:</strong> Passive heatsink</li>
</ul>

<h3>Software Support</h3>

<ul>
  <li><strong>Linux:</strong> Full support</li>
  <li><strong>Windows:</strong> Planned</li>
  <li><strong>macOS:</strong> Not supported (no PCIe)</li>
  <li><strong>Docker:</strong> Official images</li>
</ul>

<hr>

<h2>Contributing</h2>

<p>See <a href="https://github.com/loreii/SiLens/blob/main/CONTRIBUTING.md" target="_blank">CONTRIBUTING.md</a></p>

<h3>Areas Needing Help</h3>

<ul>
  <li>RTL design for transformer blocks</li>
  <li>FPGA prototyping</li>
  <li>PCB design review</li>
  <li>Driver development</li>
  <li>Documentation</li>
</ul>

<hr>

<h2>License</h2>

<p>Apache License 2.0</p>

<h3>Third-Party</h3>

<ul>
  <li><strong>SmolVLM-256M</strong> — Apache 2.0 (Hugging Face)</li>
  <li><strong>SkyWater SKY130</strong> — Apache 2.0 (Google/SkyWater)</li>
  <li><strong>OpenLane</strong> — Apache 2.0 (Efabless)</li>
</ul>

</div>
