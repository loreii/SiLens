---
layout: default
title: FAQ
permalink: /faq/
---

<section class="page-header-section">
  <h1>Frequently Asked Questions</h1>
  <p>Everything you need to know about SiLens.</p>
</section>

<div class="content-wrapper">

<h2>Product & Technology</h2>

<h3>What is SiLens?</h3>

<p>A PCIe accelerator card that runs SmolVLM-256M with weights <strong>physically etched into silicon</strong>. This eliminates memory bottlenecks for extremely fast, low-power inference.</p>

<h3>What does "hardwired" mean?</h3>

<p>Traditional accelerators store weights in memory and load them for each computation. In SiLens, each weight is a physical wire:</p>

<ul>
  <li><strong>Weight = +1</strong> → Wire to VDD (power)</li>
  <li><strong>Weight = -1</strong> → Wire to GND (ground)</li>
  <li><strong>Weight = 0</strong> → No connection</li>
</ul>

<p>The model IS the circuit—no memory access needed.</p>

<h3>Can I run different models?</h3>

<p>No. Weights are physically etched and cannot be changed. SiLens is purpose-built for SmolVLM-256M.</p>

<h3>Is SiLens a GPU?</h3>

<p>No. It's an <strong>inference-only ASIC</strong>:</p>
<ul>
  <li>Cannot run arbitrary programs</li>
  <li>Cannot train models</li>
  <li>Runs only the hardwired model</li>
  <li>Has no general-purpose memory</li>
</ul>

<h3>Why 130nm?</h3>

<ol>
  <li><strong>It's open</strong> — SkyWater SKY130 is the only fully open-source PDK</li>
  <li><strong>It's affordable</strong> — ~$100K masks vs $10M+ for modern nodes</li>
  <li><strong>It's sufficient</strong> — Our architecture isn't compute-bound</li>
</ol>

<hr>

<h2>Performance</h2>

<h3>How fast is SiLens?</h3>

<ul>
  <li><strong>Latency:</strong> &lt;5ms single image</li>
  <li><strong>Throughput:</strong> 200+ images/sec</li>
  <li><strong>Pipelined:</strong> 1000+ images/sec</li>
</ul>

<h3>Compared to RTX 4060?</h3>

<ul>
  <li><strong>Price:</strong> $149-249 vs $299 (20-50% cheaper)</li>
  <li><strong>Latency:</strong> &lt;5ms vs 300-1000ms (60-200× faster)</li>
  <li><strong>Throughput:</strong> 200+ vs 1-3 img/sec (100× faster)</li>
  <li><strong>Power:</strong> 25W vs 115W (4.6× more efficient)</li>
</ul>

<h3>Why so much faster?</h3>

<p>GPUs are limited by memory bandwidth, not compute. When running SmolVLM-256M on a GPU:</p>
<ul>
  <li>Model (500MB) sits in VRAM</li>
  <li>Weights loaded from memory each token</li>
  <li>Memory bandwidth is the bottleneck</li>
  <li>GPU compute utilization &lt;5%</li>
</ul>

<p>SiLens eliminates this—weights are circuits, not data.</p>

<h3>Can I train on SiLens?</h3>

<p>No. Inference only. Use GPUs/cloud for training.</p>

<hr>

<h2>Compatibility</h2>

<h3>Operating Systems?</h3>

<ul>
  <li><strong>Linux (Ubuntu 20.04+):</strong> ✅ Full support</li>
  <li><strong>Windows 10/11:</strong> 🟡 Planned</li>
  <li><strong>macOS:</strong> ❌ Not supported (no PCIe)</li>
</ul>

<h3>What PCIe slot?</h3>

<p>Requires <strong>PCIe 3.0 x4</strong> or higher:</p>
<ul>
  <li>✅ x4, x8, x16 slots (3.0/4.0/5.0)</li>
  <li>❌ x1 slots</li>
  <li>❌ M.2 slots</li>
  <li>❌ USB</li>
</ul>

<h3>External power needed?</h3>

<p>No. 25W from PCIe slot. No cables needed.</p>

<h3>Multiple cards?</h3>

<p>Yes! Up to 8 cards per system for:</p>
<ul>
  <li>Higher throughput</li>
  <li>Redundancy</li>
  <li>Load balancing</li>
</ul>

<h3>Programming languages?</h3>

<ul>
  <li><strong>Python:</strong> Official SDK (<code>pip install silens</code>)</li>
  <li><strong>C/C++:</strong> Native library</li>
  <li><strong>Others:</strong> Community bindings welcome</li>
</ul>

<hr>

<h2>Technical Details</h2>

<h3>What model?</h3>

<p><strong>SmolVLM-256M</strong> (246M parameters):</p>
<ul>
  <li>Vision: SigLIP-B/16 (93M)</li>
  <li>Language: SmolLM2-135M (135M)</li>
  <li>Projector: 18M</li>
</ul>

<h3>What can it do?</h3>

<p>✅ Supported:</p>
<ul>
  <li>Describe images</li>
  <li>Answer questions about images</li>
  <li>Read text in images (OCR)</li>
  <li>Compare images</li>
  <li>Process video frames</li>
</ul>

<p>❌ Not supported:</p>
<ul>
  <li>Generate images</li>
  <li>Complex reasoning</li>
  <li>Long context (&gt;2K tokens)</li>
</ul>

<h3>ASIC specs?</h3>

<ul>
  <li><strong>Process:</strong> SkyWater SKY130 (130nm)</li>
  <li><strong>Die size:</strong> ~800mm²</li>
  <li><strong>Voltage:</strong> 1.8V core, 3.3V I/O</li>
  <li><strong>Clock:</strong> 100-200 MHz</li>
  <li><strong>Package:</strong> BGA-625</li>
</ul>

<h3>Manufacturing yield?</h3>

<p>At 800mm², expected 30-50% yield. We've:</p>
<ul>
  <li>Priced conservatively (30% assumption)</li>
  <li>Added redundancy where possible</li>
  <li>Partnered with SkyWater on optimization</li>
</ul>

<hr>

<h2>Future Plans</h2>

<h3>Gen 2?</h3>

<p>Yes! Roadmap:</p>
<ul>
  <li><strong>Gen 1 (2028):</strong> 130nm, SmolVLM-256M</li>
  <li><strong>Gen 1.5 (2029):</strong> 65nm, 2× speed, 50% power</li>
  <li><strong>Gen 2 (2030):</strong> 45nm, SmolVLM-500M</li>
</ul>

<h3>USB or M.2 versions?</h3>

<p>Exploring based on demand:</p>
<ul>
  <li><strong>M.2:</strong> Lower power, smaller</li>
  <li><strong>USB:</strong> External enclosure</li>
</ul>

<h3>How to contribute?</h3>

<ul>
  <li>Test simulations</li>
  <li>Review designs</li>
  <li>Improve documentation</li>
  <li>Driver development</li>
  <li>Bug reports</li>
</ul>

<p>Join us on <a href="https://github.com/loreii/SiLens" target="_blank">GitHub</a>!</p>

<hr>

<h2>Contact</h2>

<div class="features-grid">
<div class="feature-card">
<div class="feature-icon">💬</div>
<h3>Discord</h3>
<p>Coming soon</p>
</div>

<div class="feature-card">
<div class="feature-icon">📧</div>
<h3>Email</h3>
<p><a href="mailto:hello@silens.ai">hello@silens.ai</a></p>
</div>

<div class="feature-card">
<div class="feature-icon">🐛</div>
<h3>GitHub</h3>
<p><a href="https://github.com/loreii/SiLens/issues" target="_blank">Issues & Requests</a></p>
</div>
</div>

</div>
