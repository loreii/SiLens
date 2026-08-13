---
layout: default
title: Home
---

<section class="hero">
  <div class="hero-content">
    <div class="hero-badge">
      <span class="hero-badge-dot"></span>
      Open Source • Apache 2.0 License
    </div>
    
    <h1>The World's First<br><span>Hardwired AI Accelerator</span></h1>
    
    <p class="hero-subtitle">
      SiLens etches neural network weights directly into silicon. No memory bottlenecks. 
      No GPUs. Just 200+ images per second at under 5ms latency—in 25 watts.
    </p>
    
    <div class="hero-stats">
      <div class="hero-stat">
        <div class="hero-stat-value">100×</div>
        <div class="hero-stat-label">Faster than GPU</div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-value">25W</div>
        <div class="hero-stat-label">Power Draw</div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-value">&lt;5ms</div>
        <div class="hero-stat-label">Latency</div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-value">$149</div>
        <div class="hero-stat-label">Starting Price</div>
      </div>
    </div>
    
    <div class="hero-buttons">
      <a href="{{ site.baseurl }}/getting-started/" class="btn btn-primary">Get Started</a>
      <a href="https://github.com/loreii/SiLens" class="btn btn-secondary" target="_blank">View on GitHub</a>
    </div>
  </div>
</section>

<section class="section">
  <div class="container">
    <div class="alert alert-warning">
      <span class="alert-icon">⚠️</span>
      <div>
        <strong>Early Development</strong> — SiLens is in the architectural design phase. Hardware is not yet available.
      </div>
    </div>
  </div>
</section>

<section class="section section-light">
  <div class="container">
    <div class="section-header">
      <h2>Why SiLens?</h2>
      <p>Traditional AI accelerators are bottlenecked by memory bandwidth. SiLens eliminates this entirely.</p>
    </div>
    
    <div class="features-grid">
      <div class="feature-card">
        <div class="feature-icon">🚀</div>
        <h3>100× Faster</h3>
        <p>Process 200+ images per second with sub-5ms latency.</p>
      </div>
      
      <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <h3>Ultra Low Power</h3>
        <p>Just 25W total power draw vs 115W for GPUs.</p>
      </div>
      
      <div class="feature-card">
        <div class="feature-icon">🔓</div>
        <h3>Fully Open Source</h3>
        <p>RTL, PCB, drivers, SDK—everything Apache 2.0.</p>
      </div>
      
      <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <h3>Vision + Language</h3>
        <p>Understands images AND text with SmolVLM-256M.</p>
      </div>
      
      <div class="feature-card">
        <div class="feature-icon">💰</div>
        <h3>Affordable</h3>
        <p>Starting at $149—cheaper than the GPUs it outperforms.</p>
      </div>
      
      <div class="feature-card">
        <div class="feature-icon">🔌</div>
        <h3>Plug & Play</h3>
        <p>Standard PCIe 3.0 x4 card. No external power needed.</p>
      </div>
    </div>
  </div>
</section>

<section class="section">
  <div class="container">
    <div class="section-header">
      <h2>How It Works</h2>
      <p>We bake the AI model directly into silicon.</p>
    </div>

<div class="features-grid">
<div class="feature-card">
<h3>Traditional AI Hardware</h3>
<p>Weights stored in memory. Each inference loads billions of values through a narrow memory bus—the bottleneck.</p>
</div>

<div class="feature-card">
<h3>SiLens Approach</h3>
<p><strong>Weight = +1</strong> → Wire to VDD<br>
<strong>Weight = -1</strong> → Wire to GND<br>
<strong>Weight = 0</strong> → No connection<br><br>
No memory access needed. Computation at wire speed.</p>
</div>
</div>

  </div>
</section>

<section class="section section-light">
  <div class="container">
    <div class="section-header">
      <h2>What Can SiLens Do?</h2>
      <p>Powered by SmolVLM-256M—a 246M parameter vision-language model.</p>
    </div>
    
<div class="features-grid">
<div class="feature-card">
<div class="feature-icon">📸</div>
<h3>Describe Images</h3>
<p><em>"What's in this photo?"</em><br>→ Detailed captions and descriptions</p>
</div>

<div class="feature-card">
<div class="feature-icon">❓</div>
<h3>Visual Q&A</h3>
<p><em>"How many people are here?"</em><br>→ Answer questions about images</p>
</div>

<div class="feature-card">
<div class="feature-icon">📄</div>
<h3>Document Understanding</h3>
<p><em>"Extract the total from this receipt"</em><br>→ OCR and document analysis</p>
</div>
</div>

  </div>
</section>

<section class="section">
  <div class="container">
    <div class="section-header">
      <h2>Technical Specifications</h2>
    </div>

<div class="features-grid">
<div class="feature-card">
<h3>Model</h3>
<ul>
<li>SmolVLM-256M (246M parameters)</li>
<li>Vision: SigLIP-B/16 (93M params)</li>
<li>Language: SmolLM2-135M (135M params)</li>
</ul>
</div>

<div class="feature-card">
<h3>Hardware</h3>
<ul>
<li>Process: SkyWater SKY130 (130nm)</li>
<li>Die Size: ~800mm²</li>
<li>Interface: PCIe 3.0 x4</li>
<li>Power: 25W TDP (slot-powered)</li>
</ul>
</div>

<div class="feature-card">
<h3>Performance</h3>
<ul>
<li>Latency: &lt;5ms single image</li>
<li>Throughput: 200+ images/sec</li>
<li>Token generation: 50+ tokens/sec</li>
</ul>
</div>
</div>

<div style="text-align: center; margin-top: 2rem;">
<a href="{{ site.baseurl }}/architecture/" class="btn btn-outline">View Full Architecture →</a>
</div>

  </div>
</section>

<section class="section section-light">
  <div class="container">
    <div class="section-header">
      <h2>Project Status</h2>
    </div>

<div class="features-grid">
<div class="feature-card">
<h3>✅ Complete</h3>
<ul>
<li>Model quantization tools</li>
<li>Architecture specification (in progress)</li>
</ul>
</div>

<div class="feature-card">
<h3>🔴 Not Started</h3>
<ul>
<li>RTL design (Verilog)</li>
<li>FPGA prototype</li>
<li>Physical design</li>
<li>PCB design</li>
<li>Linux drivers</li>
</ul>
</div>
</div>

<div style="text-align: center; margin-top: 2rem;">
<a href="https://github.com/loreii/SiLens" class="btn btn-primary" target="_blank">Contribute on GitHub →</a>
</div>

  </div>
</section>
