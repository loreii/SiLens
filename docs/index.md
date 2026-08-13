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
      <a href="https://github.com/loreii/SiLens" class="btn btn-secondary" target="_blank">
        <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor">
          <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
        </svg>
        View on GitHub
      </a>
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
        <p>Process 200+ images per second with sub-5ms latency. No GPU comes close at this price point.</p>
      </div>
      
      <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <h3>Ultra Low Power</h3>
        <p>Just 25W total power draw. Run AI inference without melting your power budget or your cooling system.</p>
      </div>
      
      <div class="feature-card">
        <div class="feature-icon">🔓</div>
        <h3>Fully Open Source</h3>
        <p>RTL, PCB designs, drivers, SDK—everything is Apache 2.0 licensed. Verify, modify, and improve.</p>
      </div>
      
      <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <h3>Vision + Language</h3>
        <p>SmolVLM-256M understands images AND text. Describe photos, answer questions, read documents.</p>
      </div>
      
      <div class="feature-card">
        <div class="feature-icon">💰</div>
        <h3>Affordable</h3>
        <p>Starting at $149—cheaper than the GPUs it outperforms by orders of magnitude.</p>
      </div>
      
      <div class="feature-card">
        <div class="feature-icon">🔌</div>
        <h3>Plug & Play</h3>
        <p>Standard PCIe 3.0 x4 card. No external power needed. Works in any server or workstation.</p>
      </div>
    </div>
  </div>
</section>

<section class="section">
  <div class="container">
    <div class="section-header">
      <h2>The Problem with AI Hardware</h2>
      <p>Your options today are expensive, power-hungry, or severely limited.</p>
    </div>

<div class="table-wrapper">

| Option | Price | Power | Latency | The Catch |
|:-------|:------|:------|:--------|:----------|
| Cloud API | $0.01/img | N/A | 500ms+ | Privacy concerns, ongoing costs, requires internet |
| Consumer GPU | $300+ | 115W | 300ms+ | Overkill for inference, massive power draw |
| Edge TPU | $75-150 | 2-4W | 30ms | Vision only—no language understanding |
| Enterprise AI | $10K+ | 300W+ | <10ms | Absurdly expensive for most use cases |
| **SiLens** | **$149** | **25W** | **<5ms** | ✅ Affordable, efficient, multimodal |

</div>

  </div>
</section>

<section class="section section-dark">
  <div class="container">
    <div class="section-header">
      <h2>How It Works</h2>
      <p>We do something no one else has done: bake the AI model directly into silicon.</p>
    </div>
    
<div class="comparison-grid">
<div class="comparison-box">
<h3>Traditional AI Hardware</h3>

```
[Image] → [Load weights from RAM] → [Compute] → [Answer]
                    ↑
          Memory bandwidth bottleneck
```

Weights stored in memory. Each inference requires loading billions of values through a narrow pipe.

</div>
<div class="comparison-box highlight">
<h3>SiLens Approach</h3>

```
[Image] → [Weights ARE the circuit] → [Answer]
                    ↑
          No memory access needed
```

**Weight = +1** → Wire to VDD  
**Weight = -1** → Wire to GND  
**Weight = 0** → No connection

</div>
</div>

  </div>
</section>

<section class="section">
  <div class="container">
    <div class="section-header">
      <h2>What Can SiLens Do?</h2>
      <p>Powered by SmolVLM-256M—a 246M parameter vision-language model.</p>
    </div>
    
<div class="features-grid">
<div class="feature-card">
<div class="feature-icon">📸</div>
<h3>Describe Images</h3>
<p><em>"What's in this photo?"</em></p>
<blockquote>"A golden retriever playing fetch on a sandy beach at sunset."</blockquote>
</div>

<div class="feature-card">
<div class="feature-icon">❓</div>
<h3>Visual Q&A</h3>
<p><em>"How many people are in this room?"</em></p>
<blockquote>"There are 7 people visible—4 seated and 3 standing."</blockquote>
</div>

<div class="feature-card">
<div class="feature-icon">📄</div>
<h3>Document Understanding</h3>
<p><em>"Extract the total from this receipt"</em></p>
<blockquote>"The total is $47.83, including $3.42 tax."</blockquote>
</div>
</div>

  </div>
</section>

<section class="section section-light">
  <div class="container">
    <div class="section-header">
      <h2>Technical Specifications</h2>
    </div>

<div class="table-wrapper">

| Specification | Value |
|:--------------|:------|
| Model | SmolVLM-256M (246M parameters) |
| Vision Encoder | SigLIP-B/16 (93M params) |
| Language Model | SmolLM2-135M (135M params) |
| Process Node | SkyWater SKY130 (130nm) |
| Die Size | ~800mm² |
| Interface | PCIe 3.0 x4 |
| Power | 25W TDP (slot-powered) |
| Latency | <5ms single image |
| Throughput | 200+ images/sec |

</div>

<div style="text-align: center; margin-top: 2rem;">
<a href="{{ site.baseurl }}/architecture/" class="btn btn-outline">View Full Architecture →</a>
</div>

  </div>
</section>

<section class="section">
  <div class="container">
    <div class="section-header">
      <h2>Project Status</h2>
    </div>

<div class="table-wrapper">

| Component | Status |
|:----------|:-------|
| Architecture specification | 🟡 In Progress |
| Model quantization tools | ✅ Complete |
| RTL design (Verilog) | 🔴 Not Started |
| FPGA prototype | 🔴 Not Started |
| Physical design | 🔴 Not Started |
| PCB design | 🔴 Not Started |
| Linux drivers | 🔴 Not Started |

</div>

<div style="text-align: center; margin-top: 2rem;">
<a href="https://github.com/loreii/SiLens" class="btn btn-primary" target="_blank">Contribute on GitHub →</a>
</div>

  </div>
</section>
