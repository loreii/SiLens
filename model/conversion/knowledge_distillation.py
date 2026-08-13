#!/usr/bin/env python3
"""
Knowledge Distillation for Ternary Model Training.

This module implements knowledge distillation techniques to improve the accuracy
of ternary-quantized models by learning from a full-precision teacher model.

Distillation Methods:
1. Output Distillation - Match teacher's final logits
2. Feature Distillation - Match intermediate representations  
3. Attention Transfer - Match attention patterns
4. Progressive Distillation - Gradually increase quantization

Theory:
    The student (ternary) model learns to mimic the teacher (FP16/FP32) model.
    Loss = α * CE(student, labels) + β * KL(student_logits, teacher_logits) 
         + γ * MSE(student_features, teacher_features)

Usage:
    python knowledge_distillation.py \\
        --teacher HuggingFaceTB/SmolVLM-256M-Instruct \\
        --student ./quantized_model \\
        --dataset ./calibration_data \\
        --output ./distilled_model

Author: SiLens AI/ML Team
License: Apache 2.0
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    # Loss weights
    task_loss_weight: float = 1.0          # α - weight for task (CE) loss
    distill_loss_weight: float = 2.0       # β - weight for distillation loss
    feature_loss_weight: float = 0.5       # γ - weight for feature matching
    attention_loss_weight: float = 0.1     # weight for attention transfer
    
    # Temperature for softening probabilities
    temperature: float = 4.0
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Progressive distillation
    progressive: bool = True
    progressive_stages: int = 4
    
    # Feature matching layers
    feature_layers: List[str] = field(default_factory=lambda: [
        'vision_encoder.encoder.layers.5',
        'vision_encoder.encoder.layers.11',
        'language_model.model.layers.14',
        'language_model.model.layers.29'
    ])
    
    # Attention layers to match
    attention_layers: List[str] = field(default_factory=lambda: [
        'vision_encoder.encoder.layers.11.self_attn',
        'language_model.model.layers.29.self_attn'
    ])


class SoftTargetLoss(nn.Module):
    """
    Soft target distillation loss using KL divergence.
    
    L_distill = KL(softmax(student_logits/T), softmax(teacher_logits/T)) * T^2
    
    The T^2 scaling ensures gradients have similar magnitude regardless of T.
    """
    
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_logits: torch.Tensor, 
                teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute soft target distillation loss.
        
        Args:
            student_logits: Logits from student model [B, seq_len, vocab_size]
            teacher_logits: Logits from teacher model [B, seq_len, vocab_size]
            
        Returns:
            Scalar loss value
        """
        T = self.temperature
        
        # Soften probabilities
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)
        
        # KL divergence
        loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T ** 2)
        
        return loss


class FeatureMatchingLoss(nn.Module):
    """
    Feature matching loss between student and teacher intermediate representations.
    
    Uses MSE with optional projection if dimensions differ.
    """
    
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        
        # Add projection if dimensions differ
        if student_dim != teacher_dim:
            self.projector = nn.Linear(student_dim, teacher_dim)
        else:
            self.projector = nn.Identity()
        
        self.mse = nn.MSELoss()
    
    def forward(self, student_features: torch.Tensor,
                teacher_features: torch.Tensor) -> torch.Tensor:
        """
        Compute feature matching loss.
        
        Args:
            student_features: Features from student [B, seq_len, student_dim]
            teacher_features: Features from teacher [B, seq_len, teacher_dim]
            
        Returns:
            Scalar loss value
        """
        # Project student features if needed
        projected = self.projector(student_features)
        
        # Normalize features for stable training
        projected_norm = F.normalize(projected, dim=-1)
        teacher_norm = F.normalize(teacher_features, dim=-1)
        
        return self.mse(projected_norm, teacher_norm)


class AttentionTransferLoss(nn.Module):
    """
    Attention transfer loss - matches attention patterns between student and teacher.
    
    Based on "Paying More Attention to Attention" (Zagoruyko & Komodakis, 2017)
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
    
    def forward(self, student_attention: torch.Tensor,
                teacher_attention: torch.Tensor) -> torch.Tensor:
        """
        Compute attention transfer loss.
        
        Args:
            student_attention: Attention weights [B, heads, seq, seq]
            teacher_attention: Attention weights [B, heads, seq, seq]
            
        Returns:
            Scalar loss value
        """
        # Average over heads
        student_avg = student_attention.mean(dim=1)  # [B, seq, seq]
        teacher_avg = teacher_attention.mean(dim=1)
        
        if self.normalize:
            # Normalize attention maps
            student_avg = student_avg / (student_avg.sum(dim=-1, keepdim=True) + 1e-8)
            teacher_avg = teacher_avg / (teacher_avg.sum(dim=-1, keepdim=True) + 1e-8)
        
        # MSE loss
        return F.mse_loss(student_avg, teacher_avg)


class TernaryQuantizationSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for ternary quantization.
    
    Forward: Hard ternary quantization
    Backward: Gradient passes through unchanged (STE)
    """
    
    @staticmethod
    def forward(ctx, weights: torch.Tensor, alpha: float) -> torch.Tensor:
        """Quantize to ternary values."""
        threshold = alpha * weights.abs().mean()
        
        quantized = torch.zeros_like(weights)
        quantized[weights > threshold] = 1.0
        quantized[weights < -threshold] = -1.0
        
        # Save for backward
        ctx.save_for_backward(weights)
        ctx.threshold = threshold
        
        return quantized
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """STE: pass gradients through unchanged."""
        weights, = ctx.saved_tensors
        
        # Gradient clipping for values outside threshold
        # This helps stabilize training
        grad_input = grad_output.clone()
        grad_input[weights.abs() > 2 * ctx.threshold] = 0
        
        return grad_input, None


def ternary_quantize_ste(weights: torch.Tensor, alpha: float = 0.7) -> torch.Tensor:
    """Apply ternary quantization with STE."""
    return TernaryQuantizationSTE.apply(weights, alpha)


class TernaryLinear(nn.Module):
    """
    Linear layer with ternary weights using STE for training.
    
    During training: uses STE for gradients through quantization
    During inference: uses pre-quantized weights
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, alpha: float = 0.7):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        # Full precision weights for training
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Scale factor (learned)
        self.scale = nn.Parameter(torch.ones(1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize weights with STE
        quantized_weight = ternary_quantize_ste(self.weight, self.alpha)
        
        # Scale quantized weights
        scaled_weight = quantized_weight * self.scale
        
        return F.linear(x, scaled_weight, self.bias)
    
    def get_quantized_weight(self) -> torch.Tensor:
        """Get quantized weights for export."""
        with torch.no_grad():
            threshold = self.alpha * self.weight.abs().mean()
            quantized = torch.zeros_like(self.weight)
            quantized[self.weight > threshold] = 1.0
            quantized[self.weight < -threshold] = -1.0
            return quantized


class KnowledgeDistillationTrainer:
    """
    Trainer for knowledge distillation from teacher to ternary student.
    
    Handles:
    - Loading teacher and student models
    - Setting up distillation losses
    - Progressive quantization scheduling
    - Training loop with feature/attention matching
    """
    
    def __init__(self, 
                 teacher_model_path: str,
                 student_model_path: Optional[str] = None,
                 config: Optional[DistillationConfig] = None,
                 device: str = 'cuda'):
        """
        Initialize distillation trainer.
        
        Args:
            teacher_model_path: Path to teacher model (full precision)
            student_model_path: Path to student model (or None to create from teacher)
            config: Distillation configuration
            device: Device to train on
        """
        self.teacher_path = teacher_model_path
        self.student_path = student_model_path
        self.config = config or DistillationConfig()
        self.device = device
        
        self.teacher = None
        self.student = None
        self.optimizer = None
        self.scheduler = None
        
        # Losses
        self.soft_target_loss = SoftTargetLoss(self.config.temperature)
        self.feature_losses: Dict[str, FeatureMatchingLoss] = {}
        self.attention_loss = AttentionTransferLoss()
        
        # Hooks for capturing intermediate features
        self.teacher_features: Dict[str, torch.Tensor] = {}
        self.student_features: Dict[str, torch.Tensor] = {}
        self.teacher_attentions: Dict[str, torch.Tensor] = {}
        self.student_attentions: Dict[str, torch.Tensor] = {}
        
    def load_models(self) -> None:
        """Load teacher and student models."""
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            raise ImportError("transformers not installed")
        
        logger.info(f"Loading teacher model: {self.teacher_path}")
        self.teacher = AutoModelForVision2Seq.from_pretrained(
            self.teacher_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device)
        self.teacher.eval()
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Create or load student
        if self.student_path:
            logger.info(f"Loading student model: {self.student_path}")
            self.student = AutoModelForVision2Seq.from_pretrained(
                self.student_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(self.device)
        else:
            logger.info("Creating student from teacher...")
            self.student = AutoModelForVision2Seq.from_pretrained(
                self.teacher_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(self.device)
        
        # Replace linear layers with ternary versions
        self._convert_to_ternary()
        
        # Register hooks for feature/attention capture
        self._register_hooks()
    
    def _convert_to_ternary(self) -> None:
        """Convert student linear layers to ternary."""
        def replace_linear(module: nn.Module, name: str = '') -> None:
            for child_name, child in module.named_children():
                full_name = f"{name}.{child_name}" if name else child_name
                
                if isinstance(child, nn.Linear):
                    # Skip certain layers (embeddings, final head)
                    if any(skip in full_name.lower() for skip in 
                           ['embed', 'lm_head', 'layernorm', 'norm']):
                        continue
                    
                    # Replace with ternary linear
                    ternary_layer = TernaryLinear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None
                    )
                    
                    # Copy weights
                    ternary_layer.weight.data = child.weight.data.clone()
                    if child.bias is not None:
                        ternary_layer.bias.data = child.bias.data.clone()
                    
                    setattr(module, child_name, ternary_layer)
                else:
                    replace_linear(child, full_name)
        
        replace_linear(self.student)
        logger.info("Converted student to ternary architecture")

    def _register_hooks(self) -> None:
        """Register forward hooks to capture intermediate features."""
        
        def make_feature_hook(storage: Dict, name: str):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                storage[name] = output.detach()
            return hook
        
        def make_attention_hook(storage: Dict, name: str):
            def hook(module, input, output):
                # Attention output typically includes attention weights
                if isinstance(output, tuple) and len(output) > 1:
                    storage[name] = output[1].detach()  # attention weights
            return hook
        
        # Register feature hooks
        for layer_name in self.config.feature_layers:
            try:
                teacher_layer = self._get_module_by_name(self.teacher, layer_name)
                student_layer = self._get_module_by_name(self.student, layer_name)
                
                teacher_layer.register_forward_hook(
                    make_feature_hook(self.teacher_features, layer_name))
                student_layer.register_forward_hook(
                    make_feature_hook(self.student_features, layer_name))
                
                logger.info(f"Registered feature hooks for: {layer_name}")
            except Exception as e:
                logger.warning(f"Could not register hook for {layer_name}: {e}")
        
        # Register attention hooks
        for layer_name in self.config.attention_layers:
            try:
                teacher_layer = self._get_module_by_name(self.teacher, layer_name)
                student_layer = self._get_module_by_name(self.student, layer_name)
                
                teacher_layer.register_forward_hook(
                    make_attention_hook(self.teacher_attentions, layer_name))
                student_layer.register_forward_hook(
                    make_attention_hook(self.student_attentions, layer_name))
                
                logger.info(f"Registered attention hooks for: {layer_name}")
            except Exception as e:
                logger.warning(f"Could not register hook for {layer_name}: {e}")
    
    def _get_module_by_name(self, model: nn.Module, name: str) -> nn.Module:
        """Get a submodule by its dotted name."""
        parts = name.split('.')
        module = model
        for part in parts:
            module = getattr(module, part)
        return module
    
    def setup_optimizer(self, num_training_steps: int) -> None:
        """Setup optimizer and learning rate scheduler."""
        # Only optimize student parameters
        optimizer_params = [
            {'params': [p for n, p in self.student.named_parameters() 
                       if 'bias' not in n and 'norm' not in n.lower()],
             'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.student.named_parameters() 
                       if 'bias' in n or 'norm' in n.lower()],
             'weight_decay': 0.0}
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=self.config.learning_rate
        )
        
        # Linear warmup then cosine decay
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (num_training_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda)
    
    def compute_loss(self, 
                     student_outputs,
                     teacher_outputs,
                     labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses = {}
        
        # Task loss (cross-entropy with hard labels)
        if labels is not None:
            task_loss = F.cross_entropy(
                student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            losses['task'] = task_loss.item()
        else:
            task_loss = torch.tensor(0.0, device=self.device)
        
        # Soft target distillation loss
        distill_loss = self.soft_target_loss(
            student_outputs.logits,
            teacher_outputs.logits
        )
        losses['distill'] = distill_loss.item()
        
        # Feature matching losses
        feature_loss = torch.tensor(0.0, device=self.device)
        for layer_name in self.teacher_features:
            if layer_name in self.student_features:
                t_feat = self.teacher_features[layer_name]
                s_feat = self.student_features[layer_name]
                
                # Create feature loss module if needed
                if layer_name not in self.feature_losses:
                    self.feature_losses[layer_name] = FeatureMatchingLoss(
                        s_feat.size(-1), t_feat.size(-1)
                    ).to(self.device)
                
                fl = self.feature_losses[layer_name](s_feat, t_feat)
                feature_loss = feature_loss + fl
        
        if self.teacher_features:
            feature_loss = feature_loss / len(self.teacher_features)
        losses['feature'] = feature_loss.item()
        
        # Attention transfer loss
        attention_loss = torch.tensor(0.0, device=self.device)
        attn_count = 0
        for layer_name in self.teacher_attentions:
            if layer_name in self.student_attentions:
                t_attn = self.teacher_attentions[layer_name]
                s_attn = self.student_attentions[layer_name]
                
                al = self.attention_loss(s_attn, t_attn)
                attention_loss = attention_loss + al
                attn_count += 1
        
        if attn_count > 0:
            attention_loss = attention_loss / attn_count
        losses['attention'] = attention_loss.item()
        
        # Combine losses
        total_loss = (
            self.config.task_loss_weight * task_loss +
            self.config.distill_loss_weight * distill_loss +
            self.config.feature_loss_weight * feature_loss +
            self.config.attention_loss_weight * attention_loss
        )
        losses['total'] = total_loss.item()
        
        return total_loss, losses

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.student.train()
        
        total_losses = {}
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass through teacher (no grad)
            with torch.no_grad():
                teacher_outputs = self.teacher(**batch)
            
            # Forward pass through student
            student_outputs = self.student(**batch)
            
            # Compute loss
            labels = batch.get('labels', None)
            loss, loss_dict = self.compute_loss(
                student_outputs, teacher_outputs, labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Accumulate losses
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0) + v
            num_batches += 1
            
            # Optimizer step
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), 
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Logging
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss_dict['total']:.4f} "
                    f"(task: {loss_dict['task']:.4f}, "
                    f"distill: {loss_dict['distill']:.4f})"
                )
            
            # Clear feature caches
            self.teacher_features.clear()
            self.student_features.clear()
            self.teacher_attentions.clear()
            self.student_attentions.clear()
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    def train(self, train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None,
              output_dir: str = './distilled_model') -> None:
        """
        Run full distillation training.
        
        Args:
            train_dataloader: Training data
            val_dataloader: Optional validation data
            output_dir: Directory to save checkpoints
        """
        os.makedirs(output_dir, exist_ok=True)
        
        num_training_steps = (
            len(train_dataloader) * self.config.num_epochs // 
            self.config.gradient_accumulation_steps
        )
        
        self.setup_optimizer(num_training_steps)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*60}")
            
            # Progressive quantization: gradually decrease alpha
            if self.config.progressive:
                stage = min(epoch, self.config.progressive_stages - 1)
                alpha = 0.9 - (0.2 * stage / (self.config.progressive_stages - 1))
                self._update_alpha(alpha)
                logger.info(f"Progressive alpha: {alpha:.3f}")
            
            # Train
            train_losses = self.train_epoch(train_dataloader, epoch)
            logger.info(f"Train losses: {train_losses}")
            
            # Validate
            if val_dataloader:
                val_losses = self.evaluate(val_dataloader)
                logger.info(f"Val losses: {val_losses}")
                
                # Save best model
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    self.save_model(os.path.join(output_dir, 'best_model'))
            
            # Save checkpoint
            self.save_model(os.path.join(output_dir, f'checkpoint_epoch{epoch}'))
        
        # Save final model
        self.save_model(os.path.join(output_dir, 'final_model'))
        logger.info(f"Training complete. Models saved to {output_dir}")
    
    def _update_alpha(self, alpha: float) -> None:
        """Update alpha for all ternary layers."""
        for module in self.student.modules():
            if isinstance(module, TernaryLinear):
                module.alpha = alpha
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.student.eval()
        
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                teacher_outputs = self.teacher(**batch)
                student_outputs = self.student(**batch)
                
                labels = batch.get('labels', None)
                _, loss_dict = self.compute_loss(
                    student_outputs, teacher_outputs, labels)
                
                for k, v in loss_dict.items():
                    total_losses[k] = total_losses.get(k, 0) + v
                num_batches += 1
        
        return {k: v / num_batches for k, v in total_losses.items()}
    
    def save_model(self, path: str) -> None:
        """Save student model and quantized weights."""
        os.makedirs(path, exist_ok=True)
        
        # Save full model
        self.student.save_pretrained(path)
        
        # Export quantized weights
        quantized_weights = {}
        for name, module in self.student.named_modules():
            if isinstance(module, TernaryLinear):
                quantized_weights[name] = {
                    'weight': module.get_quantized_weight().cpu().numpy(),
                    'scale': module.scale.detach().cpu().numpy(),
                    'alpha': module.alpha
                }
        
        np.savez(os.path.join(path, 'quantized_weights.npz'), **quantized_weights)
        
        # Save config
        config_dict = {
            'distillation_config': self.config.__dict__,
            'num_ternary_layers': len(quantized_weights)
        }
        with open(os.path.join(path, 'distillation_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation for SiLens')
    parser.add_argument('--teacher', required=True, help='Teacher model path')
    parser.add_argument('--student', default=None, help='Student model path')
    parser.add_argument('--output', default='./distilled', help='Output directory')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    config = DistillationConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature
    )
    
    trainer = KnowledgeDistillationTrainer(
        teacher_model_path=args.teacher,
        student_model_path=args.student,
        config=config,
        device=args.device
    )
    
    trainer.load_models()
    
    # Note: User needs to provide actual dataloader
    logger.info("Trainer initialized. Provide a DataLoader to start training.")


if __name__ == '__main__':
    main()
