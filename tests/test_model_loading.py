"""
SiLens Model Loading Tests
==========================

Tests for verifying the SmolVLM-256M model can be loaded correctly.

Tests verify:
- Model files exist and are valid
- Model architecture matches expectations
- Model parameters count matches SmolVLM-256M spec
- Model can perform basic inference

Run with:
    pytest tests/test_model_loading.py -v
    pytest tests/test_model_loading.py -v --skip-model-tests  # Skip if model not downloaded
"""

import pytest
from pathlib import Path


# =============================================================================
# Model File Tests
# =============================================================================

class TestModelFiles:
    """Tests for model file presence and validity."""
    
    def test_model_directory_exists(self, model_path):
        """Verify model directory exists."""
        assert model_path.exists(), \
            f"Model directory not found: {model_path}. Run: make model"
    
    def test_config_json_exists(self, model_path, model_available):
        """Verify config.json exists."""
        if not model_available:
            pytest.skip("Model not downloaded")
        
        config_file = model_path / "config.json"
        assert config_file.exists(), "config.json not found"
    
    def test_model_weights_exist(self, model_path, model_available):
        """Verify model weight files exist."""
        if not model_available:
            pytest.skip("Model not downloaded")
        
        # Check for safetensors or pytorch bin files
        safetensors = list(model_path.glob("*.safetensors"))
        pytorch_bin = list(model_path.glob("*.bin"))
        
        assert safetensors or pytorch_bin, \
            "No model weight files found (*.safetensors or *.bin)"

    
    def test_processor_files_exist(self, model_path, model_available):
        """Verify processor configuration files exist."""
        if not model_available:
            pytest.skip("Model not downloaded")
        
        # Check for tokenizer files
        tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        
        for fname in tokenizer_files:
            fpath = model_path / fname
            # At least one tokenizer file should exist
            if fpath.exists():
                return
        
        # Check for alternative naming
        any_tokenizer = list(model_path.glob("*tokenizer*"))
        assert any_tokenizer, "No tokenizer files found"


# =============================================================================
# Model Configuration Tests
# =============================================================================

class TestModelConfig:
    """Tests for model configuration validity."""
    
    @pytest.mark.requires_model
    def test_config_has_required_fields(self, model_config):
        """Verify config has required fields."""
        assert model_config is not None, "Model config not loaded"
        
        # Check for common required fields
        # Note: Exact fields depend on model architecture
        common_fields = ['model_type', 'architectures']
        
        for field in common_fields:
            if field in model_config:
                return
        
        # At minimum, config should have some content
        assert len(model_config) > 0, "Config is empty"
    
    @pytest.mark.requires_model
    def test_config_model_type(self, model_config):
        """Verify model type is correct."""
        if model_config is None:
            pytest.skip("Model config not available")
        
        # SmolVLM uses a specific model type
        # Check for vision-language model indicators
        config_str = str(model_config).lower()
        
        vision_indicators = ['vision', 'vlm', 'image', 'multimodal']
        has_vision = any(ind in config_str for ind in vision_indicators)
        
        assert has_vision, "Config does not indicate vision-language model"


# =============================================================================
# Model Architecture Tests
# =============================================================================

class TestModelArchitecture:
    """Tests for model architecture validation."""
    
    @pytest.mark.requires_model
    def test_model_loads(self, hf_model):
        """Verify model loads without errors."""
        assert hf_model is not None, "Model failed to load"
    
    @pytest.mark.requires_model
    def test_model_has_parameters(self, hf_model):
        """Verify model has trainable parameters."""
        param_count = sum(p.numel() for p in hf_model.parameters())
        
        assert param_count > 0, "Model has no parameters"
        
        # SmolVLM-256M should have ~246M parameters
        # Allow some variance for different versions
        expected_min = 200_000_000  # 200M
        expected_max = 300_000_000  # 300M
        
        assert expected_min < param_count < expected_max, \
            f"Parameter count {param_count:,} outside expected range"
    
    @pytest.mark.requires_model
    def test_model_parameter_breakdown(self, weight_extractor):
        """Verify parameter breakdown by component."""
        counts = weight_extractor.count_parameters()
        
        # Should have vision encoder params
        assert counts['vision'] > 0 or counts['embeddings'] > 0, \
            "No vision encoder parameters found"
        
        # Should have language model params
        assert counts['llm'] > 0 or counts['embeddings'] > 0, \
            "No language model parameters found"
        
        print(f"\nParameter breakdown:")
        for comp, count in counts.items():
            if count > 0:
                pct = 100 * count / counts.get('total', count)
                print(f"  {comp}: {count:,} ({pct:.1f}%)")
    
    @pytest.mark.requires_model
    def test_vision_encoder_exists(self, hf_model):
        """Verify vision encoder component exists."""
        # Look for vision-related modules
        vision_found = False
        
        for name, module in hf_model.named_modules():
            if 'vision' in name.lower() or 'image' in name.lower():
                vision_found = True
                break
        
        # Also check for SigLIP-style encoder
        for name, module in hf_model.named_modules():
            if 'siglip' in name.lower() or 'clip' in name.lower():
                vision_found = True
                break
        
        assert vision_found, "Vision encoder not found in model"

    
    @pytest.mark.requires_model
    def test_language_model_exists(self, hf_model):
        """Verify language model component exists."""
        # Look for language model modules
        llm_found = False
        
        for name, module in hf_model.named_modules():
            if 'language' in name.lower() or 'lm_head' in name.lower():
                llm_found = True
                break
            # Also check for decoder layers
            if 'decoder' in name.lower() or 'layers' in name.lower():
                llm_found = True
                break
        
        assert llm_found, "Language model not found"


# =============================================================================
# Model Inference Tests
# =============================================================================

class TestModelInference:
    """Tests for model inference capability."""
    
    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_processor_loads(self, hf_processor):
        """Verify processor loads correctly."""
        assert hf_processor is not None, "Processor failed to load"
    
    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_basic_text_processing(self, hf_processor):
        """Verify processor can process text."""
        test_text = "Hello, world!"
        
        try:
            inputs = hf_processor(text=test_text, return_tensors="pt")
            assert 'input_ids' in inputs or hasattr(inputs, 'input_ids'), \
                "Processor did not return input_ids"
        except Exception as e:
            pytest.fail(f"Text processing failed: {e}")
    
    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_image_processing(self, hf_processor, sample_image):
        """Verify processor can process images."""
        try:
            # Process image with dummy text
            inputs = hf_processor(
                text="Describe this image.",
                images=[sample_image],
                return_tensors="pt"
            )
            
            # Should have pixel values for vision encoder
            has_pixels = (
                'pixel_values' in inputs or 
                hasattr(inputs, 'pixel_values')
            )
            
            assert has_pixels, "Processor did not return pixel_values"
            
        except Exception as e:
            pytest.fail(f"Image processing failed: {e}")

    
    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_forward_pass(self, hf_model, hf_processor, sample_image):
        """Verify model can perform a forward pass."""
        import torch
        
        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is in this image?"}
                ]
            }
        ]
        
        try:
            prompt = hf_processor.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            inputs = hf_processor(
                text=prompt, 
                images=[sample_image], 
                return_tensors="pt"
            )
            
            # Run forward pass
            with torch.no_grad():
                outputs = hf_model(**inputs)
            
            # Should have logits output
            assert hasattr(outputs, 'logits'), "No logits in output"
            
            # Logits should have reasonable shape
            # (batch_size, seq_len, vocab_size)
            assert len(outputs.logits.shape) == 3, \
                f"Unexpected logits shape: {outputs.logits.shape}"
            
        except Exception as e:
            pytest.fail(f"Forward pass failed: {e}")
    
    @pytest.mark.requires_model
    @pytest.mark.slow
    def test_generation(self, hf_model, hf_processor, sample_image):
        """Verify model can generate text."""
        import torch
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe this image briefly."}
                ]
            }
        ]
        
        try:
            prompt = hf_processor.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            inputs = hf_processor(
                text=prompt, 
                images=[sample_image], 
                return_tensors="pt"
            )
            
            # Generate
            with torch.no_grad():
                generated_ids = hf_model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False
                )
            
            # Decode
            output_text = hf_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Should generate some text
            assert len(output_text) > 0, "No text generated"
            
            print(f"\nGenerated: {output_text[-100:]}")
            
        except Exception as e:
            pytest.fail(f"Generation failed: {e}")


# =============================================================================
# Regression Tests
# =============================================================================

class TestModelRegression:
    """
    Regression tests comparing against golden references.
    
    These tests ensure model behavior doesn't change unexpectedly
    across updates.
    """
    
    @pytest.mark.requires_model
    @pytest.mark.golden
    def test_parameter_count_stable(self, weight_extractor, golden_comparator, 
                                     update_golden):
        """Verify parameter count hasn't changed."""
        counts = weight_extractor.count_parameters()
        
        golden_name = "param_counts"
        
        if update_golden:
            golden_comparator.save_golden(golden_name, counts)
            return
        
        golden = golden_comparator.load_golden(golden_name)
        
        if golden is None:
            pytest.skip("Golden file not found. Run with --update-golden")
        
        # Compare total param count
        assert counts['total'] == golden.item().get('total', 0), \
            f"Parameter count changed: {counts['total']} vs {golden.item()['total']}"
    
    @pytest.mark.requires_model
    @pytest.mark.golden
    def test_layer_structure_stable(self, weight_extractor, golden_comparator,
                                     update_golden):
        """Verify layer structure hasn't changed."""
        layers = weight_extractor.get_layer_info()
        
        golden_name = "layer_structure"
        
        if update_golden:
            golden_comparator.save_golden(golden_name, {
                'layer_count': len(layers),
                'layer_names': [l['name'] for l in layers]
            })
            return
        
        golden = golden_comparator.load_golden(golden_name)
        
        if golden is None:
            pytest.skip("Golden file not found. Run with --update-golden")
        
        golden_dict = golden.item() if hasattr(golden, 'item') else golden
        
        assert len(layers) == golden_dict.get('layer_count', 0), \
            f"Layer count changed: {len(layers)} vs {golden_dict.get('layer_count')}"
