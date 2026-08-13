# Contributing to SiLens

Thank you for your interest in contributing to SiLens! This document provides guidelines for contributing to the project.

## Code of Conduct

Be respectful, inclusive, and constructive. We're building open-source AI hardware together.

## How to Contribute

### Reporting Issues

- Check existing issues before creating a new one
- Use issue templates when available
- Include relevant details (OS, Python version, error messages)
- For bugs, include steps to reproduce

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`make test`)
5. Commit with clear messages
6. Push and create a Pull Request

### Commit Messages

```
type(scope): brief description

Longer description if needed.

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `rtl`: RTL changes
- `test`: Test changes
- `refactor`: Code refactoring

### Code Style

**Python:**
- Use `black` for formatting
- Use `flake8` for linting
- Type hints where helpful
- Docstrings for public functions

**Verilog:**
- Follow guidelines in `rtl/README.md`
- Use consistent naming conventions
- Include comments for complex logic

## Areas Needing Help

### High Priority

1. **RTL Design**
   - Transformer block implementation
   - Attention mechanism
   - Activation functions (GELU, Softmax)
   
2. **Model Conversion**
   - Weight quantization algorithms
   - PyTorch → Verilog pipeline
   - Accuracy validation

3. **Verification**
   - Cocotb testbenches
   - Golden model comparison
   - Coverage analysis

### Medium Priority

4. **FPGA Prototyping**
   - Xilinx/Intel port
   - Resource optimization
   - Demo applications

5. **PCB Design**
   - Schematic review
   - Signal integrity
   - Power delivery

6. **Documentation**
   - Architecture docs
   - Tutorial content
   - API documentation

### Future

7. **Driver Development**
   - Linux kernel driver
   - User-space library
   - Python bindings

8. **SDK**
   - High-level API
   - Model serving
   - Benchmarking tools

## Development Setup

```bash
# Clone
git clone https://github.com/[your-fork]/SiLens.git
cd SiLens

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
make test

# Format code
make format
```

## Questions?

- Open a GitHub Discussion
- Join our Discord (coming soon)
- Email: contribute@silens.ai

## License

By contributing, you agree that your contributions will be licensed under Apache 2.0.
