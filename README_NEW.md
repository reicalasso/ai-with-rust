# 🦀 Rust ML v2.0 - Production Machine Learning

<div align="center">

[![CI/CD](https://github.com/reicalasso/ai-with-rust/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/reicalasso/ai-with-rust/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.90%2B-orange.svg)](https://www.rust-lang.org/)

*Proving Rust can compete with Python for AI/ML workloads*

[Features](#-features) •
[Installation](#-installation) •
[Usage](#-usage) •
[Examples](#-examples) •
[Documentation](#-documentation)

</div>

## 🎯 Features

### 🤖 Deep Learning Models
- **Neural Networks**: Multi-layer perceptrons with dropout and batch normalization
- **CNNs**: Convolutional networks for image classification (ResNet-style blocks)
- **LSTMs**: Recurrent networks for sequence modeling and NLP
- **Transformers**: Multi-head attention mechanisms
- **Autoencoders**: Dimensionality reduction and anomaly detection
- **VAEs**: Variational autoencoders for generative modeling
- **GANs**: Generative Adversarial Networks

### 🚀 Advanced Techniques
- **Transfer Learning**: Pretrained model fine-tuning
- **Ensemble Learning**: Parallel model inference with Rayon
- **Online Learning**: Streaming data adaptation
- **Reinforcement Learning**: Q-learning with experience replay
- **Time Series**: Forecasting with sliding windows
- **RAG**: Retrieval-Augmented Generation with vector database
- **Human-in-the-Loop**: Active learning with confidence thresholds
- **Model Quantization**: INT8/FP16 for faster inference

### 🛠️ Production Features
- ✅ **Comprehensive Testing**: Unit tests, integration tests, benchmarks
- ✅ **CI/CD Pipeline**: GitHub Actions with multi-platform support
- ✅ **Error Handling**: Custom error types with proper propagation
- ✅ **Logging**: Structured logging with tracing
- ✅ **Configuration**: TOML-based config management
- ✅ **Checkpointing**: Model save/load with metadata
- ✅ **Data Loading**: Efficient batching and shuffling
- ✅ **CLI Interface**: Command-line tool with clap
- ✅ **Documentation**: Comprehensive docs and examples

## 📦 Installation

### Prerequisites

- **Rust**: 1.90 or higher
- **Python**: 3.11+ (for PyTorch CUDA libraries)
- **CUDA**: 12.8+ (optional, for GPU support)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/reicalasso/ai-with-rust.git
cd rust-ml

# Install PyTorch (CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Automatic setup
./setup.sh

# Build and run
cargo build --release
./run_cuda.sh
```

### Manual Setup

```bash
# 1. Find PyTorch path
python3 -c "import torch; print(torch.__path__[0])"

# 2. Create environment file
cp cuda_env.sh.example cuda_env.sh
nano cuda_env.sh  # Update TORCH_PATH

# 3. Build
source cuda_env.sh
cargo build --release
```

## 🚀 Usage

### Command Line Interface

```bash
# Show all available commands
rust-ml --help

# Train a model
rust-ml train --model mlp --epochs 50

# Run demos
rust-ml demo --name cv                # Computer Vision
rust-ml demo --name nlp               # NLP Sentiment Analysis
rust-ml demo --name rag               # Retrieval-Augmented Generation
rust-ml demo --name quantization      # Model Quantization

# Generate configuration
rust-ml config --output my_config.toml

# System information
rust-ml info

# Run benchmarks
rust-ml bench --name all
```

### Library Usage

```rust
use rust_ml::*;
use tch::{nn, Device};

fn main() -> MLResult<()> {
    // Setup
    let device = Device::Cuda(0);
    let vs = nn::VarStore::new(device);
    
    // Create model
    let model = MNISTClassifier::new(&vs.root());
    
    // Load data
    let mut loader = DataLoader::new(train_data, train_labels, 128, true);
    
    // Training loop
    for epoch in 1..=50 {
        while let Some((batch_x, batch_y)) = loader.next_batch() {
            // Train...
        }
    }
    
    // Save checkpoint
    let manager = CheckpointManager::new("checkpoints")?;
    manager.save_checkpoint(&vs, &metadata)?;
    
    Ok(())
}
```

## 📚 Examples

### Image Classification

```bash
cargo run --example image_classification
```

### Transfer Learning

```bash
cargo run --example transfer_learning
```

### Time Series Forecasting

```bash
cargo run --example time_series
```

### Model Checkpointing

```bash
cargo run --example checkpointing
```

See the [examples/](examples/) directory for more examples.

## 🏗️ Project Structure

```
rust-ml/
├── src/
│   ├── lib.rs                    # Library exports
│   ├── main.rs                   # Main entry point
│   ├── cli.rs                    # Command-line interface
│   ├── models.rs                 # Model architectures
│   ├── advanced.rs               # Advanced models (CNN, LSTM, VAE)
│   ├── advanced_features.rs      # RAG, HITL, Quantization
│   ├── demos.rs                  # Demo implementations
│   ├── showcases.rs              # Production showcases
│   ├── config.rs                 # Configuration management
│   ├── checkpoint.rs             # Model serialization
│   ├── error.rs                  # Error handling
│   ├── logging.rs                # Logging utilities
│   ├── utils.rs                  # Helper functions
│   └── data.rs                   # Data loading
├── tests/                        # Integration tests
├── benches/                      # Performance benchmarks
├── examples/                     # Example programs
├── config/                       # Configuration files
├── .github/workflows/            # CI/CD pipelines
└── Cargo.toml                    # Dependencies
```

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run specific test suite
cargo test --test models_test

# Run benchmarks
cargo bench

# Generate coverage report
cargo tarpaulin --out Html
```

## 📊 Benchmarks

Run comprehensive benchmarks:

```bash
cargo bench
```

Results are saved to `target/criterion/report/index.html`.

## 🔧 Configuration

Configuration files use TOML format:

```toml
[model]
name = "mlp"
input_dim = 784
hidden_dims = [256, 128, 64]
output_dim = 10
dropout = 0.3

[training]
epochs = 50
batch_size = 128
learning_rate = 0.001
optimizer = "adam"

[data]
train_samples = 10000
test_samples = 2000
validation_split = 0.2
shuffle = true

[logging]
level = "info"
log_interval = 5
save_metrics = true
```

Generate default config:

```bash
rust-ml config --output config.toml
```

## 🎓 Documentation

### API Documentation

Generate and open API docs:

```bash
cargo doc --open
```

### Guides

- [Setup Guide](SETUP_GUIDE.md) - Detailed installation instructions
- [Architecture Guide](docs/ARCHITECTURE.md) - System design
- [Contributing Guide](CONTRIBUTING.md) - How to contribute

## 🚀 Performance

### Inference Speed

| Model | Batch Size | GPU (ms) | CPU (ms) |
|-------|-----------|----------|----------|
| MLP | 128 | 2.1 | 15.3 |
| CNN | 128 | 8.4 | 124.7 |
| LSTM | 64 | 3.4 | 42.1 |
| VAE | 32 | 5.2 | 67.8 |

### Training Throughput

| Model | Images/sec (GPU) | Images/sec (CPU) |
|-------|------------------|------------------|
| CNN | 79,000+ | 8,000+ |
| LSTM | 15,000+ | 1,500+ |

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- Built with [tch-rs](https://github.com/LaurentMazare/tch-rs) - Rust bindings for PyTorch
- Inspired by modern ML frameworks and production best practices
- Community contributions and feedback

## 📧 Contact

- **Author**: Rei Calasso
- **Repository**: [github.com/reicalasso/ai-with-rust](https://github.com/reicalasso/ai-with-rust)
- **Issues**: [github.com/reicalasso/ai-with-rust/issues](https://github.com/reicalasso/ai-with-rust/issues)

---

<div align="center">

**⭐ Star this repository if you find it useful!**

Made with ❤️ and 🦀

</div>
