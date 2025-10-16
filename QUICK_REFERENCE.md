# 🚀 Rust ML - Quick Reference Card

## ⚡ Hızlı Başlangıç

```bash
# Setup
git clone <repo>
cd rust-ml
source cuda_env.sh

# Build & Run
cargo run --release

# Test
cargo test

# Benchmark
cargo bench
```

---

## 📋 CLI Komutları

```bash
# Training
rust-ml train --model mlp --epochs 50 --batch-size 128

# Demo
rust-ml demo --technique transfer-learning

# Evaluation
rust-ml eval --checkpoint checkpoints/best_model.pt

# Config
rust-ml config --show

# Benchmark
rust-ml bench --iterations 1000

# System info
rust-ml info

# Help
rust-ml --help
```

---

## 🔧 Özellikler (Features)

### ✅ Dataset
```rust
let n_train = 100_000;  // 100K samples
let n_test = 10_000;    // 10K samples
```

### ✅ Batch Normalization
```rust
seq = seq
    .add(nn::linear(...))
    .add(BatchNormWrapper::new(...))
    .add_fn(|xs| xs.relu());
```

### ✅ Residual Connections
```rust
struct ResidualBlock {
    linear1: nn::Linear,
    bn1: nn::BatchNorm,
    shortcut: Option<nn::Linear>,
}
// out = F(x) + x
```

### ✅ Learning Rate Scheduler
```rust
use rust_ml::utils::{LRScheduler, CosineAnnealingLR};

let mut scheduler = CosineAnnealingLR::new(1e-3, 50, 1e-5);
let lr = scheduler.step(epoch);
optimizer.set_lr(lr);
```

### ✅ Early Stopping
```rust
use rust_ml::utils::{EarlyStopping, EarlyStoppingMode};

let mut early_stop = EarlyStopping::new(
    10,      // patience
    0.001,   // min_delta
    EarlyStoppingMode::Max  // for accuracy
);

if early_stop.step(test_acc) {
    println!("Early stopping at epoch {}", epoch);
    break;
}
```

---

## 📊 Model İstatistikleri

```
Parameters:    1,985,538 (~2M)
Model Size:    7.8 MB
Architecture:  20 → 768 → 1024 → 768 → 512 → 2
Layers:        10 (4 hidden + 4 batch norm + I/O)
```

---

## ⚙️ Hyperparameters

```rust
Batch Size:      128
Learning Rate:   1e-3 (initial) → 1e-5 (final)
Optimizer:       Adam
Scheduler:       CosineAnnealing
Early Stop:      Patience=10, MinDelta=0.001
Dropout:         0.3
Epochs:          50 (or until early stopping)
```

---

## 🏃 Performance

```
Training:       ~2.3-2.5 sec/epoch (100K samples)
Inference:      ~0.004 ms/sample
Throughput:     250,000 samples/sec
Memory:         ~50-60 MB GPU
CUDA:           ✅ Accelerated
cuDNN:          ✅ Enabled
```

---

## 📁 Dosya Yapısı

```
src/
├── main.rs              # Binary + training
├── lib.rs               # Library exports
├── cli.rs               # CLI (8 commands)
├── models.rs            # Neural networks
├── utils.rs             # LR scheduler + Early stop
├── error.rs             # Error types
├── config.rs            # Configuration
├── checkpoint.rs        # Model save/load
├── data.rs              # Data loading
└── [demos/showcases]    # Examples
```

---

## 🧪 Testing

```bash
# All tests
cargo test

# Specific test file
cargo test --test models_test

# With output
cargo test -- --nocapture

# Benchmarks
cargo bench
```

---

## 🐛 Debug

```bash
# Verbose build
RUST_LOG=debug cargo run

# Backtrace
RUST_BACKTRACE=1 cargo run

# Compilation warnings
cargo clippy

# Fix warnings
cargo fix --lib -p rust-ml
```

---

## 📦 Dependencies

### Core
- tch-rs 0.20.0 (PyTorch)
- CUDA 12.8 + cuDNN

### Production
- clap 4.5 (CLI)
- tracing 0.1 (Logging)
- config 0.14 (Config)
- serde 1.0 (Serialization)

### Dev
- criterion 0.5 (Benchmarks)
- proptest 1.4 (Testing)

---

## 🎯 Kullanım Örnekleri

### 1. Custom Training Loop
```rust
use rust_ml::utils::*;

let mut scheduler = CosineAnnealingLR::new(1e-3, 50, 1e-5);
let mut early_stop = EarlyStopping::new(10, 0.001, EarlyStoppingMode::Max);

for epoch in 1..=50 {
    // Train model
    train_epoch(&model, &data, &mut optimizer);
    
    // Evaluate
    let acc = evaluate(&model, &test_data);
    
    // Update LR
    optimizer.set_lr(scheduler.step(epoch));
    
    // Check early stopping
    if early_stop.step(acc) { break; }
}
```

### 2. Model Creation
```rust
use tch::nn;

let vs = nn::VarStore::new(device);
let model = create_advanced_net(
    &vs.root(), 
    20,                           // input_dim
    &[768, 1024, 768, 512],      // hidden_dims
    2                             // output_dim
);
```

### 3. Checkpointing
```rust
use rust_ml::checkpoint::CheckpointManager;

let ckpt = CheckpointManager::new("checkpoints");
ckpt.save(&vs, epoch, accuracy, "my_model")?;

// Load
ckpt.load(&mut vs, "checkpoints/best_model.pt")?;
```

---

## 🔬 Advanced Features

### RAG (Retrieval-Augmented Generation)
```rust
rust_ml::advanced_features::showcase_rag(device);
```

### Human-in-the-Loop
```rust
rust_ml::advanced_features::showcase_human_in_the_loop(device);
```

### Quantization (INT8/FP16)
```rust
rust_ml::advanced_features::showcase_quantization(device);
```

---

## 📈 Monitoring

### Training Progress
```
Epoch │  Train Loss │ Train Acc │  Test Acc │    LR    │  Time
───────┼─────────────┼───────────┼───────────┼──────────┼──────────
    1 │     0.6931  │   48.00%  │   47.50%  │ 0.00100 │ 2500 ms
    5 │     0.4523  │   78.50%  │   76.80%  │ 0.00095 │ 2400 ms
   ...
   41 │     0.0823  │   97.80%  │   96.20%  │ 0.00012 │ 2300 ms
   
⚠️  Early stopping triggered at epoch 41
    Best test accuracy: 96.50%
```

---

## 🚨 Troubleshooting

### CUDA Not Found
```bash
# Check CUDA
source cuda_env.sh
echo $LIBTORCH
```

### Compilation Errors
```bash
# Clean build
cargo clean
cargo build --release
```

### Out of Memory
```bash
# Reduce batch size
let batch_size = 64;  // Instead of 128
```

### Slow Training
```bash
# Check CUDA
nvidia-smi

# Verify device
let device = Device::Cuda(0);  // Not Device::Cpu
```

---

## 📚 Documentation

- `README.md` - Overview
- `FINAL_SUMMARY.md` - Complete summary
- `ADVANCED_FEATURES.md` - Advanced features guide
- `PARAMETER_INCREASE.md` - Model scaling details
- `DEVELOPMENT_SUMMARY.md` - Development process
- `CONTRIBUTING.md` - How to contribute

---

## 🎓 Learning Resources

### Rust
- [The Rust Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)

### tch-rs
- [tch-rs GitHub](https://github.com/LaurentMazare/tch-rs)
- [tch-rs Examples](https://github.com/LaurentMazare/tch-rs/tree/master/examples)

### Machine Learning
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

---

## 🏆 Best Practices

### 1. Always Use CUDA
```rust
let device = if tch::Cuda::is_available() {
    Device::Cuda(0)
} else {
    Device::Cpu
};
```

### 2. Error Handling
```rust
use rust_ml::MLResult;

fn train() -> MLResult<()> {
    // Your code
    Ok(())
}
```

### 3. Logging
```rust
use tracing::info;

info!("Starting training with {} samples", n_train);
```

### 4. Configuration
```rust
use rust_ml::config::Config;

let config = Config::from_file("config.toml")?;
```

---

## 🎯 Quick Tips

✅ **DO**
- Use `cargo build --release` for production
- Enable CUDA with `source cuda_env.sh`
- Use early stopping to prevent overfitting
- Save checkpoints regularly
- Monitor training progress

❌ **DON'T**
- Train on CPU (too slow)
- Use debug builds for benchmarking
- Ignore warnings (run `cargo clippy`)
- Skip testing (run `cargo test`)
- Hardcode configurations

---

## 📞 Support

Issues: GitHub Issues  
Docs: `cargo doc --open`  
Tests: `cargo test`  
Bench: `cargo bench`

---

**Version**: 2.0.0  
**Status**: ✅ Production Ready  
**Last Updated**: 16 Ekim 2025
