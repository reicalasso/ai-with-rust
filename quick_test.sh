#!/bin/bash
# Quick test with smaller dataset to verify all features work

cd /home/rei/Desktop/rust-ml

echo "🧪 Quick Feature Test (10K samples)"
echo "===================================="

# Temporarily modify for quick test
sed -i 's/let n_train = 100_000i64/let n_train = 10_000i64/' src/main.rs
sed -i 's/let n_test = 10_000i64/let n_test = 2_000i64/' src/main.rs

echo "Building..."
cargo build --release 2>&1 | grep -E "(Compiling|Finished)"

echo ""
echo "Running quick test..."
source cuda_env.sh
timeout 60 cargo run --release 2>&1 | grep -E "(Architecture|Learning rate|Training samples|Epoch|Early stopping|Best test|Total training|parameters|Model size)" || true

# Restore original values
sed -i 's/let n_train = 10_000i64/let n_train = 100_000i64/' src/main.rs
sed -i 's/let n_test = 2_000i64/let n_test = 10_000i64/' src/main.rs

echo ""
echo "✅ Test completed! Original values restored."
