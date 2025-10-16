// Performance benchmarks for model operations

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rust_ml::models::*;
use rust_ml::advanced::*;
use tch::{nn, Device, Kind, Tensor};

fn bench_residual_block(c: &mut Criterion) {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let block = ResidualBlock::new(&vs.root(), 64, 64, 1);
    
    let input = Tensor::randn(&[8, 64, 28, 28], (Kind::Float, device));
    
    c.bench_function("residual_block_forward", |b| {
        b.iter(|| {
            let _ = black_box(block.forward(&input));
        });
    });
}

fn bench_attention(c: &mut Criterion) {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let d_model = 512i64;
    let n_heads = 8i64;
    let attention = MultiHeadAttention::new(&vs.root(), d_model, n_heads);
    
    let input = Tensor::randn(&[4, 20, d_model], (Kind::Float, device));
    
    c.bench_function("multihead_attention", |b| {
        b.iter(|| {
            let _ = black_box(attention.forward(&input));
        });
    });
}

fn bench_autoencoder(c: &mut Criterion) {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let autoencoder = Autoencoder::new(&vs.root(), 784, 32);
    
    let input = Tensor::randn(&[32, 784], (Kind::Float, device));
    
    c.bench_function("autoencoder_forward", |b| {
        b.iter(|| {
            let _ = black_box(autoencoder.forward(&input));
        });
    });
}

fn bench_gan_generator(c: &mut Criterion) {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let generator = create_generator(&vs.root(), 100, 784);
    
    let noise = Tensor::randn(&[64, 100], (Kind::Float, device));
    
    c.bench_function("gan_generator", |b| {
        b.iter(|| {
            let _ = black_box(generator.forward(&noise));
        });
    });
}

fn bench_mnist_classifier(c: &mut Criterion) {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let classifier = MNISTClassifier::new(&vs.root());
    
    let mut group = c.benchmark_group("mnist_classifier");
    
    for batch_size in [1, 8, 32, 128].iter() {
        let input = Tensor::randn(&[*batch_size, 784], (Kind::Float, device));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let _ = black_box(classifier.forward_t(&input, false));
                });
            },
        );
    }
    group.finish();
}

fn bench_vae(c: &mut Criterion) {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let vae = VAE::new(&vs.root(), 784, 512, 64);
    
    let input = Tensor::randn(&[32, 784], (Kind::Float, device));
    
    c.bench_function("vae_forward", |b| {
        b.iter(|| {
            let _ = black_box(vae.forward(&input));
        });
    });
}

fn bench_vector_retrieval(c: &mut Criterion) {
    let mut db = VectorDatabase::new(128);
    
    // Add documents
    for i in 0..1000 {
        let embedding = Tensor::randn(&[128], (Kind::Float, Device::Cpu));
        db.add_document(format!("Document {}", i), embedding);
    }
    
    let query = Tensor::randn(&[128], (Kind::Float, Device::Cpu));
    
    let mut group = c.benchmark_group("vector_retrieval");
    
    for k in [1, 5, 10, 50].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(k),
            k,
            |b, &k| {
                b.iter(|| {
                    let _ = black_box(db.retrieve(&query, k));
                });
            },
        );
    }
    group.finish();
}

fn bench_tensor_operations(c: &mut Criterion) {
    let device = Device::Cpu;
    
    let mut group = c.benchmark_group("tensor_ops");
    
    for size in [100, 1000, 10000].iter() {
        let a = Tensor::randn(&[*size, *size], (Kind::Float, device));
        let b = Tensor::randn(&[*size, *size], (Kind::Float, device));
        
        group.bench_with_input(
            BenchmarkId::new("matmul", size),
            size,
            |bench, _| {
                bench.iter(|| {
                    let _ = black_box(a.matmul(&b));
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_residual_block,
    bench_attention,
    bench_autoencoder,
    bench_gan_generator,
    bench_mnist_classifier,
    bench_vae,
    bench_vector_retrieval,
    bench_tensor_operations
);

criterion_main!(benches);
