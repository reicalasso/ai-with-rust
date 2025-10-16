// Integration tests for model architectures

use rust_ml::models::*;
use tch::{nn, Device, Kind, Tensor};

#[test]
fn test_residual_block_forward() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let block = ResidualBlock::new(&vs.root(), 32, 32, 1);
    
    let input = Tensor::randn(&[4, 32, 28, 28], (Kind::Float, device));
    let output = block.forward(&input);
    
    assert_eq!(output.size(), vec![4, 32, 28, 28]);
}

#[test]
fn test_residual_block_with_stride() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let block = ResidualBlock::new(&vs.root(), 32, 64, 2);
    
    let input = Tensor::randn(&[4, 32, 28, 28], (Kind::Float, device));
    let output = block.forward(&input);
    
    // With stride 2, spatial dimensions should be halved
    assert_eq!(output.size(), vec![4, 64, 14, 14]);
}

#[test]
fn test_multihead_attention() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let d_model = 128i64;
    let n_heads = 8i64;
    let attention = MultiHeadAttention::new(&vs.root(), d_model, n_heads);
    
    let batch_size = 4;
    let seq_len = 10;
    let input = Tensor::randn(&[batch_size, seq_len, d_model], (Kind::Float, device));
    let output = attention.forward(&input);
    
    assert_eq!(output.size(), vec![batch_size, seq_len, d_model]);
}

#[test]
fn test_autoencoder_reconstruction() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let input_dim = 784i64;
    let latent_dim = 32i64;
    let autoencoder = Autoencoder::new(&vs.root(), input_dim, latent_dim);
    
    let input = Tensor::randn(&[8, input_dim], (Kind::Float, device));
    
    // Test encoding
    let latent = autoencoder.encode(&input);
    assert_eq!(latent.size(), vec![8, latent_dim]);
    
    // Test decoding
    let reconstructed = autoencoder.decode(&latent);
    assert_eq!(reconstructed.size(), vec![8, input_dim]);
    
    // Test full forward pass
    let output = autoencoder.forward(&input);
    assert_eq!(output.size(), input.size());
}

#[test]
fn test_gan_generator() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let latent_dim = 100i64;
    let output_dim = 784i64;
    let generator = create_generator(&vs.root(), latent_dim, output_dim);
    
    let noise = Tensor::randn(&[16, latent_dim], (Kind::Float, device));
    let generated = generator.forward(&noise);
    
    assert_eq!(generated.size(), vec![16, output_dim]);
    
    // Check if tanh is applied (values should be in [-1, 1])
    let max_val = generated.max().double_value(&[]);
    let min_val = generated.min().double_value(&[]);
    assert!(max_val <= 1.0 && min_val >= -1.0);
}

#[test]
fn test_gan_discriminator() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let input_dim = 784i64;
    let discriminator = create_discriminator(&vs.root(), input_dim);
    
    let input = Tensor::randn(&[16, input_dim], (Kind::Float, device));
    let output = discriminator.forward(&input);
    
    assert_eq!(output.size(), vec![16, 1]);
    
    // Check if sigmoid is applied (values should be in [0, 1])
    let max_val = output.max().double_value(&[]);
    let min_val = output.min().double_value(&[]);
    assert!(max_val <= 1.0 && min_val >= 0.0);
}

#[test]
fn test_scaled_dot_product_attention() {
    let device = Device::Cpu;
    let batch_size = 2;
    let seq_len = 5;
    let d_k = 64;
    
    let q = Tensor::randn(&[batch_size, seq_len, d_k], (Kind::Float, device));
    let k = Tensor::randn(&[batch_size, seq_len, d_k], (Kind::Float, device));
    let v = Tensor::randn(&[batch_size, seq_len, d_k], (Kind::Float, device));
    
    let output = scaled_dot_product_attention(&q, &k, &v);
    
    assert_eq!(output.size(), vec![batch_size, seq_len, d_k]);
}

#[test]
fn test_autoencoder_latent_space_properties() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let autoencoder = Autoencoder::new(&vs.root(), 100, 10);
    
    // Test that similar inputs produce similar latent representations
    let x1 = Tensor::ones(&[1, 100], (Kind::Float, device));
    let x2 = &x1 + Tensor::randn(&[1, 100], (Kind::Float, device)) * 0.1;
    
    let z1 = autoencoder.encode(&x1);
    let z2 = autoencoder.encode(&x2);
    
    // Compute cosine similarity
    let similarity = (&z1 * &z2).sum(Kind::Float) / 
                     (z1.norm() * z2.norm() + 1e-8);
    let sim_value = similarity.double_value(&[]);
    
    // Similar inputs should have high similarity
    assert!(sim_value > 0.5, "Similarity too low: {}", sim_value);
}
