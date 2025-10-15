// Demo uygulamaları - farklı ML tekniklerini gösterir

use tch::{nn::{self, Module, OptimizerConfig}, Device, Kind, Tensor, IndexOp};
use std::time::Instant;

/// Transfer Learning Demo - Önceden eğitilmiş model kullanma simülasyonu
pub fn demo_transfer_learning(device: Device) -> tch::Result<()> {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║              🔄 Transfer Learning Demo                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    
    // "Pretrained" model (simülasyon)
    let vs_pretrained = nn::VarStore::new(device);
    let feature_extractor = nn::seq()
        .add(nn::linear(&vs_pretrained.root() / "conv1", 784, 256, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs_pretrained.root() / "conv2", 256, 128, Default::default()))
        .add_fn(|xs| xs.relu());
    
    // Yeni task için classifier
    let vs_new = nn::VarStore::new(device);
    let classifier = nn::seq()
        .add(nn::linear(&vs_new.root() / "fc1", 128, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs_new.root() / "fc2", 64, 5, Default::default()));
    
    let mut opt = nn::Adam::default().build(&vs_new, 1e-3)?;
    
    println!("  ✓ Pretrained feature extractor: 784 → 128");
    println!("  ✓ New classifier: 128 → 5");
    println!("  ✓ Fine-tuning only the classifier...\n");
    
    // Eğitim
    let x = Tensor::randn(&[100, 784], (Kind::Float, device));
    let y = Tensor::randint(5, &[100], (Kind::Int64, device));
    
    for epoch in 1..=10 {
        let features = tch::no_grad(|| feature_extractor.forward(&x));
        let logits = classifier.forward(&features);
        let loss = logits.cross_entropy_for_logits(&y);
        
        opt.zero_grad();
        loss.backward();
        opt.step();
        
        if epoch % 2 == 0 {
            println!("  Epoch {} | Loss: {:.4}", epoch, loss.double_value(&[]));
        }
    }
    
    println!("\n  ✅ Transfer learning completed!\n");
    Ok(())
}

/// Anomaly Detection Demo - Autoencoder kullanarak
pub fn demo_anomaly_detection(device: Device) -> tch::Result<()> {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║              🔍 Anomaly Detection Demo                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    
    use crate::models::Autoencoder;
    
    let vs = nn::VarStore::new(device);
    let autoencoder = Autoencoder::new(&vs.root(), 20, 5);
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    
    println!("  Model: Autoencoder (20 → 5 → 20)");
    println!("  Training on normal data...\n");
    
    // Normal veri
    let normal_data = Tensor::randn(&[500, 20], (Kind::Float, device)) * 0.5;
    
    // Eğitim
    for epoch in 1..=20 {
        let reconstructed = autoencoder.forward(&normal_data);
        let loss = (&reconstructed - &normal_data).pow_tensor_scalar(2).mean(Kind::Float);
        
        opt.zero_grad();
        loss.backward();
        opt.step();
        
        if epoch % 5 == 0 {
            println!("  Epoch {} | Reconstruction Loss: {:.4}", epoch, loss.double_value(&[]));
        }
    }
    
    // Test - anomali tespiti
    let normal_test = Tensor::randn(&[10, 20], (Kind::Float, device)) * 0.5;
    let anomaly_test = Tensor::randn(&[10, 20], (Kind::Float, device)) * 3.0; // Farklı dağılım
    
    let normal_recon = tch::no_grad(|| autoencoder.forward(&normal_test));
    let anomaly_recon = tch::no_grad(|| autoencoder.forward(&anomaly_test));
    
    let normal_error = (&normal_recon - &normal_test).pow_tensor_scalar(2).mean(Kind::Float);
    let anomaly_error = (&anomaly_recon - &anomaly_test).pow_tensor_scalar(2).mean(Kind::Float);
    
    println!("\n  Test Results:");
    println!("  Normal samples error:  {:.4}", normal_error.double_value(&[]));
    println!("  Anomaly samples error: {:.4}", anomaly_error.double_value(&[]));
    println!("  Anomaly detected: {}", if anomaly_error.double_value(&[]) > normal_error.double_value(&[]) * 2.0 { "✓" } else { "✗" });
    println!("\n  ✅ Anomaly detection completed!\n");
    
    Ok(())
}

/// Reinforcement Learning Demo - Basit Q-Learning simülasyonu
pub fn demo_reinforcement_learning(device: Device) -> tch::Result<()> {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║           🎮 Reinforcement Learning Demo (Q-Network)           ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    
    // Q-Network: state → Q-values for each action
    let vs = nn::VarStore::new(device);
    let q_network = nn::seq()
        .add(nn::linear(&vs.root() / "fc1", 4, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs.root() / "fc2", 64, 32, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs.root() / "fc3", 32, 2, Default::default()));
    
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    
    println!("  Environment: Simple 4-state environment");
    println!("  Actions: 2 (left/right)");
    println!("  Training with random experiences...\n");
    
    // Simulated experience replay
    let mut total_reward = 0.0;
    for episode in 1..=50 {
        let state = Tensor::randn(&[32, 4], (Kind::Float, device));
        let next_state = Tensor::randn(&[32, 4], (Kind::Float, device));
        let actions = Tensor::randint(2, &[32], (Kind::Int64, device));
        let rewards = Tensor::randn(&[32, 1], (Kind::Float, device));
        let done = Tensor::rand(&[32, 1], (Kind::Float, device)).gt(0.9);
        
        // Q-learning update
        let q_values = q_network.forward(&state);
        let next_q_values = tch::no_grad(|| q_network.forward(&next_state));
        let max_next_q = next_q_values.max_dim(1, false).0;
        
        let target_q = &rewards + &max_next_q.unsqueeze(1) * 0.99 * done.logical_not().to_kind(Kind::Float);
        
        // Gather Q-values for taken actions
        let q_pred = q_values.gather(1, &actions.unsqueeze(1), false);
        let loss = (q_pred - target_q).pow_tensor_scalar(2).mean(Kind::Float);
        
        opt.zero_grad();
        loss.backward();
        opt.step();
        
        total_reward += rewards.mean(Kind::Float).double_value(&[]);
        
        if episode % 10 == 0 {
            let avg_reward = total_reward / 10.0;
            println!("  Episode {} | Avg Reward: {:.4} | Loss: {:.4}", 
                     episode, avg_reward, loss.double_value(&[]));
            total_reward = 0.0;
        }
    }
    
    println!("\n  ✅ RL training completed!\n");
    Ok(())
}

/// Time Series Forecasting Demo - LSTM benzeri
pub fn demo_time_series(device: Device) -> tch::Result<()> {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║              📈 Time Series Forecasting Demo                   ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    
    let vs = nn::VarStore::new(device);
    
    // Basit RNN-like model
    let model = nn::seq()
        .add(nn::linear(&vs.root() / "fc1", 10, 64, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(&vs.root() / "fc2", 64, 32, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(&vs.root() / "fc3", 32, 1, Default::default()));
    
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    
    println!("  Model: Simple feedforward network");
    println!("  Task: Predict next value from last 10 values\n");
    
    // Sentetik time series (sine wave + noise)
    for epoch in 1..=30 {
        let mut total_loss = 0.0;
        
        for _ in 0..10 {
            let t = Tensor::randn(&[64, 10], (Kind::Float, device));
            let target = t.mean_dim(&[1i64][..], true, Kind::Float);
            
            let pred = model.forward(&t);
            let loss = (pred - target).pow_tensor_scalar(2).mean(Kind::Float);
            
            opt.zero_grad();
            loss.backward();
            opt.step();
            
            total_loss += loss.double_value(&[]);
        }
        
        if epoch % 10 == 0 {
            println!("  Epoch {} | MSE Loss: {:.6}", epoch, total_loss / 10.0);
        }
    }
    
    println!("\n  ✅ Time series forecasting completed!\n");
    Ok(())
}

/// Tüm demoları çalıştır
pub fn run_all_demos(device: Device) -> tch::Result<()> {
    let start = Instant::now();
    
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║        🚀 Rust ML Advanced Techniques Showcase                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    
    demo_transfer_learning(device)?;
    demo_anomaly_detection(device)?;
    demo_reinforcement_learning(device)?;
    demo_time_series(device)?;
    
    let elapsed = start.elapsed().as_secs_f64();
    
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                    All Demos Completed!                        ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!("  Total time: {:.2}s", elapsed);
    println!("  All techniques demonstrated successfully! 🎉\n");
    
    Ok(())
}
