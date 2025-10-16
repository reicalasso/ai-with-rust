// Example: Transfer learning for custom task

use rust_ml::*;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

fn main() -> MLResult<()> {
    println!("🔄 Transfer Learning Example\n");
    
    let device = Device::Cpu;
    
    // Pretrained feature extractor (frozen)
    let vs_pretrained = nn::VarStore::new(device);
    let feature_extractor = nn::seq()
        .add(nn::linear(&vs_pretrained.root() / "fc1", 100, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs_pretrained.root() / "fc2", 64, 32, Default::default()))
        .add_fn(|xs| xs.relu());
    
    // New task head (trainable)
    let vs_new = nn::VarStore::new(device);
    let task_head = nn::seq()
        .add(nn::linear(&vs_new.root() / "head1", 32, 16, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs_new.root() / "head2", 16, 3, Default::default()));
    
    let mut opt = nn::Adam::default().build(&vs_new, 1e-3)?;
    
    // Data
    let x = Tensor::randn(&[200, 100], (Kind::Float, device));
    let y = Tensor::randint(3, &[200], (Kind::Int64, device));
    
    println!("Training new task head...\n");
    
    for epoch in 1..=20 {
        // Extract features (no gradient)
        let features = tch::no_grad(|| feature_extractor.forward(&x));
        
        // Train task head
        let logits = task_head.forward(&features);
        let loss = logits.cross_entropy_for_logits(&y);
        
        opt.zero_grad();
        loss.backward();
        opt.step();
        
        if epoch % 5 == 0 {
            println!("Epoch {}: Loss = {:.4}", epoch, loss.double_value(&[]));
        }
    }
    
    println!("\n✅ Transfer learning completed!");
    
    Ok(())
}
