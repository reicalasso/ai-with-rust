// Example: Model checkpointing and resuming

use rust_ml::*;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

fn main() -> MLResult<()> {
    println!("💾 Checkpointing Example\n");
    
    let device = Device::Cpu;
    let checkpoint_manager = CheckpointManager::new("checkpoints")?;
    
    // Create model
    let mut vs = nn::VarStore::new(device);
    let model = nn::seq()
        .add(nn::linear(&vs.root() / "fc1", 10, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs.root() / "fc2", 64, 2, Default::default()));
    
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    
    // Training data
    let x = Tensor::randn(&[100, 10], (Kind::Float, device));
    let y = Tensor::randint(2, &[100], (Kind::Int64, device));
    
    println!("Training with checkpointing...\n");
    
    for epoch in 1..=20 {
        let logits = model.forward(&x);
        let loss = logits.cross_entropy_for_logits(&y);
        
        opt.zero_grad();
        loss.backward();
        opt.step();
        
        // Save checkpoint every 5 epochs
        if epoch % 5 == 0 {
            let metadata = CheckpointMetadata::new(
                "example_model".to_string(),
                "1.0.0".to_string(),
                epoch,
                loss.double_value(&[]),
                0.0,
                serde_json::json!({"lr": 0.001}),
            );
            
            let name = checkpoint_manager.save_checkpoint(&vs, &metadata)?;
            println!("Epoch {}: Loss = {:.4} | Checkpoint: {}", epoch, loss.double_value(&[]), name);
        }
    }
    
    println!("\n📋 Available checkpoints:");
    let checkpoints = checkpoint_manager.list_checkpoints()?;
    for (i, checkpoint) in checkpoints.iter().enumerate() {
        println!("  {}. {}", i + 1, checkpoint);
    }
    
    // Load latest checkpoint
    if let Some(latest) = checkpoint_manager.get_latest_checkpoint()? {
        println!("\n🔄 Loading latest checkpoint: {}", latest);
        let metadata = checkpoint_manager.load_checkpoint(&mut vs, &latest)?;
        println!("  Model: {}", metadata.model_name);
        println!("  Epoch: {}", metadata.epoch);
        println!("  Loss: {:.4}", metadata.train_loss);
    }
    
    // Cleanup old checkpoints
    checkpoint_manager.cleanup_old_checkpoints(2)?;
    println!("\n🗑️  Cleaned up old checkpoints, kept last 2");
    
    Ok(())
}
