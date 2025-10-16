// Example: Simple image classification with CNN

use rust_ml::*;
use rust_ml::advanced::MNISTClassifier;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

fn main() -> MLResult<()> {
    println!("🖼️  Image Classification Example\n");
    
    // Setup
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let model = MNISTClassifier::new(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    
    // Generate synthetic data
    let train_images = Tensor::randn(&[1000, 784], (Kind::Float, device));
    let train_labels = Tensor::randint(10, &[1000], (Kind::Int64, device));
    
    // Create data loader
    let mut loader = DataLoader::new(train_images, train_labels, 32, true);
    
    println!("Training for 10 epochs...\n");
    
    // Training loop
    for epoch in 1..=10 {
        let mut total_loss = 0.0;
        let mut batch_count = 0;
        
        loader.reset();
        while let Some((batch_images, batch_labels)) = loader.next_batch() {
            let logits = model.forward_t(&batch_images, true);
            let loss = logits.cross_entropy_for_logits(&batch_labels);
            
            opt.zero_grad();
            loss.backward();
            opt.step();
            
            total_loss += loss.double_value(&[]);
            batch_count += 1;
        }
        
        let avg_loss = total_loss / batch_count as f64;
        println!("Epoch {}: Loss = {:.4}", epoch, avg_loss);
    }
    
    println!("\n✅ Training completed!");
    
    // Save model
    let checkpoint_manager = CheckpointManager::new("checkpoints")?;
    let metadata = CheckpointMetadata::new(
        "mnist_classifier".to_string(),
        "1.0.0".to_string(),
        10,
        0.0,
        0.0,
        serde_json::json!({"lr": 0.001, "batch_size": 32}),
    );
    
    let checkpoint_name = checkpoint_manager.save_checkpoint(&vs, &metadata)?;
    println!("Model saved as: {}", checkpoint_name);
    
    Ok(())
}
