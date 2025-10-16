// Example: Time series forecasting

use rust_ml::*;
use rust_ml::data::TimeSeriesProcessor;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

fn main() -> MLResult<()> {
    println!("📈 Time Series Forecasting Example\n");
    
    let device = Device::Cpu;
    
    // Generate synthetic time series
    let t = Tensor::arange(1000, (Kind::Float, device)) * 0.1;
    let series = (&t * 2.0).sin() + Tensor::randn(&[1000], (Kind::Float, device)) * 0.1;
    
    // Create sliding windows
    let processor = TimeSeriesProcessor::new(20, 1);
    let (x, y) = processor.create_windows(&series);
    
    println!("Created {} windows", x.size()[0]);
    
    // Simple forecasting model
    let vs = nn::VarStore::new(device);
    let model = nn::seq()
        .add(nn::linear(&vs.root() / "fc1", 20, 64, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs.root() / "fc2", 64, 32, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&vs.root() / "fc3", 32, 1, Default::default()));
    
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    
    println!("Training forecasting model...\n");
    
    for epoch in 1..=50 {
        let predictions = model.forward(&x);
        let loss = predictions.mse_loss(&y, tch::Reduction::Mean);
        
        opt.zero_grad();
        loss.backward();
        opt.step();
        
        if epoch % 10 == 0 {
            println!("Epoch {}: MSE = {:.6}", epoch, loss.double_value(&[]));
        }
    }
    
    println!("\n✅ Forecasting model trained!");
    
    // Make predictions
    let test_window = x.get(0);
    let prediction = tch::no_grad(|| model.forward(&test_window.unsqueeze(0)));
    let actual = y.get(0);
    
    println!("\nSample prediction:");
    println!("  Predicted: {:.4}", prediction.double_value(&[0, 0]));
    println!("  Actual:    {:.4}", actual.double_value(&[]));
    
    Ok(())
}
