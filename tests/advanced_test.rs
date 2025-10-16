// Integration tests for advanced features

use rust_ml::advanced::*;
use tch::{nn, Device, Kind, Tensor};

#[test]
fn test_mnist_classifier() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let classifier = MNISTClassifier::new(&vs.root());
    
    let images = Tensor::randn(&[4, 784], (Kind::Float, device));
    let output = classifier.forward_t(&images, false);
    
    assert_eq!(output.size(), vec![4, 10]);
}

#[test]
fn test_mnist_classifier_training_mode() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let classifier = MNISTClassifier::new(&vs.root());
    
    let images = Tensor::randn(&[8, 784], (Kind::Float, device));
    
    // Training mode
    let output_train = classifier.forward_t(&images, true);
    assert_eq!(output_train.size(), vec![8, 10]);
    
    // Inference mode
    let output_eval = classifier.forward_t(&images, false);
    assert_eq!(output_eval.size(), vec![8, 10]);
}

#[test]
fn test_sentiment_analyzer() {
    let device = Device::Cpu;
    let vocab_size = 1000i64;
    let embed_dim = 64i64;
    let hidden_dim = 128i64;
    
    let vs = nn::VarStore::new(device);
    let analyzer = SentimentAnalyzer::new(&vs.root(), vocab_size, embed_dim, hidden_dim);
    
    let sequences = Tensor::randint(vocab_size, &[4, 20], (Kind::Int64, device));
    let output = analyzer.forward(&sequences);
    
    assert_eq!(output.size(), vec![4, 2]); // Binary sentiment
}

#[test]
fn test_vae_forward() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let input_dim = 784i64;
    let hidden_dim = 256i64;
    let latent_dim = 32i64;
    
    let vae = VAE::new(&vs.root(), input_dim, hidden_dim, latent_dim);
    
    let input = Tensor::randn(&[8, input_dim], (Kind::Float, device));
    let (reconstructed, mu, logvar) = vae.forward(&input);
    
    assert_eq!(reconstructed.size(), vec![8, input_dim]);
    assert_eq!(mu.size(), vec![8, latent_dim]);
    assert_eq!(logvar.size(), vec![8, latent_dim]);
}

#[test]
fn test_vae_reparameterization() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let vae = VAE::new(&vs.root(), 100, 128, 16);
    
    let mu = Tensor::zeros(&[4, 16], (Kind::Float, device));
    let logvar = Tensor::zeros(&[4, 16], (Kind::Float, device));
    
    let z = vae.reparameterize(&mu, &logvar);
    
    assert_eq!(z.size(), vec![4, 16]);
}

#[test]
fn test_vae_loss_computation() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let vae = VAE::new(&vs.root(), 100, 128, 16);
    
    let input = Tensor::randn(&[8, 100], (Kind::Float, device));
    let (reconstructed, mu, logvar) = vae.forward(&input);
    
    let loss = vae.loss(&input, &reconstructed, &mu, &logvar);
    
    // Loss should be a scalar
    assert_eq!(loss.size(), vec![]);
    assert!(!loss.double_value(&[]).is_nan());
}

#[test]
fn test_ensemble_model() {
    let device = Device::Cpu;
    let vs1 = nn::VarStore::new(device);
    let vs2 = nn::VarStore::new(device);
    let vs3 = nn::VarStore::new(device);
    
    let model1 = MNISTClassifier::new(&vs1.root());
    let model2 = MNISTClassifier::new(&vs2.root());
    let model3 = MNISTClassifier::new(&vs3.root());
    
    let ensemble = EnsembleModel::new(vec![
        Box::new(model1),
        Box::new(model2),
        Box::new(model3),
    ]);
    
    let input = Tensor::randn(&[4, 784], (Kind::Float, device));
    let output = ensemble.predict(&input);
    
    assert_eq!(output.size(), vec![4, 10]);
}

#[test]
fn test_online_learning() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    
    let mut learner = OnlineLearner::new(&vs, device, 100);
    
    // Add some experiences
    for i in 0..50 {
        let x = Tensor::randn(&[1, 20], (Kind::Float, device));
        let y = Tensor::randint(2, &[1], (Kind::Int64, device));
        learner.add_experience(x, y);
    }
    
    assert_eq!(learner.buffer_size(), 50);
    
    // Update model
    let loss = learner.update(5);
    assert!(!loss.is_nan());
}

#[test]
fn test_benchmark_struct() {
    let mut benchmark = Benchmark::new("Test");
    
    let time1 = benchmark.measure(|| {
        std::thread::sleep(std::time::Duration::from_millis(10));
    });
    
    let time2 = benchmark.measure(|| {
        std::thread::sleep(std::time::Duration::from_millis(20));
    });
    
    assert!(time1 >= 10.0);
    assert!(time2 >= 20.0);
    assert_eq!(benchmark.measurements.len(), 2);
}
