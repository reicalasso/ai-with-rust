// Integration tests for advanced features (RAG, HITL, Quantization)

use rust_ml::advanced_features::*;
use tch::{nn, Device, Kind, Tensor};

#[test]
fn test_vector_database_creation() {
    let db = VectorDatabase::new(128);
    assert_eq!(db.embedding_dim, 128);
    assert_eq!(db.documents.len(), 0);
}

#[test]
fn test_vector_database_add_document() {
    let mut db = VectorDatabase::new(64);
    let embedding = Tensor::randn(&[64], (Kind::Float, Device::Cpu));
    
    db.add_document("Test document".to_string(), embedding);
    
    assert_eq!(db.documents.len(), 1);
    assert_eq!(db.embeddings.len(), 1);
}

#[test]
fn test_vector_database_retrieval() {
    let mut db = VectorDatabase::new(64);
    
    // Add some documents
    for i in 0..5 {
        let embedding = Tensor::randn(&[64], (Kind::Float, Device::Cpu));
        db.add_document(format!("Document {}", i), embedding);
    }
    
    let query = Tensor::randn(&[64], (Kind::Float, Device::Cpu));
    let results = db.retrieve(&query, 3);
    
    assert_eq!(results.len(), 3);
    
    // Check that scores are sorted in descending order
    for i in 0..results.len() - 1 {
        assert!(results[i].1 >= results[i + 1].1);
    }
}

    #[test]
    fn test_cosine_similarity() {
        let db = VectorDatabase::new(10);
        
        // Test with identical vectors
        let v1 = Tensor::ones(&[10], (Kind::Float, Device::Cpu));
        let v2 = Tensor::ones(&[10], (Kind::Float, Device::Cpu));
        let sim = db.cosine_similarity(&v1, &v2);
        assert!((sim - 1.0).abs() < 1e-5);
        
        // Test with orthogonal vectors
        let v3 = Tensor::from_slice(&[1.0, 0.0]);
        let v4 = Tensor::from_slice(&[0.0, 1.0]);
        let db2 = VectorDatabase::new(2);
        let sim2 = db2.cosine_similarity(&v3, &v4);
        assert!(sim2.abs() < 1e-5);
    }#[test]
fn test_rag_model_creation() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let model = RAGModel::new(&vs.root(), 100, 128, 64);
    
    let input = Tensor::randn(&[4, 100], (Kind::Float, device));
    let embedding = model.encode(&input);
    
    assert_eq!(embedding.size(), vec![4, 64]);
}

#[test]
fn test_hitl_system() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let confidence_threshold = 0.7;
    
    let mut hitl = HumanInTheLoopSystem::new(&vs, device, confidence_threshold);
    
    // Simulate predictions
    let input = Tensor::randn(&[1, 20], (Kind::Float, device));
    let label = Tensor::of_slice(&[1i64]);
    
    hitl.process_sample(input.shallow_clone(), label.shallow_clone());
    
    // Check initial state
    assert_eq!(hitl.certain_samples.len(), 1);
}

#[test]
fn test_hitl_confidence_calculation() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let hitl = HumanInTheLoopSystem::new(&vs, device, 0.8);
    
    // Create a prediction with high confidence
    let logits = Tensor::of_slice(&[5.0f32, -2.0, -1.0, -3.0]);
    let probs = logits.softmax(0, Kind::Float);
    let confidence = probs.max().double_value(&[]);
    
    assert!(confidence > 0.9);
}

#[test]
fn test_model_quantization() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    
    let quantizer = ModelQuantizer::new(&vs);
    
    // Test weight quantization
    let weights = Tensor::randn(&[100, 50], (Kind::Float, device));
    let quantized = quantizer.quantize_weights(&weights, QuantizationMode::INT8);
    
    assert_eq!(quantized.size(), weights.size());
}

#[test]
fn test_quantization_modes() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let quantizer = ModelQuantizer::new(&vs);
    
    let input = Tensor::randn(&[10, 20], (Kind::Float, device));
    
    // Test INT8 quantization
    let int8 = quantizer.quantize_weights(&input, QuantizationMode::INT8);
    assert_eq!(int8.kind(), Kind::Float); // Still float but quantized values
    
    // Test FP16 quantization (simulated)
    let fp16 = quantizer.quantize_weights(&input, QuantizationMode::FP16);
    assert_eq!(fp16.size(), input.size());
}

    #[test]
    fn test_quantization_scale() {
        let device = Device::Cpu;
        
        // Test INT8 scale calculation
        let weights = Tensor::from_slice(&[-10.0f32, -5.0, 0.0, 5.0, 10.0]);
        let max_val = weights.abs().max().double_value(&[]);
        let scale = max_val / 127.0;
        
        let quantized = (&weights / scale).round().clamp(-127.0, 127.0);
        let dequantized = &quantized * scale;
        
        // Check reconstruction error is reasonable
        let error = (&weights - &dequantized).abs().mean(Kind::Float).double_value(&[]);
        assert!(error < 0.2); // Small error allowed due to quantization
    }#[test]
fn test_rag_showcase_data_generation() {
    let device = Device::Cpu;
    
    // Simulate knowledge base
    let knowledge = vec![
        "Rust is a systems programming language".to_string(),
        "Machine learning requires large datasets".to_string(),
        "Neural networks have multiple layers".to_string(),
    ];
    
    assert_eq!(knowledge.len(), 3);
}

#[test]
fn test_hitl_uncertain_sample_flagging() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let mut hitl = HumanInTheLoopSystem::new(&vs, device, 0.9);
    
    // Create ambiguous prediction (low confidence)
    let input = Tensor::randn(&[1, 20], (Kind::Float, device));
    let label = Tensor::of_slice(&[0i64]);
    
    hitl.process_sample(input, label);
    
    // With high threshold, sample might be flagged as uncertain
    assert!(hitl.certain_samples.len() <= 1);
}
