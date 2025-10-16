// Custom error types for better error handling

use std::fmt;

/// Main error type for the rust-ml library
#[derive(Debug)]
pub enum MLError {
    /// Tensor operation errors
    TensorError(String),
    
    /// Model architecture errors
    ModelError(String),
    
    /// Training errors
    TrainingError(String),
    
    /// Data loading errors
    DataError(String),
    
    /// Configuration errors
    ConfigError(String),
    
    /// IO errors
    IoError(std::io::Error),
    
    /// Serialization errors
    SerdeError(String),
    
    /// PyTorch/tch errors
    TchError(tch::TchError),
}

impl fmt::Display for MLError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MLError::TensorError(msg) => write!(f, "Tensor error: {}", msg),
            MLError::ModelError(msg) => write!(f, "Model error: {}", msg),
            MLError::TrainingError(msg) => write!(f, "Training error: {}", msg),
            MLError::DataError(msg) => write!(f, "Data error: {}", msg),
            MLError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            MLError::IoError(err) => write!(f, "IO error: {}", err),
            MLError::SerdeError(msg) => write!(f, "Serialization error: {}", msg),
            MLError::TchError(err) => write!(f, "PyTorch error: {}", err),
        }
    }
}

impl std::error::Error for MLError {}

impl From<std::io::Error> for MLError {
    fn from(err: std::io::Error) -> Self {
        MLError::IoError(err)
    }
}

impl From<tch::TchError> for MLError {
    fn from(err: tch::TchError) -> Self {
        MLError::TchError(err)
    }
}

impl From<serde_json::Error> for MLError {
    fn from(err: serde_json::Error) -> Self {
        MLError::SerdeError(err.to_string())
    }
}

impl From<config::ConfigError> for MLError {
    fn from(err: config::ConfigError) -> Self {
        MLError::ConfigError(err.to_string())
    }
}

/// Result type alias for convenience
pub type MLResult<T> = Result<T, MLError>;

/// Training result with metrics
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub final_loss: f64,
    pub best_accuracy: f64,
    pub total_epochs: usize,
    pub training_time_ms: f64,
}

impl TrainingResult {
    pub fn new(final_loss: f64, best_accuracy: f64, total_epochs: usize, training_time_ms: f64) -> Self {
        Self {
            final_loss,
            best_accuracy,
            total_epochs,
            training_time_ms,
        }
    }
}
