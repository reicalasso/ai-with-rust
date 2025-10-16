// Configuration management for models and training

use serde::{Deserialize, Serialize};
use crate::error::{MLError, MLResult};
use std::path::Path;
use std::fs;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub data: DataConfig,
    pub logging: LoggingConfig,
}

impl Config {
    /// Load configuration from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> MLResult<Self> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)
            .map_err(|e| MLError::ConfigError(e.to_string()))?;
        Ok(config)
    }
    
    /// Load configuration from TOML string
    pub fn from_toml(content: &str) -> MLResult<Self> {
        toml::from_str(content)
            .map_err(|e| MLError::ConfigError(e.to_string()))
    }
    
    /// Save configuration to file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> MLResult<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| MLError::ConfigError(e.to_string()))?;
        fs::write(path, content)?;
        Ok(())
    }
    
    /// Create default configuration
    pub fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
            data: DataConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

/// Model architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub input_dim: i64,
    pub hidden_dims: Vec<i64>,
    pub output_dim: i64,
    pub dropout: f64,
    pub activation: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "mlp".to_string(),
            input_dim: 784,
            hidden_dims: vec![256, 128, 64],
            output_dim: 10,
            dropout: 0.3,
            activation: "relu".to_string(),
        }
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: i64,
    pub learning_rate: f64,
    pub optimizer: String,
    pub weight_decay: f64,
    pub gradient_clip: Option<f64>,
    pub early_stopping_patience: Option<usize>,
    pub checkpoint_interval: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 50,
            batch_size: 128,
            learning_rate: 1e-3,
            optimizer: "adam".to_string(),
            weight_decay: 0.0,
            gradient_clip: Some(1.0),
            early_stopping_patience: Some(10),
            checkpoint_interval: 10,
        }
    }
}

/// Data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub train_samples: i64,
    pub test_samples: i64,
    pub validation_split: f64,
    pub shuffle: bool,
    pub augmentation: bool,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            train_samples: 10000,
            test_samples: 2000,
            validation_split: 0.2,
            shuffle: true,
            augmentation: false,
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub log_interval: usize,
    pub save_metrics: bool,
    pub metrics_path: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            log_interval: 5,
            save_metrics: true,
            metrics_path: "metrics.json".to_string(),
        }
    }
}

/// CNN-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CNNConfig {
    pub conv_channels: Vec<i64>,
    pub kernel_sizes: Vec<i64>,
    pub pool_sizes: Vec<i64>,
    pub fc_dims: Vec<i64>,
}

impl Default for CNNConfig {
    fn default() -> Self {
        Self {
            conv_channels: vec![32, 64, 128],
            kernel_sizes: vec![3, 3, 3],
            pool_sizes: vec![2, 2, 2],
            fc_dims: vec![256, 128],
        }
    }
}

/// LSTM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTMConfig {
    pub vocab_size: i64,
    pub embedding_dim: i64,
    pub hidden_dim: i64,
    pub num_layers: i64,
    pub dropout: f64,
    pub bidirectional: bool,
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            vocab_size: 10000,
            embedding_dim: 128,
            hidden_dim: 256,
            num_layers: 2,
            dropout: 0.3,
            bidirectional: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.model.name, "mlp");
        assert_eq!(config.training.epochs, 50);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        assert!(toml_str.contains("[model]"));
        assert!(toml_str.contains("[training]"));
    }

    #[test]
    fn test_config_deserialization() {
        let toml_str = r#"
            [model]
            name = "test_model"
            input_dim = 100
            hidden_dims = [64, 32]
            output_dim = 5
            dropout = 0.5
            activation = "relu"
            
            [training]
            epochs = 10
            batch_size = 32
            learning_rate = 0.001
            optimizer = "adam"
            weight_decay = 0.0
            checkpoint_interval = 5
            
            [data]
            train_samples = 1000
            test_samples = 200
            validation_split = 0.2
            shuffle = true
            augmentation = false
            
            [logging]
            level = "debug"
            log_interval = 1
            save_metrics = true
            metrics_path = "test_metrics.json"
        "#;
        
        let config = Config::from_toml(toml_str).unwrap();
        assert_eq!(config.model.name, "test_model");
        assert_eq!(config.training.epochs, 10);
    }
}
