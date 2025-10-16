// Model serialization and checkpointing

use crate::error::{MLError, MLResult};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::fs;
use tch::nn::VarStore;
use chrono::{DateTime, Utc};

/// Model checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub model_name: String,
    pub version: String,
    pub epoch: usize,
    pub timestamp: DateTime<Utc>,
    pub train_loss: f64,
    pub test_accuracy: f64,
    pub hyperparameters: serde_json::Value,
    pub description: Option<String>,
}

impl CheckpointMetadata {
    pub fn new(
        model_name: String,
        version: String,
        epoch: usize,
        train_loss: f64,
        test_accuracy: f64,
        hyperparameters: serde_json::Value,
    ) -> Self {
        Self {
            model_name,
            version,
            epoch,
            timestamp: Utc::now(),
            train_loss,
            test_accuracy,
            hyperparameters,
            description: None,
        }
    }
    
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }
}

/// Checkpoint manager for saving and loading models
pub struct CheckpointManager {
    checkpoint_dir: String,
}

impl CheckpointManager {
    pub fn new(checkpoint_dir: impl Into<String>) -> MLResult<Self> {
        let dir = checkpoint_dir.into();
        fs::create_dir_all(&dir)?;
        Ok(Self { checkpoint_dir: dir })
    }
    
    /// Save model checkpoint
    pub fn save_checkpoint(
        &self,
        vs: &VarStore,
        metadata: &CheckpointMetadata,
    ) -> MLResult<String> {
        let checkpoint_name = format!(
            "{}_epoch_{}_{}",
            metadata.model_name,
            metadata.epoch,
            metadata.timestamp.format("%Y%m%d_%H%M%S")
        );
        
        // Save model weights
        let weights_path = format!("{}/{}.pt", self.checkpoint_dir, checkpoint_name);
        vs.save(&weights_path)
            .map_err(|e| MLError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string()
            )))?;
        
        // Save metadata
        let metadata_path = format!("{}/{}.json", self.checkpoint_dir, checkpoint_name);
        let metadata_json = serde_json::to_string_pretty(metadata)?;
        fs::write(&metadata_path, metadata_json)?;
        
        Ok(checkpoint_name)
    }
    
    /// Load model checkpoint
    pub fn load_checkpoint(
        &self,
        vs: &mut VarStore,
        checkpoint_name: &str,
    ) -> MLResult<CheckpointMetadata> {
        // Load weights
        let weights_path = format!("{}/{}.pt", self.checkpoint_dir, checkpoint_name);
        vs.load(&weights_path)
            .map_err(|e| MLError::IoError(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string()
            )))?;
        
        // Load metadata
        let metadata_path = format!("{}/{}.json", self.checkpoint_dir, checkpoint_name);
        let metadata_json = fs::read_to_string(&metadata_path)?;
        let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)?;
        
        Ok(metadata)
    }
    
    /// List all checkpoints
    pub fn list_checkpoints(&self) -> MLResult<Vec<String>> {
        let mut checkpoints = Vec::new();
        
        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("pt") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    checkpoints.push(stem.to_string());
                }
            }
        }
        
        checkpoints.sort();
        Ok(checkpoints)
    }
    
    /// Get latest checkpoint
    pub fn get_latest_checkpoint(&self) -> MLResult<Option<String>> {
        let checkpoints = self.list_checkpoints()?;
        Ok(checkpoints.last().cloned())
    }
    
    /// Delete old checkpoints, keeping only the last N
    pub fn cleanup_old_checkpoints(&self, keep_last_n: usize) -> MLResult<()> {
        let mut checkpoints = self.list_checkpoints()?;
        
        if checkpoints.len() <= keep_last_n {
            return Ok(());
        }
        
        checkpoints.sort();
        let to_delete = &checkpoints[..checkpoints.len() - keep_last_n];
        
        for checkpoint in to_delete {
            let weights_path = format!("{}/{}.pt", self.checkpoint_dir, checkpoint);
            let metadata_path = format!("{}/{}.json", self.checkpoint_dir, checkpoint);
            
            if Path::new(&weights_path).exists() {
                fs::remove_file(weights_path)?;
            }
            if Path::new(&metadata_path).exists() {
                fs::remove_file(metadata_path)?;
            }
        }
        
        Ok(())
    }
}

/// Training state for resuming training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    pub epoch: usize,
    pub best_loss: f64,
    pub best_accuracy: f64,
    pub optimizer_state: Option<String>, // Path to optimizer state
}

impl TrainingState {
    pub fn new(epoch: usize, best_loss: f64, best_accuracy: f64) -> Self {
        Self {
            epoch,
            best_loss,
            best_accuracy,
            optimizer_state: None,
        }
    }
    
    pub fn save<P: AsRef<Path>>(&self, path: P) -> MLResult<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }
    
    pub fn load<P: AsRef<Path>>(path: P) -> MLResult<Self> {
        let json = fs::read_to_string(path)?;
        let state: TrainingState = serde_json::from_str(&json)?;
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{nn, Device};
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_metadata() {
        let metadata = CheckpointMetadata::new(
            "test_model".to_string(),
            "1.0.0".to_string(),
            10,
            0.5,
            0.95,
            serde_json::json!({"lr": 0.001}),
        );
        
        assert_eq!(metadata.model_name, "test_model");
        assert_eq!(metadata.epoch, 10);
    }

    #[test]
    fn test_checkpoint_manager() -> MLResult<()> {
        let temp_dir = TempDir::new().unwrap();
        let manager = CheckpointManager::new(temp_dir.path().to_str().unwrap())?;
        
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);
        let _layer = nn::linear(&vs.root(), 10, 5, Default::default());
        
        let metadata = CheckpointMetadata::new(
            "test".to_string(),
            "1.0".to_string(),
            1,
            0.1,
            0.9,
            serde_json::json!({}),
        );
        
        let checkpoint_name = manager.save_checkpoint(&vs, &metadata)?;
        assert!(!checkpoint_name.is_empty());
        
        let checkpoints = manager.list_checkpoints()?;
        assert_eq!(checkpoints.len(), 1);
        
        Ok(())
    }
}
