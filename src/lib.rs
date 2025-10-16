// Library exports

pub mod models;
pub mod demos;
pub mod advanced;
pub mod showcases;
pub mod advanced_features;
pub mod error;
pub mod logging;
pub mod config;
pub mod checkpoint;
pub mod utils;
pub mod data;

// Re-export commonly used types
pub use error::{MLError, MLResult, TrainingResult};
pub use config::Config;
pub use checkpoint::{CheckpointManager, CheckpointMetadata};
pub use utils::{calculate_accuracy, Timer, ProgressBar};
pub use data::{DataLoader, Dataset, MemoryDataset};
