// Utility functions for ML operations

use tch::{Tensor, Kind, Device};
use std::time::Instant;

/// Calculate accuracy from predictions and labels
pub fn calculate_accuracy(predictions: &Tensor, labels: &Tensor) -> f64 {
    let correct = predictions.eq_tensor(labels).to_kind(Kind::Float).sum(Kind::Float);
    let total = labels.size()[0] as f64;
    (correct.double_value(&[]) / total) * 100.0
}

/// Calculate F1 score (binary classification)
pub fn calculate_f1_score(predictions: &Tensor, labels: &Tensor) -> f64 {
    let pred_eq_label = predictions.eq_tensor(labels);
    let label_eq_1 = labels.eq(1);
    let true_positives = (&pred_eq_label * &label_eq_1)
        .to_kind(Kind::Float).sum(Kind::Float).double_value(&[]);
    
    let pred_ne_label = predictions.ne_tensor(labels);
    let pred_eq_1 = predictions.eq(1);
    let false_positives = (&pred_ne_label * &pred_eq_1)
        .to_kind(Kind::Float).sum(Kind::Float).double_value(&[]);
    
    let pred_eq_0 = predictions.eq(0);
    let false_negatives = (&pred_ne_label * &pred_eq_0)
        .to_kind(Kind::Float).sum(Kind::Float).double_value(&[]);
    
    let precision = true_positives / (true_positives + false_positives + 1e-10);
    let recall = true_positives / (true_positives + false_negatives + 1e-10);
    
    2.0 * (precision * recall) / (precision + recall + 1e-10)
}

/// Calculate mean squared error
pub fn calculate_mse(predictions: &Tensor, targets: &Tensor) -> f64 {
    (predictions - targets).pow_tensor_scalar(2).mean(Kind::Float).double_value(&[])
}

/// Calculate R² score (coefficient of determination)
pub fn calculate_r2_score(predictions: &Tensor, targets: &Tensor) -> f64 {
    let mean = targets.mean(Kind::Float);
    let ss_tot = (targets - &mean).pow_tensor_scalar(2).sum(Kind::Float).double_value(&[]);
    let ss_res = (targets - predictions).pow_tensor_scalar(2).sum(Kind::Float).double_value(&[]);
    1.0 - (ss_res / ss_tot)
}

/// Normalize tensor to [0, 1] range
pub fn normalize_min_max(tensor: &Tensor) -> Tensor {
    let min = tensor.min();
    let max = tensor.max();
    (tensor - &min) / (&max - &min + 1e-10)
}

/// Standardize tensor (zero mean, unit variance)
pub fn standardize(tensor: &Tensor) -> Tensor {
    let mean = tensor.mean(Kind::Float);
    let std = tensor.std(false);
    (tensor - &mean) / (&std + 1e-10)
}

/// One-hot encode labels
pub fn one_hot_encode(labels: &Tensor, num_classes: i64) -> Tensor {
    let batch_size = labels.size()[0];
    let one_hot = Tensor::zeros(&[batch_size, num_classes], (labels.kind(), labels.device()));
    
    for i in 0..batch_size {
        let label = labels.int64_value(&[i]);
        let _ = one_hot.get(i).get(label).fill_(1.0);
    }
    
    one_hot
}

/// Split data into train and test sets
pub fn train_test_split(
    data: &Tensor,
    labels: &Tensor,
    test_ratio: f64,
) -> (Tensor, Tensor, Tensor, Tensor) {
    let total_samples = data.size()[0];
    let test_samples = (total_samples as f64 * test_ratio) as i64;
    let train_samples = total_samples - test_samples;
    
    let train_data = data.narrow(0, 0, train_samples);
    let test_data = data.narrow(0, train_samples, test_samples);
    let train_labels = labels.narrow(0, 0, train_samples);
    let test_labels = labels.narrow(0, train_samples, test_samples);
    
    (train_data, train_labels, test_data, test_labels)
}

/// Timer for benchmarking
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            start: Instant::now(),
            name: name.into(),
        }
    }
    
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
    
    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
    
    pub fn stop(&self) -> f64 {
        let elapsed = self.elapsed_ms();
        println!("⏱️  {} took {:.2}ms", self.name, elapsed);
        elapsed
    }
}

/// Progress bar for training
pub struct ProgressBar {
    total: usize,
    current: usize,
    width: usize,
    description: String,
}

impl ProgressBar {
    pub fn new(total: usize, description: impl Into<String>) -> Self {
        Self {
            total,
            current: 0,
            width: 50,
            description: description.into(),
        }
    }
    
    pub fn update(&mut self, current: usize) {
        self.current = current;
        self.display();
    }
    
    pub fn increment(&mut self) {
        self.current += 1;
        self.display();
    }
    
    fn display(&self) {
        let progress = self.current as f64 / self.total as f64;
        let filled = (progress * self.width as f64) as usize;
        let bar: String = "█".repeat(filled) + &"░".repeat(self.width - filled);
        
        print!("\r{} [{}] {}/{} ({:.1}%)", 
               self.description, bar, self.current, self.total, progress * 100.0);
        
        if self.current >= self.total {
            println!();
        }
    }
}

/// Moving average for smoothing metrics
pub struct MovingAverage {
    window: Vec<f64>,
    size: usize,
}

impl MovingAverage {
    pub fn new(size: usize) -> Self {
        Self {
            window: Vec::with_capacity(size),
            size,
        }
    }
    
    pub fn update(&mut self, value: f64) -> f64 {
        self.window.push(value);
        if self.window.len() > self.size {
            self.window.remove(0);
        }
        self.average()
    }
    
    pub fn average(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            self.window.iter().sum::<f64>() / self.window.len() as f64
        }
    }
}

/// Generate synthetic data for testing
pub fn generate_synthetic_classification_data(
    n_samples: i64,
    n_features: i64,
    n_classes: i64,
    device: Device,
) -> (Tensor, Tensor) {
    let x = Tensor::randn(&[n_samples, n_features], (Kind::Float, device));
    let y = Tensor::randint(n_classes, &[n_samples], (Kind::Int64, device));
    (x, y)
}

/// Generate synthetic regression data
pub fn generate_synthetic_regression_data(
    n_samples: i64,
    n_features: i64,
    device: Device,
) -> (Tensor, Tensor) {
    let x = Tensor::randn(&[n_samples, n_features], (Kind::Float, device));
    let weights = Tensor::randn(&[n_features, 1], (Kind::Float, device));
    let y = x.matmul(&weights) + Tensor::randn(&[n_samples, 1], (Kind::Float, device)) * 0.1;
    (x, y.squeeze())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_accuracy() {
        let predictions = Tensor::from_slice(&[1i64, 0, 1, 1, 0]);
        let labels = Tensor::from_slice(&[1i64, 0, 0, 1, 0]);
        let accuracy = calculate_accuracy(&predictions, &labels);
        assert_eq!(accuracy, 80.0);
    }

    #[test]
    fn test_normalize_min_max() {
        let tensor = Tensor::from_slice(&[0.0f32, 5.0, 10.0]);
        let normalized = normalize_min_max(&tensor);
        let values: Vec<f32> = Vec::<f32>::try_from(normalized).unwrap();
        assert!((values[0] - 0.0).abs() < 1e-5);
        assert!((values[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_moving_average() {
        let mut ma = MovingAverage::new(3);
        assert_eq!(ma.update(1.0), 1.0);
        assert_eq!(ma.update(2.0), 1.5);
        assert_eq!(ma.update(3.0), 2.0);
        assert_eq!(ma.update(4.0), 3.0); // Only last 3 values
    }
}

/// Learning Rate Scheduler trait
pub trait LRScheduler {
    fn step(&mut self, epoch: usize) -> f64;
    fn get_lr(&self) -> f64;
}

/// Step Learning Rate Scheduler
/// Decays the learning rate by gamma every step_size epochs
pub struct StepLR {
    initial_lr: f64,
    current_lr: f64,
    gamma: f64,
    step_size: usize,
}

impl StepLR {
    pub fn new(initial_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            gamma,
            step_size,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self, epoch: usize) -> f64 {
        if epoch > 0 && epoch % self.step_size == 0 {
            self.current_lr *= self.gamma;
        }
        self.current_lr
    }
    
    fn get_lr(&self) -> f64 {
        self.current_lr
    }
}

/// Cosine Annealing Learning Rate Scheduler
/// Decreases learning rate following a cosine curve
pub struct CosineAnnealingLR {
    initial_lr: f64,
    min_lr: f64,
    current_lr: f64,
    t_max: usize,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f64, t_max: usize, min_lr: f64) -> Self {
        Self {
            initial_lr,
            min_lr,
            current_lr: initial_lr,
            t_max,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self, epoch: usize) -> f64 {
        let progress = (epoch % self.t_max) as f64 / self.t_max as f64;
        self.current_lr = self.min_lr + 
            (self.initial_lr - self.min_lr) * (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
        self.current_lr
    }
    
    fn get_lr(&self) -> f64 {
        self.current_lr
    }
}

/// Exponential Learning Rate Scheduler
pub struct ExponentialLR {
    initial_lr: f64,
    current_lr: f64,
    gamma: f64,
}

impl ExponentialLR {
    pub fn new(initial_lr: f64, gamma: f64) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            gamma,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn step(&mut self, _epoch: usize) -> f64 {
        self.current_lr *= self.gamma;
        self.current_lr
    }
    
    fn get_lr(&self) -> f64 {
        self.current_lr
    }
}

/// Early Stopping utility
/// Stops training when validation metric stops improving
pub struct EarlyStopping {
    patience: usize,
    min_delta: f64,
    counter: usize,
    best_score: Option<f64>,
    should_stop: bool,
    mode: EarlyStoppingMode,
}

#[derive(Debug, Clone, Copy)]
pub enum EarlyStoppingMode {
    Min,  // For loss (lower is better)
    Max,  // For accuracy (higher is better)
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f64, mode: EarlyStoppingMode) -> Self {
        Self {
            patience,
            min_delta,
            counter: 0,
            best_score: None,
            should_stop: false,
            mode,
        }
    }
    
    /// Update early stopping state with new metric value
    /// Returns true if training should stop
    pub fn step(&mut self, metric: f64) -> bool {
        let improved = match self.best_score {
            None => {
                self.best_score = Some(metric);
                true
            }
            Some(best) => {
                let delta = match self.mode {
                    EarlyStoppingMode::Min => best - metric,
                    EarlyStoppingMode::Max => metric - best,
                };
                
                if delta > self.min_delta {
                    self.best_score = Some(metric);
                    true
                } else {
                    false
                }
            }
        };
        
        if improved {
            self.counter = 0;
        } else {
            self.counter += 1;
            if self.counter >= self.patience {
                self.should_stop = true;
            }
        }
        
        self.should_stop
    }
    
    pub fn should_stop(&self) -> bool {
        self.should_stop
    }
    
    pub fn best_score(&self) -> Option<f64> {
        self.best_score
    }
    
    pub fn reset(&mut self) {
        self.counter = 0;
        self.best_score = None;
        self.should_stop = false;
    }
}
