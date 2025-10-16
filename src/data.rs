// Data loading and preprocessing utilities

use tch::{Tensor, Kind, Device};
use std::collections::VecDeque;
use crate::error::{MLError, MLResult};

/// Data loader with batching and shuffling
pub struct DataLoader {
    data: Tensor,
    labels: Tensor,
    batch_size: i64,
    shuffle: bool,
    device: Device,
    current_idx: i64,
    indices: Vec<i64>,
}

impl DataLoader {
    pub fn new(data: Tensor, labels: Tensor, batch_size: i64, shuffle: bool) -> Self {
        let n_samples = data.size()[0];
        let device = data.device();
        let mut indices: Vec<i64> = (0..n_samples).collect();
        
        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }
        
        Self {
            data,
            labels,
            batch_size,
            shuffle,
            device,
            current_idx: 0,
            indices,
        }
    }
    
    /// Get next batch
    pub fn next_batch(&mut self) -> Option<(Tensor, Tensor)> {
        if self.current_idx >= self.indices.len() as i64 {
            return None;
        }
        
        let end_idx = (self.current_idx + self.batch_size).min(self.indices.len() as i64);
        let batch_indices = &self.indices[self.current_idx as usize..end_idx as usize];
        
        let indices_tensor = Tensor::from_slice(batch_indices).to_device(self.device);
        let batch_data = self.data.index_select(0, &indices_tensor);
        let batch_labels = self.labels.index_select(0, &indices_tensor);
        
        self.current_idx = end_idx;
        
        Some((batch_data, batch_labels))
    }
    
    /// Reset iterator
    pub fn reset(&mut self) {
        self.current_idx = 0;
        
        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }
    
    /// Get number of batches
    pub fn num_batches(&self) -> i64 {
        (self.indices.len() as i64 + self.batch_size - 1) / self.batch_size
    }
}

/// Iterator implementation for DataLoader
impl Iterator for DataLoader {
    type Item = (Tensor, Tensor);
    
    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

/// Dataset trait for custom datasets
pub trait Dataset {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> MLResult<(Tensor, Tensor)>;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Simple in-memory dataset
pub struct MemoryDataset {
    data: Vec<Tensor>,
    labels: Vec<Tensor>,
}

impl MemoryDataset {
    pub fn new(data: Vec<Tensor>, labels: Vec<Tensor>) -> MLResult<Self> {
        if data.len() != labels.len() {
            return Err(MLError::DataError(
                "Data and labels must have same length".to_string()
            ));
        }
        Ok(Self { data, labels })
    }
}

impl Dataset for MemoryDataset {
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn get(&self, index: usize) -> MLResult<(Tensor, Tensor)> {
        if index >= self.len() {
            return Err(MLError::DataError(format!("Index {} out of bounds", index)));
        }
        Ok((self.data[index].shallow_clone(), self.labels[index].shallow_clone()))
    }
}

/// Data augmentation for images
pub struct ImageAugmentation {
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
    pub rotate: bool,
    pub noise_std: f64,
}

impl ImageAugmentation {
    pub fn new() -> Self {
        Self {
            flip_horizontal: true,
            flip_vertical: false,
            rotate: false,
            noise_std: 0.01,
        }
    }
    
    pub fn augment(&self, image: &Tensor) -> Tensor {
        let mut img = image.shallow_clone();
        
        // Horizontal flip
        if self.flip_horizontal && rand::random::<bool>() {
            img = img.flip(&[-1]);
        }
        
        // Vertical flip
        if self.flip_vertical && rand::random::<bool>() {
            img = img.flip(&[-2]);
        }
        
        // Add noise
        if self.noise_std > 0.0 {
            let noise = Tensor::randn_like(&img) * self.noise_std;
            img = img + noise;
        }
        
        img
    }
}

impl Default for ImageAugmentation {
    fn default() -> Self {
        Self::new()
    }
}

/// Replay buffer for reinforcement learning
pub struct ReplayBuffer {
    capacity: usize,
    buffer: VecDeque<(Tensor, Tensor, f64, Tensor, bool)>, // (state, action, reward, next_state, done)
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
        }
    }
    
    pub fn push(&mut self, state: Tensor, action: Tensor, reward: f64, next_state: Tensor, done: bool) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back((state, action, reward, next_state, done));
    }
    
    pub fn sample(&self, batch_size: usize) -> Option<Vec<(Tensor, Tensor, f64, Tensor, bool)>> {
        if self.buffer.len() < batch_size {
            return None;
        }
        
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let indices: Vec<usize> = (0..self.buffer.len()).collect();
        let selected_indices: Vec<usize> = indices
            .choose_multiple(&mut rng, batch_size)
            .copied()
            .collect();
        
        let samples: Vec<_> = selected_indices
            .iter()
            .map(|&i| {
                let (s, a, r, ns, d) = &self.buffer[i];
                (s.shallow_clone(), a.shallow_clone(), *r, ns.shallow_clone(), *d)
            })
            .collect();
        
        Some(samples)
    }
    
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

/// Time series data processor
pub struct TimeSeriesProcessor {
    window_size: usize,
    stride: usize,
}

impl TimeSeriesProcessor {
    pub fn new(window_size: usize, stride: usize) -> Self {
        Self { window_size, stride }
    }
    
    /// Create sliding windows from time series data
    pub fn create_windows(&self, data: &Tensor) -> (Tensor, Tensor) {
        let seq_len = data.size()[0] as usize;
        let n_features = if data.dim() > 1 { data.size()[1] } else { 1 };
        
        let mut windows = Vec::new();
        let mut targets = Vec::new();
        
        let mut i = 0;
        while i + self.window_size < seq_len {
            let window = data.narrow(0, i as i64, self.window_size as i64);
            let target = data.narrow(0, (i + self.window_size) as i64, 1);
            
            windows.push(window);
            targets.push(target);
            
            i += self.stride;
        }
        
        if windows.is_empty() {
            return (
                Tensor::zeros(&[0, self.window_size as i64, n_features], (Kind::Float, Device::Cpu)),
                Tensor::zeros(&[0, n_features], (Kind::Float, Device::Cpu)),
            );
        }
        
        let x = Tensor::stack(&windows, 0);
        let y = Tensor::stack(&targets, 0).squeeze_dim(1);
        
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_loader() {
        let data = Tensor::randn(&[100, 10], (Kind::Float, Device::Cpu));
        let labels = Tensor::randint(5, &[100], (Kind::Int64, Device::Cpu));
        
        let mut loader = DataLoader::new(data, labels, 32, false);
        
        assert_eq!(loader.num_batches(), 4); // 100 / 32 = 3.125 -> 4 batches
        
        let mut batch_count = 0;
        while let Some((batch_data, batch_labels)) = loader.next_batch() {
            assert!(batch_data.size()[0] <= 32);
            assert_eq!(batch_data.size()[0], batch_labels.size()[0]);
            batch_count += 1;
        }
        
        assert_eq!(batch_count, 4);
    }

    #[test]
    fn test_replay_buffer() {
        let mut buffer = ReplayBuffer::new(100);
        
        for i in 0..50 {
            let state = Tensor::of_slice(&[i as f32]);
            let action = Tensor::of_slice(&[0i64]);
            let next_state = Tensor::of_slice(&[(i + 1) as f32]);
            buffer.push(state, action, 1.0, next_state, false);
        }
        
        assert_eq!(buffer.len(), 50);
        
        let batch = buffer.sample(10);
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 10);
    }
}
