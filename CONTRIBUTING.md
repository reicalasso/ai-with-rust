# Contributing to Rust ML

Thank you for your interest in contributing to Rust ML! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or suggest features
- Search existing issues before creating a new one
- Provide detailed information:
  - Steps to reproduce (for bugs)
  - Expected vs actual behavior
  - System information (OS, Rust version, CUDA version)
  - Code samples if applicable

### Pull Requests

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-with-rust.git
   cd rust-ml
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Write clean, idiomatic Rust code
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   cargo test
   cargo clippy
   cargo fmt -- --check
   ```

5. **Commit**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```
   
   Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test additions/changes
   - `refactor:` Code refactoring
   - `perf:` Performance improvements

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub

## 📋 Code Style

### Rust Guidelines

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` for formatting
- Address all `cargo clippy` warnings
- Write descriptive variable and function names
- Add documentation comments for public APIs

### Example

```rust
/// Calculate the accuracy of predictions against ground truth labels.
///
/// # Arguments
///
/// * `predictions` - Tensor of predicted class indices
/// * `labels` - Tensor of ground truth class indices
///
/// # Returns
///
/// Accuracy as a percentage (0-100)
///
/// # Example
///
/// ```
/// use rust_ml::calculate_accuracy;
/// use tch::Tensor;
///
/// let predictions = Tensor::of_slice(&[1i64, 0, 1]);
/// let labels = Tensor::of_slice(&[1i64, 0, 0]);
/// let acc = calculate_accuracy(&predictions, &labels);
/// assert_eq!(acc, 66.67);
/// ```
pub fn calculate_accuracy(predictions: &Tensor, labels: &Tensor) -> f64 {
    // Implementation
}
```

## 🧪 Testing

### Writing Tests

- Add unit tests in the same file as the code
- Add integration tests in the `tests/` directory
- Add benchmarks in the `benches/` directory for performance-critical code

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_calculation() {
        let predictions = Tensor::of_slice(&[1i64, 0, 1, 1]);
        let labels = Tensor::of_slice(&[1i64, 0, 0, 1]);
        let acc = calculate_accuracy(&predictions, &labels);
        assert_eq!(acc, 75.0);
    }
}
```

### Running Tests

```bash
# All tests
cargo test

# Specific test
cargo test test_accuracy_calculation

# With output
cargo test -- --nocapture

# Integration tests only
cargo test --test '*'

# Benchmarks
cargo bench
```

## 📚 Documentation

### Code Documentation

- Document all public APIs with `///` doc comments
- Include examples in doc comments when helpful
- Document parameters, return values, and panics
- Keep documentation up-to-date with code changes

### Generating Documentation

```bash
cargo doc --open
```

## 🏗️ Architecture

### Adding New Models

1. Define model struct in appropriate module
2. Implement forward pass
3. Add tests
4. Add example usage
5. Update documentation

Example:

```rust
// src/models.rs

/// Custom neural network architecture
pub struct CustomNet {
    fc1: nn::Linear,
    fc2: nn::Linear,
    dropout: f64,
}

impl CustomNet {
    pub fn new(vs: &nn::Path, input_dim: i64, output_dim: i64) -> Self {
        // Implementation
    }
    
    pub fn forward(&self, xs: &Tensor) -> Tensor {
        // Implementation
    }
}

#[cfg(test)]
mod tests {
    // Tests
}
```

### Adding New Features

1. Discuss in GitHub Issues first for major features
2. Create a new module if appropriate
3. Follow existing patterns
4. Add comprehensive tests
5. Update CLI if needed
6. Add examples
7. Update README

## 🔍 Code Review Process

1. All PRs require review before merging
2. Address reviewer feedback
3. Keep PRs focused and reasonably sized
4. Ensure CI passes
5. Update PR description if scope changes

## 📊 Performance Considerations

- Profile code for performance-critical sections
- Use benchmarks to measure improvements
- Consider memory usage
- Document performance characteristics
- Use parallel processing where appropriate (Rayon)

## 🐛 Debugging

### Enable Logging

```bash
RUST_LOG=debug cargo run
```

### Use Debug Assertions

```rust
debug_assert!(condition, "Error message");
```

### Profiling

```bash
cargo install flamegraph
cargo flamegraph
```

## 📝 Commit Message Guidelines

```
<type>(<scope>): <subject>

<body>

<footer>
```

Example:

```
feat(models): add transformer architecture

Implement multi-head self-attention and position encoding
for sequence-to-sequence tasks.

Closes #123
```

## ✅ Checklist Before Submitting

- [ ] Code compiles without warnings
- [ ] All tests pass
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] Ran `cargo fmt`
- [ ] Ran `cargo clippy` and addressed warnings
- [ ] Updated CHANGELOG.md if applicable
- [ ] Followed commit message conventions

## 🙏 Thank You!

Your contributions help make Rust ML better for everyone. We appreciate your time and effort!

## 📧 Questions?

- Open a GitHub Discussion
- Check existing documentation
- Reach out to maintainers

---

**Happy Contributing! 🦀**
