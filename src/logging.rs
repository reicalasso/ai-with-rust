// Logging utilities using tracing

use tracing::{info, warn, error, debug, Level};
use tracing_subscriber::{fmt, EnvFilter};
use std::io;

/// Initialize the logging system
pub fn init_logging() {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    
    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_level(true)
        .with_writer(io::stdout)
        .init();
    
    info!("🦀 Rust ML v2.0 - Logging initialized");
}

/// Initialize logging with custom level
pub fn init_logging_with_level(level: Level) {
    let filter = EnvFilter::new(level.to_string());
    
    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_thread_ids(false)
        .with_level(true)
        .with_writer(io::stdout)
        .init();
}

/// Log training metrics
pub fn log_training_metrics(epoch: usize, loss: f64, train_acc: f64, test_acc: f64, time_ms: f64) {
    info!(
        epoch = epoch,
        loss = format_args!("{:.4}", loss),
        train_acc = format_args!("{:.2}%", train_acc),
        test_acc = format_args!("{:.2}%", test_acc),
        time_ms = format_args!("{:.2}", time_ms),
        "Training progress"
    );
}

/// Log model information
pub fn log_model_info(name: &str, params: i64, memory_mb: f64) {
    info!(
        model = name,
        parameters = params,
        memory_mb = format_args!("{:.2}", memory_mb),
        "Model information"
    );
}

/// Log warning message
pub fn log_warning(context: &str, message: &str) {
    warn!(context = context, message = message);
}

/// Log error message
pub fn log_error(context: &str, error: &dyn std::error::Error) {
    error!(context = context, error = format!("{}", error));
}

/// Log debug message
pub fn log_debug(context: &str, message: &str) {
    debug!(context = context, message = message);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logging_initialization() {
        // This should not panic
        init_logging_with_level(Level::DEBUG);
    }
}
