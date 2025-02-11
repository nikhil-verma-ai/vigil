/// Zero-Copy Inference Side-Channel Agent — library entry point.
///
/// Re-exports all public modules so that integration tests in `tests/`
/// and downstream crates can import from `side_channel::*`.
pub mod capture;
pub mod config;
pub mod metrics;
pub mod publisher;
