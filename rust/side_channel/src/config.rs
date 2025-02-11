/// Configuration for the inference side-channel agent.
///
/// Values are sourced from (in ascending priority order):
///   1. Compiled-in defaults
///   2. `config.toml` in the working directory
///   3. Environment variables with the `SIDE_CHANNEL_` prefix
///
/// All fields are validated on load via [`AppConfig::validate`].
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// Top-level application configuration.
///
/// # Fields
/// - `kafka_brokers`: Comma-separated list of `host:port` broker addresses.
/// - `kafka_topic`: Target topic name for `LogprobSignalEvent` messages.
/// - `tenant_id`: Opaque tenant identifier stamped on every event.
/// - `model_version`: Model version tag stamped on every event.
/// - `sampling_rate`: Fraction `[0.0, 1.0]` of events that pass to the publisher;
///   `1.0` means "capture everything", `0.5` means "capture ~50%".
/// - `buffer_max_events`: In-process back-pressure buffer depth (events).
///   When the ring buffer and this secondary buffer are both full, the oldest
///   entry is evicted.
/// - `prometheus_port`: TCP port for the `/metrics` HTTP endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppConfig {
    /// Comma-separated Kafka bootstrap brokers, e.g. `"localhost:9092"`.
    pub kafka_brokers: String,

    /// Kafka topic to publish `LogprobSignalEvent` messages to.
    #[serde(default = "default_kafka_topic")]
    pub kafka_topic: String,

    /// Tenant identifier embedded in every published event.
    pub tenant_id: String,

    /// Model version string embedded in every published event.
    pub model_version: String,

    /// Fraction of events to forward.  Must be in `[0.0, 1.0]`.
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: f64,

    /// Maximum number of events held in the in-process back-pressure buffer
    /// while Kafka is unavailable.
    #[serde(default = "default_buffer_max_events")]
    pub buffer_max_events: usize,

    /// TCP port on which Prometheus `/metrics` is exposed.
    #[serde(default = "default_prometheus_port")]
    pub prometheus_port: u16,
}

fn default_kafka_topic() -> String {
    "logprob.signals".to_string()
}

fn default_sampling_rate() -> f64 {
    1.0
}

fn default_buffer_max_events() -> usize {
    50_000
}

fn default_prometheus_port() -> u16 {
    9090
}

impl AppConfig {
    /// Load configuration from optional `config.toml` overlaid with environment
    /// variables that carry the `SIDE_CHANNEL_` prefix.
    ///
    /// # Errors
    /// Returns an error if the configuration cannot be parsed or fails
    /// validation (see [`AppConfig::validate`]).
    pub fn load() -> Result<Self> {
        let builder = config::Config::builder()
            // Defaults — all have serde defaults except required fields.
            .set_default("kafka_topic", "logprob.signals")?
            .set_default("sampling_rate", 1.0_f64)?
            .set_default("buffer_max_events", 50_000_i64)?
            .set_default("prometheus_port", 9090_i64)?
            // Optional TOML file in cwd.
            .add_source(
                config::File::with_name("config")
                    .format(config::FileFormat::Toml)
                    .required(false),
            )
            // Environment variables override everything.
            .add_source(
                config::Environment::with_prefix("SIDE_CHANNEL")
                    .separator("__")
                    .try_parsing(true),
            );

        let raw = builder.build().context("failed to build configuration")?;
        let cfg: AppConfig = raw.try_deserialize().context("failed to deserialise configuration")?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Load configuration from a set of explicit key-value overrides.
    ///
    /// Useful for unit tests that need full control over every field without
    /// touching environment variables or the filesystem.
    ///
    /// # Arguments
    /// - `overrides`: Iterator of `(key, value)` string pairs.  Keys must
    ///   match the `AppConfig` field names.
    pub fn from_overrides<I, K, V>(overrides: I) -> Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Into<config::Value>,
    {
        let mut builder = config::Config::builder()
            .set_default("kafka_topic", "logprob.signals")?
            .set_default("sampling_rate", 1.0_f64)?
            .set_default("buffer_max_events", 50_000_i64)?
            .set_default("prometheus_port", 9090_i64)?;

        for (k, v) in overrides {
            builder = builder.set_override(k.as_ref(), v)?;
        }

        let raw = builder.build().context("failed to build configuration from overrides")?;
        let cfg: AppConfig = raw.try_deserialize().context("failed to deserialise configuration from overrides")?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate all invariants that cannot be expressed as type constraints.
    ///
    /// # Errors
    /// - `sampling_rate` outside `[0.0, 1.0]`
    /// - `buffer_max_events` is zero
    /// - `kafka_brokers` is empty
    /// - `tenant_id` is empty
    /// - `model_version` is empty
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.sampling_rate) {
            bail!(
                "sampling_rate must be in [0.0, 1.0], got {}",
                self.sampling_rate
            );
        }
        if self.buffer_max_events == 0 {
            bail!("buffer_max_events must be > 0");
        }
        if self.kafka_brokers.trim().is_empty() {
            bail!("kafka_brokers must not be empty");
        }
        if self.tenant_id.trim().is_empty() {
            bail!("tenant_id must not be empty");
        }
        if self.model_version.trim().is_empty() {
            bail!("model_version must not be empty");
        }
        Ok(())
    }
}
