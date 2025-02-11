/// Zero-Copy Inference Side-Channel Agent — entry point.
///
/// # Runtime topology
/// ```text
///  main thread
///   ├── load AppConfig
///   ├── init tracing subscriber
///   ├── start Prometheus metrics server (Tokio task)
///   ├── [MOCK_MODE=1] spawn MockVllmSource @ 10k events/sec
///   ├── spawn KafkaPublisher task
///   └── await SIGTERM / SIGINT → graceful shutdown
/// ```
///
/// # Environment variables
/// | Variable | Effect |
/// |----------|--------|
/// | `MOCK_MODE=1` | Spawn `MockVllmSource` at 10k events/sec |
/// | `SIDE_CHANNEL__KAFKA_BROKERS` | Override Kafka brokers |
/// | `SIDE_CHANNEL__TENANT_ID` | Override tenant ID |
/// | `SIDE_CHANNEL__MODEL_VERSION` | Override model version |
/// | `RUST_LOG` | Tracing filter, e.g. `info,side_channel=debug` |
use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::{info, warn};

use side_channel::capture::{make_channel, MockVllmSource, CaptureStats};
use side_channel::config::AppConfig;
use side_channel::metrics::{serve_metrics, Metrics};
use side_channel::publisher::{KafkaPublisher, PublisherStats, RdKafkaBackend};

#[tokio::main]
async fn main() -> Result<()> {
    // ── Tracing ───────────────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .json()
        .init();

    info!("inference side-channel agent starting");

    // ── Configuration ─────────────────────────────────────────────────────
    let config = AppConfig::load().context("failed to load configuration")?;
    info!(
        kafka_brokers = %config.kafka_brokers,
        kafka_topic = %config.kafka_topic,
        tenant_id = %config.tenant_id,
        model_version = %config.model_version,
        sampling_rate = config.sampling_rate,
        buffer_max_events = config.buffer_max_events,
        "configuration loaded"
    );

    // ── Metrics ───────────────────────────────────────────────────────────
    let metrics = Arc::new(Metrics::new().context("failed to initialise metrics")?);

    // Shutdown broadcast channel: when set to `true` all subsystems exit.
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    // Start Prometheus server.
    let metrics_handle = {
        let m = Arc::clone(&metrics);
        let rx = shutdown_rx.clone();
        let port = config.prometheus_port;
        tokio::spawn(async move {
            if let Err(e) = serve_metrics(m, port, rx).await {
                warn!("metrics server error: {e}");
            }
        })
    };

    // ── Ring buffer + stats ────────────────────────────────────────────────
    let buf_capacity = config.buffer_max_events;
    let (sender, receiver) = make_channel(buf_capacity);
    let capture_stats = CaptureStats::new_shared();
    let publisher_stats = PublisherStats::new_shared();

    // ── Optional mock source ──────────────────────────────────────────────
    let mock_source = if std::env::var("MOCK_MODE").as_deref() == Ok("1") {
        info!("MOCK_MODE=1: spawning MockVllmSource at 10_000 events/sec");
        Some(MockVllmSource::start(
            sender.clone(),
            Arc::clone(&capture_stats),
            10_000,
            config.model_version.clone(),
            config.tenant_id.clone(),
        ))
    } else {
        None
    };

    // ── Publisher task ─────────────────────────────────────────────────────
    let producer = RdKafkaBackend::new(&config)
        .context("failed to create Kafka producer")?;

    let publisher = KafkaPublisher::new(
        config.clone(),
        producer,
        Arc::clone(&publisher_stats),
        Arc::clone(&metrics),
    );

    let publisher_handle = tokio::spawn(async move {
        if let Err(e) = publisher.run(receiver).await {
            warn!("publisher exited with error: {e}");
        }
    });

    // ── Graceful shutdown on SIGTERM / SIGINT ─────────────────────────────
    wait_for_signal().await;
    info!("shutdown signal received — draining");

    // Stop mock source first so no more events enter the ring buffer.
    if let Some(src) = mock_source {
        src.stop();
    }

    // Signal all subsystems.
    let _ = shutdown_tx.send(true);

    // Wait for publisher to drain — it exits automatically when the channel
    // is closed and the back-pressure buffer is empty.
    publisher_handle.await.ok();
    metrics_handle.await.ok();

    info!(
        captured = capture_stats.events_captured.load(std::sync::atomic::Ordering::Relaxed),
        dropped = capture_stats.events_dropped.load(std::sync::atomic::Ordering::Relaxed),
        published = publisher_stats.events_published.load(std::sync::atomic::Ordering::Relaxed),
        "side-channel agent stopped"
    );

    Ok(())
}

/// Wait for SIGTERM or SIGINT (Ctrl-C) via Tokio's signal API.
async fn wait_for_signal() {
    use tokio::signal;

    #[cfg(unix)]
    {
        use signal::unix::{signal, SignalKind};
        let mut sigterm = signal(SignalKind::terminate())
            .expect("failed to register SIGTERM handler");
        tokio::select! {
            _ = sigterm.recv() => { info!("received SIGTERM"); }
            _ = signal::ctrl_c() => { info!("received SIGINT"); }
        }
    }

    #[cfg(not(unix))]
    {
        signal::ctrl_c()
            .await
            .expect("failed to register Ctrl-C handler");
        info!("received Ctrl-C");
    }
}
