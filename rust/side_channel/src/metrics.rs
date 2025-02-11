/// Prometheus metrics registry and HTTP exposition server.
///
/// # Metrics exposed
/// | Name | Type | Labels | Description |
/// |------|------|--------|-------------|
/// | `inference_side_channel_capture_latency_ns` | Histogram | — | Hot-path capture latency |
/// | `inference_side_channel_events_total` | Counter | `status` | Events captured or dropped |
/// | `inference_side_channel_kafka_buffer_depth` | Gauge | — | Back-pressure buffer depth |
/// | `inference_side_channel_publish_latency_us` | Histogram | — | Kafka round-trip latency |
///
/// # Exposition
/// A lightweight Hyper HTTP server listens on `0.0.0.0:{config.prometheus_port}`
/// and responds to every request on `/metrics` with the Prometheus text format.
use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use prometheus::{
    Gauge, Histogram, HistogramOpts, IntCounterVec, Opts, Registry,
    TextEncoder,
};
use tokio::net::TcpListener;
use tracing::{error, info};

/// All Prometheus metric handles owned by this module.
///
/// Clone cheaply — every inner type is backed by `Arc`.
#[derive(Clone, Debug)]
pub struct Metrics {
    pub registry: Registry,

    /// Histogram of hot-path capture latencies in nanoseconds.
    ///
    /// Buckets: 500 ns, 1 µs, 2 µs, 5 µs, 10 µs, 50 µs, 100 µs, +Inf
    pub capture_latency_ns: Histogram,

    /// Counter of events processed; label `status` ∈ `{captured, dropped}`.
    pub events_total: IntCounterVec,

    /// Gauge: current depth of the in-process Kafka back-pressure buffer.
    pub kafka_buffer_depth: Gauge,

    /// Histogram of end-to-end Kafka publish latencies in microseconds.
    ///
    /// Buckets: 100 µs, 500 µs, 1 ms, 5 ms, 10 ms, 50 ms, 100 ms, +Inf
    pub publish_latency_us: Histogram,
}

impl Metrics {
    /// Create and register all metrics into a fresh `Registry`.
    ///
    /// # Errors
    /// Returns an error if any metric fails to register (duplicate name, etc.).
    pub fn new() -> Result<Self> {
        let registry = Registry::new();

        // ── Capture latency histogram ─────────────────────────────────────
        let capture_latency_ns = Histogram::with_opts(
            HistogramOpts::new(
                "inference_side_channel_capture_latency_ns",
                "Hot-path capture latency in nanoseconds",
            )
            .buckets(vec![500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0, 50_000.0, 100_000.0]),
        )
        .context("failed to create capture_latency_ns histogram")?;
        registry
            .register(Box::new(capture_latency_ns.clone()))
            .context("failed to register capture_latency_ns")?;

        // ── Events total counter ──────────────────────────────────────────
        let events_total = IntCounterVec::new(
            Opts::new(
                "inference_side_channel_events_total",
                "Total events processed by the side-channel agent",
            ),
            &["status"],
        )
        .context("failed to create events_total counter")?;
        registry
            .register(Box::new(events_total.clone()))
            .context("failed to register events_total")?;

        // Eagerly initialise both label values so the metric appears in
        // /metrics even before any events are captured.
        events_total.with_label_values(&["captured"]);
        events_total.with_label_values(&["dropped"]);

        // ── Kafka buffer depth gauge ──────────────────────────────────────
        let kafka_buffer_depth = Gauge::new(
            "inference_side_channel_kafka_buffer_depth",
            "Number of events buffered in-process due to Kafka back-pressure",
        )
        .context("failed to create kafka_buffer_depth gauge")?;
        registry
            .register(Box::new(kafka_buffer_depth.clone()))
            .context("failed to register kafka_buffer_depth")?;

        // ── Publish latency histogram ─────────────────────────────────────
        let publish_latency_us = Histogram::with_opts(
            HistogramOpts::new(
                "inference_side_channel_publish_latency_us",
                "End-to-end Kafka publish latency in microseconds",
            )
            .buckets(vec![
                100.0, 500.0, 1_000.0, 5_000.0, 10_000.0, 50_000.0, 100_000.0,
            ]),
        )
        .context("failed to create publish_latency_us histogram")?;
        registry
            .register(Box::new(publish_latency_us.clone()))
            .context("failed to register publish_latency_us")?;

        Ok(Self {
            registry,
            capture_latency_ns,
            events_total,
            kafka_buffer_depth,
            publish_latency_us,
        })
    }

    /// Render all registered metrics in Prometheus text format.
    ///
    /// # Errors
    /// Returns an error if the `TextEncoder` fails to gather metrics.
    pub fn render_text(&self) -> Result<String> {
        let encoder = TextEncoder::new();
        let families = self.registry.gather();
        encoder
            .encode_to_string(&families)
            .context("failed to encode metrics")
    }
}

// ── HTTP server ───────────────────────────────────────────────────────────────

/// Start the Prometheus metrics HTTP server on the configured port.
///
/// Responds to any path with the Prometheus text format exposition.  The
/// server runs until the `shutdown` future resolves.
///
/// # Arguments
/// - `metrics`: Shared metrics handle.
/// - `port`: TCP port to bind on `0.0.0.0`.
/// - `shutdown`: A future that resolves when the server should stop accepting
///   new connections.
///
/// # Errors
/// Returns an error if the TCP listener cannot be bound.
pub async fn serve_metrics(
    metrics: Arc<Metrics>,
    port: u16,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> Result<()> {
    let addr: SocketAddr = ([0, 0, 0, 0], port).into();
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind metrics server on :{port}"))?;

    info!(addr = %addr, "Prometheus metrics server listening");

    loop {
        tokio::select! {
            accept_result = listener.accept() => {
                match accept_result {
                    Ok((stream, peer)) => {
                        let metrics_clone = Arc::clone(&metrics);
                        let io = TokioIo::new(stream);
                        tokio::spawn(async move {
                            let svc = service_fn(move |req: Request<hyper::body::Incoming>| {
                                let m = Arc::clone(&metrics_clone);
                                async move { handle_metrics_request(req, m).await }
                            });
                            if let Err(e) = http1::Builder::new().serve_connection(io, svc).await {
                                error!(peer = %peer, "metrics connection error: {e}");
                            }
                        });
                    }
                    Err(e) => {
                        error!("metrics accept error: {e}");
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("metrics server shutting down");
                    break;
                }
            }
        }
    }

    Ok(())
}

/// Handle a single HTTP request to the metrics endpoint.
async fn handle_metrics_request(
    _req: Request<hyper::body::Incoming>,
    metrics: Arc<Metrics>,
) -> Result<Response<Full<Bytes>>, hyper::Error> {
    match metrics.render_text() {
        Ok(body) => Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/plain; version=0.0.4")
            .body(Full::new(Bytes::from(body)))
            .unwrap()),
        Err(e) => {
            error!("failed to render metrics: {e}");
            Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::new(Bytes::from("metrics unavailable")))
                .unwrap())
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_registry_creation() {
        let metrics = Metrics::new().expect("metrics creation should succeed");
        let text = metrics.render_text().expect("render should succeed");
        assert!(
            text.contains("inference_side_channel_events_total"),
            "events_total not found in output"
        );
        assert!(
            text.contains("inference_side_channel_capture_latency_ns"),
            "capture_latency_ns not found"
        );
        assert!(
            text.contains("inference_side_channel_kafka_buffer_depth"),
            "kafka_buffer_depth not found"
        );
        assert!(
            text.contains("inference_side_channel_publish_latency_us"),
            "publish_latency_us not found"
        );
    }

    #[test]
    fn test_counter_increments() {
        let metrics = Metrics::new().unwrap();
        metrics.events_total.with_label_values(&["captured"]).inc();
        metrics.events_total.with_label_values(&["captured"]).inc();
        metrics.events_total.with_label_values(&["dropped"]).inc();

        let text = metrics.render_text().unwrap();
        // Both label values should appear.
        assert!(text.contains(r#"status="captured""#));
        assert!(text.contains(r#"status="dropped""#));
    }
}
