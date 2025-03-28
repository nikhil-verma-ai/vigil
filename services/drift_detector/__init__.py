"""
Drift Detector service — Module 2 of the Autonomous Fine-Tuning Platform.

Consumes logprob.signals from Kafka, maintains per-(model_version, tenant_id)
rolling baselines, computes z-score anomaly composites, and emits DriftEvents
to drift.events when evidence-qualified thresholds are breached.
"""
