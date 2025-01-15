"""
Canonical event schemas for all Kafka topics.
All services import from here — never redefine these.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, List, Dict, Any
import json
import time
import uuid


class AlertLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class DriftType(str, Enum):
    INPUT_DISTRIBUTION = "INPUT_DISTRIBUTION"
    CONCEPT = "CONCEPT"
    CATASTROPHIC = "CATASTROPHIC"
    GRADUAL = "GRADUAL"


class AdapterStatus(str, Enum):
    CANDIDATE = "CANDIDATE"
    STAGING = "STAGING"
    PROMOTING = "PROMOTING"
    PRODUCTION = "PRODUCTION"
    SUPERSEDED = "SUPERSEDED"
    ROLLED_BACK = "ROLLED_BACK"


class LoopState(str, Enum):
    IDLE = "IDLE"
    EVIDENCE_QUALIFYING = "EVIDENCE_QUALIFYING"
    SYNTHESIZING = "SYNTHESIZING"
    TRAINING_SFT = "TRAINING_SFT"
    TRAINING_DPO = "TRAINING_DPO"
    EVALUATING = "EVALUATING"
    DEPLOYING_CANARY = "DEPLOYING_CANARY"
    PROMOTING_FULL = "PROMOTING_FULL"
    ROLLED_BACK = "ROLLED_BACK"
    REJECTED = "REJECTED"


class TriggerType(str, Enum):
    DRIFT = "DRIFT"
    SCHEDULE = "SCHEDULE"
    MANUAL = "MANUAL"


@dataclass
class TokenUncertaintySpike:
    position: int
    logprob: float


@dataclass
class LogprobPercentiles:
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    p99: float


@dataclass
class LogprobSignalEvent:
    """Topic: logprob.signals — emitted by Rust side-channel per inference request."""
    request_id: str
    timestamp_ns: int
    model_version: str
    tenant_id: str
    input_token_count: int
    output_token_count: int
    mean_logprob: float
    min_logprob: float
    logprob_entropy_mean: float
    logprob_variance: float
    token_uncertainty_spikes: List[TokenUncertaintySpike]
    sequence_logprob_percentiles: LogprobPercentiles

    @classmethod
    def make_test_event(
        cls,
        request_id: Optional[str] = None,
        mean_logprob: float = -1.5,
        logprob_entropy_mean: float = 2.0,
        logprob_variance: float = 0.3,
        model_version: str = "adapter-v1",
        tenant_id: str = "test-tenant",
    ) -> "LogprobSignalEvent":
        """Factory for test events with sensible defaults."""
        return cls(
            request_id=request_id or str(uuid.uuid4()),
            timestamp_ns=time.time_ns(),
            model_version=model_version,
            tenant_id=tenant_id,
            input_token_count=128,
            output_token_count=64,
            mean_logprob=mean_logprob,
            min_logprob=mean_logprob - 1.0,
            logprob_entropy_mean=logprob_entropy_mean,
            logprob_variance=logprob_variance,
            token_uncertainty_spikes=[],
            sequence_logprob_percentiles=LogprobPercentiles(
                p10=mean_logprob - 0.5,
                p25=mean_logprob - 0.3,
                p50=mean_logprob,
                p75=mean_logprob + 0.2,
                p90=mean_logprob + 0.4,
                p99=mean_logprob + 0.8,
            ),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, data: str) -> "LogprobSignalEvent":
        d = json.loads(data)
        d["token_uncertainty_spikes"] = [
            TokenUncertaintySpike(**s) for s in d["token_uncertainty_spikes"]
        ]
        d["sequence_logprob_percentiles"] = LogprobPercentiles(
            **d["sequence_logprob_percentiles"]
        )
        return cls(**d)


@dataclass
class SignalBreakdown:
    mean_logprob_zscore: float
    variance_zscore: float
    entropy_zscore: float
    spike_frequency_zscore: float


@dataclass
class DriftEvent:
    """Topic: drift.events — emitted by drift detector."""
    event_id: str
    detected_at: str
    model_version: str
    alert_level: AlertLevel
    drift_type: DriftType
    composite_anomaly_score: float
    signal_breakdown: SignalBreakdown
    affected_request_fraction: float
    evidence_request_ids: List[str]
    qualification_status: str  # PENDING | QUALIFIED | REJECTED
    training_cycle_id: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class DatasetStats:
    failure_examples_input: int
    preference_pairs_synthesized: int
    pairs_passing_quality_gate: int
    amplification_factor: float


@dataclass
class TrainingCycleEvent:
    """Topic: training.jobs — emitted by orchestrator."""
    cycle_id: str
    triggered_by: TriggerType
    trigger_event_id: Optional[str]
    started_at: str
    status: str  # RUNNING | COMPLETED | FAILED | CANCELLED
    base_adapter_id: str
    candidate_adapter_id: Optional[str] = None
    dataset_stats: Optional[DatasetStats] = None
    cost_usd: float = 0.0
    gpu_hours: float = 0.0
    completed_at: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class EvaluationResults:
    tier1_regression_passed: bool
    tier2_behavioral_passed: bool
    tier3_improvement_verified: bool
    tier4_safety_passed: bool
    all_passed: bool
    score_deltas: Dict[str, float]


@dataclass
class AdapterVersion:
    """Core adapter entity stored in adapter registry."""
    adapter_id: str
    version: str
    created_at: str
    base_model_id: str
    training_cycle_id: str
    status: AdapterStatus
    evaluation_results: Optional[EvaluationResults] = None
    artifact_location: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class AdapterPromotionEvent:
    """Topic: adapter.promotions — emitted by deployment engine."""
    adapter_id: str
    previous_adapter_id: Optional[str]
    event_type: str  # PROMOTED | ROLLED_BACK | CANARY_START | CANARY_PASS | CANARY_FAIL
    timestamp: str
    cluster_id: str
    reason: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# Kafka topic names — single source of truth
TOPIC_LOGPROB_SIGNALS = "logprob.signals"
TOPIC_DRIFT_EVENTS = "drift.events"
TOPIC_SYNTHESIS_JOBS = "synthesis.jobs"
TOPIC_TRAINING_JOBS = "training.jobs"
TOPIC_EVALUATION_RESULTS = "evaluation.results"
TOPIC_ADAPTER_PROMOTIONS = "adapter.promotions"
