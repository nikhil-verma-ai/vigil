"""
Cost tracking for training cycles.

Provides:
  - CostRecord:  immutable record of a single cost event
  - CostTracker: accumulates cost records per cycle and enforces a budget guard

Invariants:
  - total_for_cycle() is always the sum of all records for that cycle_id
  - would_exceed_budget() is referentially transparent — same inputs → same output
  - Records are append-only; no mutation after creation
  - Thread safety: not required (single-threaded orchestrator use case)
"""
from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

@dataclass
class CostRecord:
    """
    An immutable record of a single cost event for one training cycle.

    Fields:
      cycle_id   — identifies the training cycle this cost belongs to
      component  — cost category: "sft_gpu" | "dpo_gpu" | "synthesis_api" | "storage"
      amount_usd — cost in US dollars (must be >= 0)
      timestamp  — ISO-8601 UTC timestamp of when the cost was recorded
      metadata   — arbitrary JSON-serialisable dict for debugging / auditing
    """
    cycle_id: str
    component: str
    amount_usd: float
    timestamp: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class CostTracker:
    """
    Accumulates and queries cost records for training cycles.

    Purpose:  track per-component costs across a training cycle and enforce
              a configurable budget guard before initiating expensive operations.
    Inputs:
      max_cycle_budget_usd — hard budget ceiling per cycle; used by
                             would_exceed_budget() to gate phase transitions
    Side effects: maintains an in-memory list of CostRecord objects
    """

    def __init__(self, max_cycle_budget_usd: float = 30.0) -> None:
        if max_cycle_budget_usd <= 0:
            raise ValueError(f"max_cycle_budget_usd must be > 0, got {max_cycle_budget_usd}")
        self.max_cycle_budget_usd = max_cycle_budget_usd
        self._records: List[CostRecord] = []

    def record(
        self,
        cycle_id: str,
        component: str,
        amount_usd: float,
        metadata: Optional[dict] = None,
    ) -> CostRecord:
        """
        Append a new cost record for a training cycle.

        Purpose:  persist a cost event for later querying and budget checks.
        Inputs:
          cycle_id    — training cycle identifier
          component   — cost category string (e.g. "sft_gpu", "dpo_gpu")
          amount_usd  — non-negative cost amount in USD
          metadata    — optional dict with additional context (instance type, hours, etc.)
        Outputs:  the created CostRecord
        Complexity: O(1) append
        Side effects: appends to internal list; logs the record
        Raises:   ValueError if amount_usd < 0
        """
        if amount_usd < 0:
            raise ValueError(f"Cost amount must be >= 0, got {amount_usd}")

        rec = CostRecord(
            cycle_id=cycle_id,
            component=component,
            amount_usd=amount_usd,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            metadata=metadata or {},
        )
        self._records.append(rec)

        log.info(
            "cost_recorded",
            cycle_id=cycle_id,
            component=component,
            amount_usd=amount_usd,
        )
        return rec

    def total_for_cycle(self, cycle_id: str) -> float:
        """
        Return the total cost accumulated for a given cycle_id.

        Purpose:  summarise all spending for a training cycle.
        Inputs:   cycle_id — training cycle identifier
        Outputs:  total USD as a float (0.0 if no records exist)
        Complexity: O(N) where N = total records across all cycles
        Side effects: none
        """
        return sum(
            r.amount_usd for r in self._records if r.cycle_id == cycle_id
        )

    def would_exceed_budget(self, cycle_id: str, additional_usd: float) -> bool:
        """
        Check whether adding additional_usd would exceed max_cycle_budget_usd.

        Purpose:  gate expensive operations (e.g. DPO phase start) when the
                  cycle is already at or near its budget limit.
        Inputs:
          cycle_id        — training cycle identifier
          additional_usd  — hypothetical additional cost to check
        Outputs:  True if total_for_cycle(cycle_id) + additional_usd > budget
        Complexity: O(N)
        Side effects: none — purely a query, no state mutation
        """
        current = self.total_for_cycle(cycle_id)
        return (current + additional_usd) > self.max_cycle_budget_usd

    def get_breakdown(self, cycle_id: str) -> Dict[str, float]:
        """
        Return per-component cost totals for a given cycle_id.

        Purpose:  produce a detailed cost breakdown for reporting / invoicing.
        Inputs:   cycle_id — training cycle identifier
        Outputs:  dict mapping component → total_cost_usd
                  (only components with at least one record are included)
        Complexity: O(N)
        Side effects: none
        """
        breakdown: Dict[str, float] = {}
        for r in self._records:
            if r.cycle_id != cycle_id:
                continue
            breakdown[r.component] = breakdown.get(r.component, 0.0) + r.amount_usd
        return breakdown

    def all_records_for_cycle(self, cycle_id: str) -> List[CostRecord]:
        """
        Return all raw CostRecord objects for a given cycle_id.

        Purpose:  enable auditing, export, or downstream aggregation.
        Inputs:   cycle_id — training cycle identifier
        Outputs:  list of CostRecord objects in insertion order
        Complexity: O(N)
        Side effects: none
        """
        return [r for r in self._records if r.cycle_id == cycle_id]
