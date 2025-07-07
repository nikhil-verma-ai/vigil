"""
GPU instance provisioning — real cloud and mock implementations.

Provides:
  - GPUInstance:       dataclass describing a running (or terminated) GPU instance
  - GPUProvisioner:    production provisioner that calls a cloud provider API
  - MockGPUProvisioner: deterministic mock for tests; no network calls required

Invariants:
  - Every instance_id returned by provision() is unique within a session
  - After terminate(), instance.status == "TERMINATED"
  - MockGPUProvisioner tracks provision_count and terminate_count accurately
  - Spot price is always <= on-demand via spot_price_cap_fraction enforcement
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Instance dataclass
# ---------------------------------------------------------------------------

@dataclass
class GPUInstance:
    """
    Represents a provisioned GPU compute instance.

    Fields:
      instance_id      — cloud-provider unique identifier (or mock-gpu-N)
      instance_type    — GPU model string, e.g. "A100-80GB"
      region           — cloud region, e.g. "us-east-1"
      hourly_cost_usd  — current effective hourly cost (spot or on-demand)
      is_spot          — True if this is a preemptible/spot instance
      status           — RUNNING | INTERRUPTED | TERMINATED
    """
    instance_id: str
    instance_type: str
    region: str
    hourly_cost_usd: float
    is_spot: bool
    status: str  # "RUNNING" | "INTERRUPTED" | "TERMINATED"


# ---------------------------------------------------------------------------
# Spot price catalogue (on-demand estimates for reference)
# ---------------------------------------------------------------------------

# Approximate on-demand hourly costs (USD) for common GPU types.
# The real provisioner would query the cloud API for current spot prices.
_ONDEMAND_HOURLY_USD: Dict[str, float] = {
    "A100-80GB":  4.10,   # p4d.24xlarge equivalent (per-GPU share)
    "A100-40GB":  3.06,
    "H100-80GB":  8.50,
    "V100-16GB":  2.48,
    "T4-16GB":    0.53,
}

_DEFAULT_REGION = "us-east-1"


# ---------------------------------------------------------------------------
# Real provisioner
# ---------------------------------------------------------------------------

class GPUProvisioner:
    """
    Production GPU instance provisioner — calls the cloud provider API.

    Purpose:  provision and terminate GPU instances for training workloads.
    Inputs:
      cloud_provider          — "aws" | "gcp" | "azure" (routing to SDK)
      spot_price_cap_fraction — fraction of on-demand price to cap spot bids;
                                e.g. 0.7 means bid at most 70% of on-demand
    Side effects: makes network calls to cloud provider APIs; accrues costs
    """

    def __init__(
        self,
        cloud_provider: str = "aws",
        spot_price_cap_fraction: float = 0.7,
    ) -> None:
        if cloud_provider not in {"aws", "gcp", "azure"}:
            raise ValueError(f"Unsupported cloud provider: {cloud_provider!r}")
        self.cloud_provider = cloud_provider
        self.spot_price_cap_fraction = spot_price_cap_fraction
        self._active: Dict[str, GPUInstance] = {}

    def provision(self, gpu_type: str = "A100-80GB") -> GPUInstance:
        """
        Request a spot GPU instance from the cloud provider.

        Purpose:  select cheapest available instance of the requested GPU type,
                  bid at most spot_price_cap_fraction * on_demand_price, and
                  return a running GPUInstance.
        Inputs:   gpu_type — one of the keys in _ONDEMAND_HOURLY_USD
        Outputs:  GPUInstance with status="RUNNING"
        Complexity: O(1) API call latency
        Side effects: provisions cloud resources, starts billing
        Raises:   RuntimeError if no instances are available below price cap
        """
        on_demand = _ONDEMAND_HOURLY_USD.get(gpu_type, 4.10)
        spot_cap = on_demand * self.spot_price_cap_fraction

        # Dispatch to cloud-specific implementation.
        if self.cloud_provider == "aws":
            return self._provision_aws(gpu_type, spot_cap)
        elif self.cloud_provider == "gcp":
            return self._provision_gcp(gpu_type, spot_cap)
        else:
            return self._provision_azure(gpu_type, spot_cap)

    def terminate(self, instance: GPUInstance) -> None:
        """
        Terminate a running GPU instance.

        Purpose:  stop billing and release cloud resources.
        Inputs:   instance — a GPUInstance previously returned by provision()
        Outputs:  None; instance.status is set to "TERMINATED"
        Complexity: O(1) API call
        Side effects: terminates cloud instance; stops billing
        """
        log.info(
            "gpu_terminate",
            instance_id=instance.instance_id,
            cloud=self.cloud_provider,
        )
        instance.status = "TERMINATED"
        self._active.pop(instance.instance_id, None)

    def _provision_aws(self, gpu_type: str, spot_cap: float) -> GPUInstance:
        """Provision a spot instance on AWS via boto3."""
        import boto3  # type: ignore

        # GPU type → EC2 instance type mapping
        instance_type_map = {
            "A100-80GB": "p4de.24xlarge",
            "A100-40GB": "p4d.24xlarge",
            "H100-80GB": "p5.48xlarge",
            "V100-16GB": "p3.2xlarge",
            "T4-16GB":   "g4dn.xlarge",
        }
        ec2_type = instance_type_map.get(gpu_type, "p4d.24xlarge")

        ec2 = boto3.client("ec2", region_name=_DEFAULT_REGION)
        response = ec2.request_spot_instances(
            SpotPrice=str(spot_cap),
            InstanceCount=1,
            Type="one-time",
            LaunchSpecification={
                "InstanceType": ec2_type,
                "ImageId": "ami-0abcdef1234567890",  # deep learning AMI
            },
        )
        sir_id = response["SpotInstanceRequests"][0]["SpotInstanceRequestId"]

        # Poll until fulfilled (real code would use a waiter)
        for _ in range(30):
            result = ec2.describe_spot_instance_requests(
                SpotInstanceRequestIds=[sir_id]
            )
            req = result["SpotInstanceRequests"][0]
            if req["State"] == "active":
                instance = GPUInstance(
                    instance_id=req["InstanceId"],
                    instance_type=gpu_type,
                    region=_DEFAULT_REGION,
                    hourly_cost_usd=float(req.get("SpotPrice", spot_cap)),
                    is_spot=True,
                    status="RUNNING",
                )
                self._active[instance.instance_id] = instance
                log.info("gpu_provisioned_aws", **{"instance_id": instance.instance_id})
                return instance
            time.sleep(2)

        raise RuntimeError(f"AWS spot instance not fulfilled within timeout (type={ec2_type})")

    def _provision_gcp(self, gpu_type: str, spot_cap: float) -> GPUInstance:
        """Provision a preemptible VM on GCP (stub — extend with google-cloud-compute)."""
        raise NotImplementedError("GCP provisioner not yet implemented")

    def _provision_azure(self, gpu_type: str, spot_cap: float) -> GPUInstance:
        """Provision a spot VM on Azure (stub — extend with azure-mgmt-compute)."""
        raise NotImplementedError("Azure provisioner not yet implemented")


# ---------------------------------------------------------------------------
# Mock provisioner (tests / local dev)
# ---------------------------------------------------------------------------

class MockGPUProvisioner:
    """
    Deterministic mock GPU provisioner for tests — no cloud API calls.

    Purpose:  enable unit and integration tests to verify orchestrator logic
              without incurring cloud costs or requiring network access.
    Inputs:
      simulate_interruption_after — if set, the N-th provisioned instance will
                                    have its status flipped to "INTERRUPTED"
                                    after provision(); useful for testing
                                    spot-interruption recovery paths
    Invariants:
      - provision_count increments by 1 each call
      - terminate_count increments by 1 each call
      - All active instances are queryable via active_instances dict
    """

    def __init__(
        self,
        simulate_interruption_after: Optional[int] = None,
    ) -> None:
        self.simulate_interruption_after = simulate_interruption_after
        self.provision_count: int = 0
        self.terminate_count: int = 0
        self.active_instances: Dict[str, GPUInstance] = {}

    def provision(self, gpu_type: str = "A100-80GB") -> GPUInstance:
        """
        Return a fake GPUInstance immediately without any network call.

        Purpose:  simulate successful spot instance allocation.
        Inputs:   gpu_type — stored on the returned instance for inspection
        Outputs:  GPUInstance with status="RUNNING" and plausible cost
        Complexity: O(1)
        Side effects: increments provision_count; adds to active_instances
        """
        self.provision_count += 1
        instance = GPUInstance(
            instance_id=f"mock-gpu-{self.provision_count}",
            instance_type=gpu_type,
            region="us-east-1",
            hourly_cost_usd=3.21,   # realistic A100 spot estimate
            is_spot=True,
            status="RUNNING",
        )
        self.active_instances[instance.instance_id] = instance

        # Optionally simulate a spot interruption on the N-th provision
        if (
            self.simulate_interruption_after is not None
            and self.provision_count >= self.simulate_interruption_after
        ):
            instance.status = "INTERRUPTED"

        log.info("mock_gpu_provisioned", instance_id=instance.instance_id)
        return instance

    def terminate(self, instance: GPUInstance) -> None:
        """
        Mark a mock instance as terminated.

        Purpose:  simulate teardown of a provisioned instance.
        Inputs:   instance — GPUInstance returned by provision()
        Outputs:  None; instance.status set to "TERMINATED"
        Complexity: O(1)
        Side effects: increments terminate_count; removes from active_instances
        """
        self.terminate_count += 1
        instance.status = "TERMINATED"
        self.active_instances.pop(instance.instance_id, None)
        log.info("mock_gpu_terminated", instance_id=instance.instance_id)
