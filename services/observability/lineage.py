"""
Adapter Lineage DAG with SQLite persistence.

Design decisions:
  - aiosqlite gives us async I/O without blocking the event loop.
  - Each public method opens and closes its own connection; connection-per-
    operation keeps the implementation thread-safe across concurrent coroutines
    without a connection pool.  SQLite WAL mode would be the next step if
    write concurrency becomes a bottleneck.
  - Ancestor traversal is done in Python (recursive CTE is not available in
    all SQLite versions shipped with macOS), bounded by max_depth to prevent
    infinite loops in the event of a data cycle.
  - JSON columns store list/dict payloads; we serialise on write and
    deserialise on read so callers always see native Python types.

Invariants:
  - adapter_id is the primary key; upsert (INSERT OR REPLACE) is idempotent.
  - parent_adapter_id forms a forest (no cycles enforced by convention and
    bounded walk).
  - target_failure_cluster_ids and evaluation_scores are always deserialised
    before returning a LineageNode (never raw JSON strings).
"""

import aiosqlite
import json
from dataclasses import dataclass, asdict
from typing import Optional, List
import asyncio


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class LineageNode:
    """
    A single vertex in the adapter lineage DAG.

    Fields:
        adapter_id: Unique adapter identifier (PK).
        version: Semantic version string (e.g. "1.3.0").
        base_model_id: Foundation model the LoRA adapter sits on.
        training_cycle_id: The training cycle that produced this adapter.
        created_at: ISO-8601 creation timestamp.
        status: One of AdapterStatus enum values (stored as string).
        parent_adapter_id: Parent adapter or None if root.
        target_failure_cluster_ids: Failure clusters targeted in this cycle.
        evaluation_scores: Benchmark name → score mapping.
        deployment_record: Optional dict with promotion metadata.
    """

    adapter_id: str
    version: str
    base_model_id: str
    training_cycle_id: str
    created_at: str
    status: str
    parent_adapter_id: Optional[str]
    target_failure_cluster_ids: List[int]
    evaluation_scores: dict
    deployment_record: Optional[dict]


@dataclass
class LineageEdge:
    """
    A directed edge parent → child in the lineage DAG.

    Fields:
        parent_id: Parent adapter_id.
        child_id: Child adapter_id.
        improvement_target: Human-readable description of the failure cluster.
        cost_usd: GPU compute cost that produced the child.
        created_at: ISO-8601 edge creation timestamp.
    """

    parent_id: str
    child_id: str
    improvement_target: str
    cost_usd: float
    created_at: str


# ---------------------------------------------------------------------------
# Storage layer
# ---------------------------------------------------------------------------


class LineageStore:
    """
    Async SQLite-backed store for the adapter lineage DAG.

    Usage:
        store = LineageStore("/path/to/lineage.db")
        await store.initialize()   # must be called once before any other op
        await store.add_node(node)
        node = await store.get_node("adapter-v2")
        ancestors = await store.get_ancestors("adapter-v3")

    Thread / concurrency safety:
        Each method is an independent coroutine that opens its own aiosqlite
        connection.  Concurrent reads are safe; concurrent writes are
        serialised by SQLite's file-level write lock.
    """

    def __init__(self, db_path: str = "/tmp/lineage.db"):
        """
        Args:
            db_path: Filesystem path for the SQLite database.  Defaults to
                     /tmp/lineage.db which is fine for single-node deployments.
                     For production, point at a persistent volume.
        """
        self.db_path = db_path
        self._initialized = False

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    async def initialize(self):
        """
        Create tables if they do not exist.  Idempotent — safe to call on
        every startup.  Must complete before any other operation.

        Side effects: writes schema DDL to the SQLite file at db_path.
        Complexity: O(1).
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Nodes table — one row per adapter version
            await db.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    adapter_id                TEXT PRIMARY KEY,
                    version                   TEXT NOT NULL,
                    base_model_id             TEXT NOT NULL,
                    training_cycle_id         TEXT NOT NULL,
                    created_at                TEXT NOT NULL,
                    status                    TEXT NOT NULL,
                    parent_adapter_id         TEXT,
                    target_failure_cluster_ids TEXT NOT NULL,  -- JSON array
                    evaluation_scores          TEXT NOT NULL,  -- JSON object
                    deployment_record          TEXT            -- JSON object, nullable
                )
            """)
            # Edges table — one row per parent→child relationship
            await db.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    parent_id          TEXT NOT NULL,
                    child_id           TEXT NOT NULL,
                    improvement_target TEXT NOT NULL,
                    cost_usd           REAL NOT NULL,
                    created_at         TEXT NOT NULL,
                    PRIMARY KEY (parent_id, child_id)
                )
            """)
            await db.commit()
        self._initialized = True

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def add_node(self, node: LineageNode):
        """
        Upsert a LineageNode into the store.

        INSERT OR REPLACE semantics: if adapter_id already exists the row is
        replaced atomically, enabling status updates without a separate UPDATE
        path.

        Args:
            node: LineageNode to persist.
        Side effects: writes to SQLite.
        Complexity: O(1).
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO nodes
                    (adapter_id, version, base_model_id, training_cycle_id,
                     created_at, status, parent_adapter_id,
                     target_failure_cluster_ids, evaluation_scores,
                     deployment_record)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    node.adapter_id,
                    node.version,
                    node.base_model_id,
                    node.training_cycle_id,
                    node.created_at,
                    node.status,
                    node.parent_adapter_id,
                    json.dumps(node.target_failure_cluster_ids),
                    json.dumps(node.evaluation_scores),
                    json.dumps(node.deployment_record)
                    if node.deployment_record is not None
                    else None,
                ),
            )
            await db.commit()

    async def add_edge(self, edge: LineageEdge):
        """
        Upsert a directed edge into the lineage graph.

        Args:
            edge: LineageEdge describing the parent→child relationship.
        Side effects: writes to SQLite.
        Complexity: O(1).
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO edges
                    (parent_id, child_id, improvement_target, cost_usd, created_at)
                VALUES (?,?,?,?,?)
                """,
                (
                    edge.parent_id,
                    edge.child_id,
                    edge.improvement_target,
                    edge.cost_usd,
                    edge.created_at,
                ),
            )
            await db.commit()

    async def update_status(self, adapter_id: str, status: str):
        """
        Update the status column of an existing node.

        Args:
            adapter_id: Target adapter.
            status:     New status string (e.g. "PRODUCTION", "ROLLED_BACK").
        Side effects: writes to SQLite.  No-op if adapter_id is not found.
        Complexity: O(1).
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE nodes SET status = ? WHERE adapter_id = ?",
                (status, adapter_id),
            )
            await db.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_node(row) -> LineageNode:
        """
        Deserialise a SQLite row tuple into a LineageNode.

        JSON columns are decoded back to native Python types so callers
        never touch raw JSON strings.

        Args:
            row: aiosqlite row (tuple indexed 0–9 matching the column order).
        Returns:
            LineageNode with all fields populated.
        Complexity: O(k) where k is the number of JSON-encoded fields.
        Side effects: None.
        """
        (
            adapter_id,
            version,
            base_model_id,
            training_cycle_id,
            created_at,
            status,
            parent_adapter_id,
            target_failure_cluster_ids_json,
            evaluation_scores_json,
            deployment_record_json,
        ) = row

        return LineageNode(
            adapter_id=adapter_id,
            version=version,
            base_model_id=base_model_id,
            training_cycle_id=training_cycle_id,
            created_at=created_at,
            status=status,
            parent_adapter_id=parent_adapter_id,
            target_failure_cluster_ids=json.loads(target_failure_cluster_ids_json),
            evaluation_scores=json.loads(evaluation_scores_json),
            deployment_record=json.loads(deployment_record_json)
            if deployment_record_json is not None
            else None,
        )

    async def get_node(self, adapter_id: str) -> Optional[LineageNode]:
        """
        Fetch a single LineageNode by its adapter_id.

        Args:
            adapter_id: Primary key to look up.
        Returns:
            LineageNode if found, None otherwise.
        Complexity: O(1) — PK lookup.
        Side effects: None (read-only).
        """
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM nodes WHERE adapter_id = ?", (adapter_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_node(row)

    async def get_ancestors(
        self, adapter_id: str, max_depth: int = 10
    ) -> List[LineageNode]:
        """
        Walk the parent chain from adapter_id up to the root, bounded by
        max_depth to prevent infinite loops in degenerate data.

        Returns an ordered list [root, …, adapter_id] i.e. the root is first
        and the requested node is last.

        Args:
            adapter_id: Starting node (included in result).
            max_depth:  Maximum number of hops to follow (default 10).
        Returns:
            Ordered list of LineageNodes from root to adapter_id.
            If adapter_id does not exist returns [].
        Complexity: O(d) queries where d ≤ max_depth.
        Side effects: None (read-only).
        """
        chain: List[LineageNode] = []
        current_id: Optional[str] = adapter_id
        visited: set = set()

        async with aiosqlite.connect(self.db_path) as db:
            for _ in range(max_depth + 1):
                if current_id is None or current_id in visited:
                    break
                visited.add(current_id)

                async with db.execute(
                    "SELECT * FROM nodes WHERE adapter_id = ?", (current_id,)
                ) as cursor:
                    row = await cursor.fetchone()

                if row is None:
                    break

                node = self._row_to_node(row)
                chain.append(node)
                current_id = node.parent_adapter_id

        # chain is currently [adapter_id, …, root]; reverse to [root, …, adapter_id]
        chain.reverse()
        return chain

    async def get_children(self, adapter_id: str) -> List[LineageNode]:
        """
        Return all direct children of adapter_id.

        Args:
            adapter_id: Parent adapter whose children to retrieve.
        Returns:
            List of LineageNodes whose parent_adapter_id == adapter_id.
            Empty list if none.
        Complexity: O(k) where k is the number of children.
        Side effects: None (read-only).
        """
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT * FROM nodes WHERE parent_adapter_id = ?", (adapter_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_node(row) for row in rows]

    async def get_all_nodes(
        self, status_filter: Optional[str] = None
    ) -> List[LineageNode]:
        """
        Retrieve all nodes, optionally filtered by status.

        Args:
            status_filter: If provided, only nodes with this status are
                           returned.  None returns all nodes.
        Returns:
            List of LineageNodes ordered by created_at ascending.
        Complexity: O(n) full table scan.  For large lineage graphs an index
                    on status + created_at would be warranted.
        Side effects: None (read-only).
        """
        async with aiosqlite.connect(self.db_path) as db:
            if status_filter is not None:
                async with db.execute(
                    "SELECT * FROM nodes WHERE status = ? ORDER BY created_at ASC",
                    (status_filter,),
                ) as cursor:
                    rows = await cursor.fetchall()
            else:
                async with db.execute(
                    "SELECT * FROM nodes ORDER BY created_at ASC"
                ) as cursor:
                    rows = await cursor.fetchall()

            return [self._row_to_node(row) for row in rows]
