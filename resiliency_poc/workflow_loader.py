"""
workflow_loader.py

Parses n8n workflow JSONs into a WorkflowGraph used by the trace generator.
Extracts node types, adjacency, BFS depth, position, and path enumeration.
Works with both the email responder and the SeaTable webhook validator.
"""

from __future__ import annotations
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── Unified node-type vocabulary ──────────────────────────────────────────────

NODE_TYPE_VOCAB: Dict[str, int] = {
    "gmailTrigger":       0,
    "textClassifier":     1,
    "lmChatGoogleGemini": 2,
    "emailSend":          3,
    "gmail":              4,
    "if":                 5,
    "googleCalendar":     6,
    "splitInBatches":     7,
    "code":               8,
    "webhook":            9,
    "crypto":             10,
    "respondToWebhook":   11,
    "noOp":               12,
    "unknown":            12,   # fallback
}
NUM_NODE_TYPES = 13

# Node types that call external services (higher failure surface)
EXTERNAL_NODE_TYPES: Set[str] = {
    "gmailTrigger", "gmail", "emailSend",
    "googleCalendar", "lmChatGoogleGemini",
    "webhook", "respondToWebhook",
}

# Node types where crypto_error is plausible
CRYPTO_NODE_TYPES: Set[str] = {"crypto", "webhook"}

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class NodeInfo:
    name:        str
    type_name:   str
    type_idx:    int
    is_external: bool
    is_crypto:   bool
    depth:       int   = 0    # BFS depth from entry
    position:    float = 0.0  # normalised position in traversal order [0,1]


@dataclass
class WorkflowGraph:
    name:           str
    nodes:          Dict[str, NodeInfo]         # name → NodeInfo
    adj:            Dict[str, List[str]]        # name → [successor names]
    rev_adj:        Dict[str, List[str]]        # name → [predecessor names]
    entry_nodes:    List[str]
    terminal_nodes: List[str]
    all_paths:      List[List[str]] = field(default_factory=list)

    # ── Convenience ──────────────────────────────────────────────────────────

    def depth_norm(self, name: str) -> float:
        max_d = max((n.depth for n in self.nodes.values()), default=1)
        return self.nodes[name].depth / max(max_d, 1)

    def path_count(self) -> int:
        return len(self.all_paths)


# ── Parser ────────────────────────────────────────────────────────────────────

def _type_name(full_type: str) -> str:
    """'n8n-nodes-base.gmailTrigger'  →  'gmailTrigger'"""
    return full_type.split(".")[-1]


def _build_adj(connections: dict, node_names: Set[str]) -> Tuple[Dict, Dict]:
    adj: Dict[str, List[str]] = {n: [] for n in node_names}
    rev: Dict[str, List[str]] = {n: [] for n in node_names}
    for src, conn_map in connections.items():
        for output_group in conn_map.get("main", []):
            for target_info in output_group:
                tgt = target_info["node"]
                if tgt in node_names and src in adj:
                    if tgt not in adj[src]:
                        adj[src].append(tgt)
                    if src not in rev[tgt]:
                        rev[tgt].append(src)
    return adj, rev


def _bfs_depth(adj: Dict[str, List[str]], entries: List[str]) -> Dict[str, int]:
    depth: Dict[str, int] = {}
    q: deque = deque()
    for e in entries:
        depth[e] = 0
        q.append(e)
    while q:
        node = q.popleft()
        for child in adj.get(node, []):
            if child not in depth:
                depth[child] = depth[node] + 1
                q.append(child)
    return depth


def _enumerate_paths(
    adj: Dict[str, List[str]],
    entries: List[str],
    terminals: Set[str],
    max_depth: int = 20,
    max_paths: int = 200,
) -> List[List[str]]:
    paths: List[List[str]] = []

    def dfs(node: str, path: List[str], visited: Set[str]) -> None:
        if len(paths) >= max_paths:
            return
        is_terminal = node in terminals or not adj.get(node)
        if is_terminal or len(path) >= max_depth:
            paths.append(path[:])
            return
        for child in adj.get(node, []):
            if child not in visited:
                visited.add(child)
                path.append(child)
                dfs(child, path, visited)
                path.pop()
                visited.remove(child)

    for entry in entries:
        dfs(entry, [entry], {entry})

    return paths if paths else [[e] for e in entries]


def load_workflow(json_path: str | Path) -> WorkflowGraph:
    """Load a single n8n workflow JSON and return a WorkflowGraph."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))

    # Filter out non-executable nodes (stickyNote, etc.)
    skip_types = {"stickyNote"}
    raw_nodes = [
        n for n in data.get("nodes", [])
        if _type_name(n.get("type", "")) not in skip_types
    ]

    node_names: Set[str] = {n["name"] for n in raw_nodes}

    # Build initial node map (depth and position filled later)
    nodes: Dict[str, NodeInfo] = {}
    for raw in raw_nodes:
        tname = _type_name(raw.get("type", "unknown"))
        nodes[raw["name"]] = NodeInfo(
            name=raw["name"],
            type_name=tname,
            type_idx=NODE_TYPE_VOCAB.get(tname, NODE_TYPE_VOCAB["unknown"]),
            is_external=tname in EXTERNAL_NODE_TYPES,
            is_crypto=tname in CRYPTO_NODE_TYPES,
        )

    adj, rev = _build_adj(data.get("connections", {}), node_names)

    # Entry nodes: no predecessors
    entry_nodes = [n for n in node_names if not rev.get(n)]
    if not entry_nodes:
        entry_nodes = [raw_nodes[0]["name"]]

    # Terminal nodes: no successors
    terminal_nodes = [n for n in node_names if not adj.get(n)]
    terminal_set = set(terminal_nodes)

    # BFS depth from entries
    depth_map = _bfs_depth(adj, entry_nodes)
    for name, info in nodes.items():
        info.depth = depth_map.get(name, 0)

    # Normalised position: BFS order index / total
    bfs_order: List[str] = []
    seen: Set[str] = set()
    q: deque = deque(entry_nodes)
    for e in entry_nodes:
        seen.add(e)
    while q:
        n = q.popleft()
        bfs_order.append(n)
        for child in adj.get(n, []):
            if child not in seen:
                seen.add(child)
                q.append(child)
    total = max(len(bfs_order) - 1, 1)
    for idx, name in enumerate(bfs_order):
        nodes[name].position = idx / total

    paths = _enumerate_paths(adj, entry_nodes, terminal_set)

    return WorkflowGraph(
        name=data.get("name", Path(json_path).stem),
        nodes=nodes,
        adj=adj,
        rev_adj=rev,
        entry_nodes=entry_nodes,
        terminal_nodes=terminal_nodes,
        all_paths=paths,
    )


def load_both_workflows(
    email_path: Optional[str | Path] = None,
    webhook_path: Optional[str | Path] = None,
) -> List[WorkflowGraph]:
    """
    Load the two reference workflows.  Paths default to the locations used in
    the project; override if you move the files.
    """
    base = Path(__file__).resolve().parent.parent

    if email_path is None:
        email_path = (
            base.parent.parent
            / "projectwork" / "workflowgeneration" / "workflow1"
            / "smart_email_responder_workflow_using_ai.json"
        )
    if webhook_path is None:
        webhook_path = (
            base / "workflow_10"
            / "validate_seatable_webhooks_with_hmac_sha256_authentication.json"
        )

    graphs = []
    for p in (email_path, webhook_path):
        p = Path(p)
        if p.exists():
            graphs.append(load_workflow(p))
        else:
            print(f"[workflow_loader] WARNING: file not found → {p}")
    return graphs


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    graphs = load_both_workflows()
    for g in graphs:
        print(f"\n{'='*60}")
        print(f"Workflow : {g.name}")
        print(f"Nodes    : {len(g.nodes)}")
        print(f"Edges    : {sum(len(v) for v in g.adj.values())}")
        print(f"Entry    : {g.entry_nodes}")
        print(f"Terminal : {g.terminal_nodes}")
        print(f"Paths    : {g.path_count()}")
        for name, info in g.nodes.items():
            print(f"  {name:35s} type={info.type_name:25s} "
                  f"depth={info.depth} pos={info.position:.2f} "
                  f"ext={info.is_external}")
