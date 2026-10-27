"""
dsb_workflow_loader.py

Builds WorkflowGraph objects from the real DeathStarBench socialNetwork
microservice call graphs — the same services that appear in the X-Trace dataset.

Each DSB workload type (compose_post, read_home_timeline, etc.) has a known
call DAG. We hard-code these topologies from the DSB paper and source code.

Nine workload graphs are provided:
  1. compose_post          (healthy)  — 8 services, fan-out write
  2. read_home_timeline    (healthy)  — 4 services, simple read
  3. read_user_timeline    (healthy)  — 4 services, simple read
  4. user_follow           (healthy)  — 4 services, social graph update
  5. user_register         (healthy)  — 3 services, user creation
  6. compose_broken        (DOWN)     — same topology as compose_post, service down
  7. read_home_broken      (TRANSIENT)— same as read_home_timeline, broken pipe
  8. read_user_broken      (TRANSIENT)— same as read_user_timeline, broken pipe
  9. register_broken       (DEGRADED) — same as user_register, degraded crypto

Each node is a DSB microservice.
is_external = True for services that call downstream services or external stores.
type_name   = DSB service name (maps to DSB_SERVICE_VOCAB in dsb_loader.py).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .workflow_loader import WorkflowGraph, NodeInfo

# ── DSB service vocabulary — mirrors dsb_loader.py ────────────────────────────

DSB_NODE_VOCAB: Dict[str, int] = {
    "nginx-thrift":             0,
    "UserService":              1,
    "ComposePostService":       2,
    "MediaService":             3,
    "UniqueIdService":          4,
    "HomeTimelineService":      5,
    "PostStorageService":       6,
    "SocialGraphService":       7,
    "WriteHomeTimelineService": 8,
    "TextService":              9,
    "UserMentionService":      10,
    "UserTimelineService":     11,
    "UrlShortenService":       12,
}

# Services that call external stores / downstream services
DSB_EXTERNAL: Set[str] = {
    "nginx-thrift",
    "HomeTimelineService",
    "UserTimelineService",
    "PostStorageService",
    "WriteHomeTimelineService",
    "SocialGraphService",
}

# Services with crypto / auth operations
DSB_CRYPTO: Set[str] = {
    "UserService",
    "nginx-thrift",
}


# ── Graph builder helpers ──────────────────────────────────────────────────────

def _build_graph(
    name:  str,
    edges: List[Tuple[str, str]],   # (caller, callee)
) -> WorkflowGraph:
    """
    Build a WorkflowGraph from a list of directed edges.
    Entry = no incoming edges.  Terminal = no outgoing edges.
    """
    # Collect all nodes
    all_nodes: Set[str] = set()
    for src, dst in edges:
        all_nodes.add(src)
        all_nodes.add(dst)

    adj:     Dict[str, List[str]] = {n: [] for n in all_nodes}
    rev_adj: Dict[str, List[str]] = {n: [] for n in all_nodes}

    for src, dst in edges:
        if dst not in adj[src]:
            adj[src].append(dst)
        if src not in rev_adj[dst]:
            rev_adj[dst].append(src)

    entry_nodes    = [n for n in all_nodes if not rev_adj[n]]
    terminal_nodes = [n for n in all_nodes if not adj[n]]
    if not entry_nodes:
        entry_nodes = [next(iter(all_nodes))]

    # BFS depth
    depth_map: Dict[str, int] = {}
    q: deque = deque()
    for e in entry_nodes:
        depth_map[e] = 0
        q.append(e)
    while q:
        node = q.popleft()
        for child in adj[node]:
            if child not in depth_map:
                depth_map[child] = depth_map[node] + 1
                q.append(child)

    # BFS order for position normalisation
    bfs_order: List[str] = []
    seen: Set[str] = set(entry_nodes)
    q2: deque = deque(entry_nodes)
    while q2:
        n = q2.popleft()
        bfs_order.append(n)
        for child in adj[n]:
            if child not in seen:
                seen.add(child)
                q2.append(child)
    total = max(len(bfs_order) - 1, 1)

    nodes: Dict[str, NodeInfo] = {}
    for idx, name_n in enumerate(bfs_order):
        t = name_n   # type_name = service name for DSB
        nodes[name_n] = NodeInfo(
            name=name_n,
            type_name=t,
            type_idx=DSB_NODE_VOCAB.get(t, len(DSB_NODE_VOCAB) - 1),
            is_external=name_n in DSB_EXTERNAL,
            is_crypto=name_n in DSB_CRYPTO,
            depth=depth_map.get(name_n, 0),
            position=idx / total,
        )

    # Simple path enumeration (DFS, max 200 paths)
    terminal_set = set(terminal_nodes)
    paths: List[List[str]] = []

    def dfs(node: str, path: List[str], visited: Set[str]) -> None:
        if len(paths) >= 200:
            return
        if node in terminal_set or not adj[node]:
            paths.append(path[:])
            return
        for child in adj[node]:
            if child not in visited:
                visited.add(child)
                path.append(child)
                dfs(child, path, visited)
                path.pop()
                visited.remove(child)

    for entry in entry_nodes:
        dfs(entry, [entry], {entry})
    if not paths:
        paths = [[e] for e in entry_nodes]

    return WorkflowGraph(
        name=name,
        nodes=nodes,
        adj=adj,
        rev_adj=rev_adj,
        entry_nodes=entry_nodes,
        terminal_nodes=terminal_nodes,
        all_paths=paths,
    )


# ── DSB call graph topologies ─────────────────────────────────────────────────
#
# Derived from DSB paper (ASPLOS 2019) Figure 3 and the DSB source code at
# github.com/delimitrou/DeathStarBench/socialNetwork/.
#
# nginx-thrift is the API gateway (entry for all workloads).

def _compose_post_graph() -> WorkflowGraph:
    """
    Compose Post: fan-out write to 8 services.
    nginx -> ComposePostService -> {UniqueId, User, Text, Media, UserMention, Url}
                                -> PostStorage
                                -> WriteHomeTimeline
    """
    edges = [
        ("nginx-thrift",        "ComposePostService"),
        ("ComposePostService",  "UniqueIdService"),
        ("ComposePostService",  "UserService"),
        ("ComposePostService",  "TextService"),
        ("ComposePostService",  "MediaService"),
        ("ComposePostService",  "UserMentionService"),
        ("ComposePostService",  "UrlShortenService"),
        ("ComposePostService",  "PostStorageService"),
        ("ComposePostService",  "WriteHomeTimelineService"),
    ]
    return _build_graph("DSB/compose_post (HEALTHY)", edges)


def _read_home_timeline_graph() -> WorkflowGraph:
    """
    Read Home Timeline:
    nginx -> HomeTimelineService -> PostStorageService -> UserService
    """
    edges = [
        ("nginx-thrift",       "HomeTimelineService"),
        ("HomeTimelineService","PostStorageService"),
        ("PostStorageService", "UserService"),
    ]
    return _build_graph("DSB/read_home_timeline (HEALTHY)", edges)


def _read_user_timeline_graph() -> WorkflowGraph:
    """
    Read User Timeline:
    nginx -> UserTimelineService -> PostStorageService -> UserService
    """
    edges = [
        ("nginx-thrift",        "UserTimelineService"),
        ("UserTimelineService", "PostStorageService"),
        ("PostStorageService",  "UserService"),
    ]
    return _build_graph("DSB/read_user_timeline (HEALTHY)", edges)


def _user_follow_graph() -> WorkflowGraph:
    """
    User Follow:
    nginx -> UserService -> SocialGraphService -> HomeTimelineService
    """
    edges = [
        ("nginx-thrift",      "UserService"),
        ("UserService",       "SocialGraphService"),
        ("SocialGraphService","HomeTimelineService"),
    ]
    return _build_graph("DSB/user_follow (HEALTHY)", edges)


def _user_register_graph() -> WorkflowGraph:
    """
    User Register:
    nginx -> UserService -> SocialGraphService
    """
    edges = [
        ("nginx-thrift", "UserService"),
        ("UserService",  "SocialGraphService"),
    ]
    return _build_graph("DSB/user_register (HEALTHY)", edges)


def _compose_broken_graph() -> WorkflowGraph:
    """
    Compose Broken (SERVICE_DOWN):
    Same topology as compose_post — PostStorageService is down.
    The hardening engine assigns risk based on node type, not scenario label.
    """
    edges = [
        ("nginx-thrift",        "ComposePostService"),
        ("ComposePostService",  "UniqueIdService"),
        ("ComposePostService",  "UserService"),
        ("ComposePostService",  "TextService"),
        ("ComposePostService",  "MediaService"),
        ("ComposePostService",  "UserMentionService"),
        ("ComposePostService",  "UrlShortenService"),
        ("ComposePostService",  "PostStorageService"),
        ("ComposePostService",  "WriteHomeTimelineService"),
    ]
    return _build_graph("DSB/compose_broken (DOWN)", edges)


def _read_home_broken_graph() -> WorkflowGraph:
    """
    Read Home Timeline Broken Pipe (TRANSIENT):
    Same topology — HomeTimelineService has broken pipe failures.
    """
    edges = [
        ("nginx-thrift",       "HomeTimelineService"),
        ("HomeTimelineService","PostStorageService"),
        ("PostStorageService", "UserService"),
    ]
    return _build_graph("DSB/read_home_broken_pipe (TRANSIENT)", edges)


def _read_user_broken_graph() -> WorkflowGraph:
    """
    Read User Timeline Broken Pipe (TRANSIENT):
    Same topology — UserTimelineService has broken pipe failures.
    """
    edges = [
        ("nginx-thrift",        "UserTimelineService"),
        ("UserTimelineService", "PostStorageService"),
        ("PostStorageService",  "UserService"),
    ]
    return _build_graph("DSB/read_user_broken_pipe (TRANSIENT)", edges)


def _register_broken_graph() -> WorkflowGraph:
    """
    Register Broken (DEGRADED):
    Same topology — UserService is degraded (crypto failures).
    """
    edges = [
        ("nginx-thrift", "UserService"),
        ("UserService",  "SocialGraphService"),
    ]
    return _build_graph("DSB/register_broken (DEGRADED)", edges)


# ── Public API ─────────────────────────────────────────────────────────────────

def load_dsb_workflows() -> List[WorkflowGraph]:
    """
    Return all DSB workload graphs.
    These represent real DeathStarBench socialNetwork call topologies.
    """
    return [
        _compose_post_graph(),
        _read_home_timeline_graph(),
        _read_user_timeline_graph(),
        _user_follow_graph(),
        _user_register_graph(),
        _compose_broken_graph(),
        _read_home_broken_graph(),
        _read_user_broken_graph(),
        _register_broken_graph(),
    ]


def load_dsb_failure_workflows() -> List[WorkflowGraph]:
    """
    Return only the failure-scenario DSB workload graphs —
    the ones that appear as broken/degraded in the X-Trace dataset.
    Useful for hardening experiments focused on failure conditions.
    """
    return [
        _compose_broken_graph(),
        _read_home_broken_graph(),
        _read_user_broken_graph(),
        _register_broken_graph(),
    ]


def load_dsb_healthy_and_failure() -> List[WorkflowGraph]:
    """
    Return two representative graphs for comparison:
      - compose_post  (large fan-out, healthy topology)
      - compose_broken (same topology under SERVICE_DOWN)
    Good for a focused hardening demo showing the same topology under different conditions.
    """
    return [
        _compose_post_graph(),
        _read_home_timeline_graph(),
        _compose_broken_graph(),
        _read_home_broken_graph(),
    ]


if __name__ == "__main__":
    graphs = load_dsb_workflows()
    for g in graphs:
        print(f"\n{'='*60}")
        print(f"Workflow : {g.name}")
        print(f"Nodes    : {len(g.nodes)}")
        print(f"Entry    : {g.entry_nodes}")
        print(f"Terminal : {g.terminal_nodes}")
        for name, info in g.nodes.items():
            print(f"  {name:<30} depth={info.depth}  ext={info.is_external}  crypto={info.is_crypto}")
