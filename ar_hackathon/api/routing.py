"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the drive_unit_next_move function in this module.

*****IMPORTANT*****
Team name:
Email address:
*******************
"""

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

from ar_hackathon.models.graph_state import GraphState

OCCUPANCY_PENALTY = 3.0
SCORE_SCALE = 50.0
INF = float("inf")

_ASSIGN: Optional[Dict[int, Optional[str]]] = None
_SEEN_SIG: Optional[tuple] = None
_RECOMPUTE_STEP: Optional[int] = None

_TOPOLOGY_FP: Optional[Tuple[tuple, tuple]] = None
_IDX: Dict[int, int] = {}
_ADJ: Dict[int, List[tuple]] = {}
_FW: List[List[float]] = []
_STORAGE: Set[int] = set()

_PLANS: Dict[int, List[int]] = {}
_GOALS: Dict[int, object] = {}


def _fingerprint(state: GraphState) -> Tuple[tuple, tuple]:
    nodes = tuple(sorted((n.id, n.node_type, n.capacity) for n in state.nodes))
    edges = tuple(sorted((e.from_node, e.to_node, e.weight, e.capacity, e.bidirectional) for e in state.edges))
    return nodes, edges


def _ensure_topology(state: GraphState) -> None:
    global _TOPOLOGY_FP, _IDX, _ADJ, _FW, _STORAGE, _ASSIGN, _PLANS, _GOALS
    fp = _fingerprint(state)
    if fp == _TOPOLOGY_FP:
        return
    _PLANS.clear()
    _GOALS.clear()
    _ASSIGN = None
    node_list = sorted(state.nodes, key=lambda n: n.id)
    idx = {n.id: i for i, n in enumerate(node_list)}
    adj = {n.id: [] for n in node_list}
    for e in state.edges:
        adj[e.from_node].append((e.to_node, e.weight, e.capacity, e.bidirectional))
        if e.bidirectional:
            adj[e.to_node].append((e.from_node, e.weight, e.capacity, True))
    n = len(node_list)
    fw = [[0.0 if i == j else INF for j in range(n)] for i in range(n)]
    for e in state.edges:
        a, b = idx[e.from_node], idx[e.to_node]
        if e.weight < fw[a][b]:
            fw[a][b] = e.weight
        if e.bidirectional and e.weight < fw[b][a]:
            fw[b][a] = e.weight
    for k in range(n):
        dk = fw[k]
        for i in range(n):
            dik = fw[i][k]
            if dik >= INF:
                continue
            di = fw[i]
            for j in range(n):
                nd = dik + dk[j]
                if nd < di[j]:
                    di[j] = nd
    _TOPOLOGY_FP = fp
    _IDX = idx
    _ADJ = adj
    _FW = fw
    _STORAGE = {n.id for n in node_list if n.node_type == "storage"}


def _dijkstra(state: GraphState, start: int, goal: int) -> Optional[List[int]]:
    if start == goal:
        return [start]
    occ = {}
    for e in state.edges:
        if e.bidirectional:
            key = (min(e.from_node, e.to_node), max(e.from_node, e.to_node))
        else:
            key = (e.from_node, e.to_node)
        occ[key] = state.edge_occupancy(e.from_node, e.to_node)
    adj = _ADJ
    pq = [(0.0, start)]
    dist = {start: 0.0}
    prev = {}
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, INF):
            continue
        if u == goal:
            break
        for (v, w, cap, bidir) in adj.get(u, ()):
            key = (min(u, v), max(u, v)) if bidir else (u, v)
            cnt = occ.get(key, 0)
            cost = w if cap is None else w * (1.0 + OCCUPANCY_PENALTY * cnt)
            nd = d + cost
            if nd < dist.get(v, INF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if start not in _IDX or goal not in _IDX:
        return None
    if goal not in dist or dist[goal] >= INF:
        return None
    path = [goal]
    u = goal
    while u != start:
        u = prev[u]
        path.append(u)
    path.reverse()
    return path


def _min_hungarian(n: int, m: int, cost: List[List[float]]) -> List[int]:
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    res = [0] * m
    for j in range(1, m + 1):
        res[j - 1] = p[j] - 1 if p[j] != 0 else -1
    return res


def _hungarian_maximize(values: List[List[float]], nr: int, nc: int) -> List[int]:
    if nr == 0 or nc == 0:
        return [-1] * nr
    transpose = nr > nc
    if transpose:
        a = [[values[r][c] for r in range(nr)] for c in range(nc)]
    else:
        a = [list(row) for row in values]
    n, m = len(a), len(a[0])
    cost = [[-a[i][j] for j in range(m)] for i in range(n)]
    match = _min_hungarian(n, m, cost)
    res = [-1] * nr
    if transpose:
        for c in range(m):
            r = match[c]
            if r >= 0:
                res[c] = r
    else:
        for c in range(m):
            r = match[c]
            if r >= 0:
                res[r] = c
    return res


def _pod_reward(state: GraphState, unit_node: int, pod) -> float:
    if unit_node not in _IDX or pod.current_node not in _IDX or pod.destination_station not in _IDX:
        return 0.0
    d1 = _FW[_IDX[unit_node]][_IDX[pod.current_node]]
    d2 = _FW[_IDX[pod.current_node]][_IDX[pod.destination_station]]
    if d1 >= INF or d2 >= INF:
        return 0.0
    age = state.current_time_step - pod.entry_time
    total = age + d1 + d2
    return 100.0 * math.exp(-total / SCORE_SCALE)


def _signature(state: GraphState) -> tuple:
    return tuple(sorted((p.id, p.current_node, p.carried_by) for p in state.active_pods))


def _recompute(state: GraphState) -> None:
    global _ASSIGN, _SEEN_SIG, _RECOMPUTE_STEP
    units = sorted(
        (u for u in state.drive_units if not u.in_transit and not u.carrying and u.has_capacity),
        key=lambda u: u.id,
    )
    pods = [p for p in state.active_pods if p.carried_by is None and p.current_node is not None]
    new_assign: Dict[int, Optional[str]] = {u.id: None for u in state.drive_units}
    if units and pods:
        rewards: List[List[float]] = []
        for u in units:
            row = [_pod_reward(state, u.current_node, p) for p in pods]
            rewards.append(row)
        matched = _hungarian_maximize(rewards, len(units), len(pods))
        for i, pod_idx in enumerate(matched):
            if pod_idx >= 0:
                new_assign[units[i].id] = pods[pod_idx].id
    _ASSIGN = new_assign
    _SEEN_SIG = _signature(state)
    _RECOMPUTE_STEP = state.current_time_step


def _unit_wants_work(state: GraphState, u) -> bool:
    if u.carrying or not u.has_capacity:
        return False
    pid = _ASSIGN.get(u.id) if _ASSIGN else None
    if pid:
        p = state.get_pod(pid)
        if p is not None and p.carried_by is None and p.current_node is not None:
            return False
    return True


def _ensure_assignments(state: GraphState) -> None:
    global _ASSIGN, _SEEN_SIG, _RECOMPUTE_STEP
    sig = _signature(state)
    if _ASSIGN is None or sig != _SEEN_SIG:
        _recompute(state)
    elif _RECOMPUTE_STEP != state.current_time_step and any(
        _unit_wants_work(state, u) for u in state.drive_units if not u.in_transit
    ):
        _recompute(state)


def _compute_goal(state: GraphState, unit) -> Optional[int]:
    if unit.carrying:
        pods = [state.get_pod(pid) for pid in unit.carrying]
        pods = [p for p in pods if p is not None]
        if not pods:
            return None
        oldest = min(pods, key=lambda p: (p.entry_time, p.id))
        if oldest.destination_station == unit.current_node:
            return None
        return oldest.destination_station
    if not unit.has_capacity:
        return None
    pid = _ASSIGN.get(unit.id)
    if pid is None:
        return None
    p = state.get_pod(pid)
    if p is None or p.carried_by is not None or p.current_node is None:
        return None
    if p.current_node == unit.current_node:
        return None
    return p.current_node


def _reposition(state: GraphState, unit) -> None:
    cur = unit.current_node
    if cur not in _IDX:
        return
    best = None
    best_d = INF
    for s in _STORAGE:
        if s not in _IDX:
            continue
        d = _FW[_IDX[cur]][_IDX[s]]
        if d < best_d:
            best_d = d
            best = s
    if best is None or best == cur:
        return
    plan = _dijkstra(state, cur, best)
    if plan:
        _PLANS[unit.id] = plan
        _GOALS[unit.id] = ("repos", best)


def _commit(state: GraphState, unit) -> Optional[int]:
    plan = _PLANS.get(unit.id)
    if not plan or len(plan) < 2:
        return None
    cur = unit.current_node
    if plan[0] != cur:
        _PLANS[unit.id] = []
        _GOALS[unit.id] = None
        return None
    nxt = plan[1]
    edge = state.get_edge(cur, nxt)
    if edge is None:
        _PLANS[unit.id] = []
        _GOALS[unit.id] = None
        return None
    if edge.capacity is not None and state.edge_occupancy(cur, nxt) >= edge.capacity:
        return None
    node = state.get_node(nxt)
    if node is not None and node.capacity is not None and state.node_occupancy(nxt) >= node.capacity:
        return None
    _PLANS[unit.id] = plan[1:]
    return nxt


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    try:
        unit = state.get_drive_unit(drive_unit_id)
        if unit is None or unit.in_transit:
            return None
        _ensure_topology(state)
        _ensure_assignments(state)
        goal = _compute_goal(state, unit)
        if goal is not None:
            if _GOALS.get(drive_unit_id) != goal or not _PLANS.get(drive_unit_id):
                plan = _dijkstra(state, unit.current_node, goal)
                if plan and plan[-1] == goal:
                    _PLANS[drive_unit_id] = plan
                    _GOALS[drive_unit_id] = goal
                else:
                    _PLANS[drive_unit_id] = []
                    _GOALS[drive_unit_id] = None
        else:
            if _GOALS.get(drive_unit_id) is not None:
                _PLANS[drive_unit_id] = []
                _GOALS[drive_unit_id] = None
            plan = _PLANS.get(drive_unit_id, [])
            if not plan:
                _reposition(state, unit)
        return _commit(state, unit)
    except Exception:
        return None