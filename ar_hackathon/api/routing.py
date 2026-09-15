"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the drive_unit_next_move function in this module.

*****IMPORTANT*****
Team name: Short Eric
Email address: xaviergbradford@gmail.com, eric.yoon4@gmail.com, avahxiao@gmail.com
*******************
"""

import heapq
import itertools
import math
from typing import Dict, List, Optional, Set, Tuple

from ar_hackathon.models.graph_state import GraphState

OCCUPANCY_PENALTY = 3.0
PLAN_BIAS = 0.4
SCORE_SCALE = 50.0
INF = float("inf")

_ASSIGN: Optional[Dict[int, Optional[str]]] = None
_SEEN_SIG: Optional[tuple] = None
_RECOMPUTE_STEP: Optional[int] = None

_TOPOLOGY_FP: Optional[Tuple[tuple, tuple]] = None
_LAST_STEP: Optional[int] = None
_IDX: Dict[int, int] = {}
_ADJ: Dict[int, List[tuple]] = {}
_FW: List[List[float]] = []
_STORAGE: Set[int] = set()

_PLANS: Dict[int, List[int]] = {}
_GOALS: Dict[int, object] = {}
_BATCH: Dict[int, List[str]] = {}


def _fingerprint(state: GraphState) -> Tuple[tuple, tuple]:
    nodes = tuple(sorted((n.id, n.node_type, n.capacity) for n in state.nodes))
    edges = tuple(sorted((e.from_node, e.to_node, e.weight, e.capacity, e.bidirectional) for e in state.edges))
    return nodes, edges


def _ensure_topology(state: GraphState) -> None:
    global _TOPOLOGY_FP, _LAST_STEP, _IDX, _ADJ, _FW, _STORAGE, _ASSIGN, _PLANS, _GOALS, _BATCH
    now = state.current_time_step
    if _LAST_STEP is not None and now < _LAST_STEP:
        _PLANS.clear()
        _GOALS.clear()
        _BATCH.clear()
        _ASSIGN = None
    _LAST_STEP = now
    fp = _fingerprint(state)
    if fp == _TOPOLOGY_FP:
        return
    _PLANS.clear()
    _GOALS.clear()
    _BATCH.clear()
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


def _plan_used_edges(uid: int) -> set:
    used = set()
    for u, pl in _PLANS.items():
        if u == uid or len(pl) < 2:
            continue
        for i in range(len(pl) - 1):
            a, b = pl[i], pl[i + 1]
            if a > b:
                a, b = b, a
            used.add((a, b))
    return used


def _dijkstra(state: GraphState, start: int, goal: int, uid: Optional[int] = None) -> Optional[List[int]]:
    if start == goal:
        return [start]
    occ = {}
    for e in state.edges:
        if e.bidirectional:
            key = (min(e.from_node, e.to_node), max(e.from_node, e.to_node))
        else:
            key = (e.from_node, e.to_node)
        occ[key] = state.edge_occupancy(e.from_node, e.to_node)
    used = _plan_used_edges(uid) if uid is not None else set()
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
            if key in used:
                cost += PLAN_BIAS
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


def _dist(a: int, b: int) -> float:
    if a in _IDX and b in _IDX:
        return _FW[_IDX[a]][_IDX[b]]
    return INF


def _build_legs(state: GraphState, batch_ids: List[str], carried_ids: List[str]) -> List[tuple]:
    items: List[tuple] = []
    seen = set()
    for pid in carried_ids:
        p = state.get_pod(pid)
        if p is None:
            continue
        seen.add(pid)
        items.append(("delivery", p.destination_station, pid))
    for pid in batch_ids:
        if pid in seen:
            continue
        p = state.get_pod(pid)
        if p is None:
            continue
        if pid in carried_ids:
            items.append(("delivery", p.destination_station, pid))
        elif p.carried_by is None and p.current_node is not None:
            items.append(("pickup", p.current_node, pid))
            items.append(("delivery", p.destination_station, pid))
        else:
            continue
        seen.add(pid)
    return items


def _order_legs(state: GraphState, current: int, items: List[tuple], now: int):
    n = len(items)
    if n == 0:
        return [], {}, 0.0

    def feasible(order):
        pickup_pos = {}
        for i, (kind, node, pid) in enumerate(order):
            if kind == "pickup":
                pickup_pos[pid] = i
        for i, (kind, node, pid) in enumerate(order):
            if kind == "delivery" and pid in pickup_pos and i < pickup_pos[pid]:
                return False
        return True

    def cost(order):
        prev = current
        t = 0.0
        for (kind, node, pid) in order:
            t += _dist(prev, node)
            prev = node
        return t

    def tie_key(order):
        total = 0
        pos = 0
        for (kind, node, pid) in order:
            p = state.get_pod(pid)
            if kind == "delivery" and p is not None:
                total += pos * max(0, now - p.entry_time)
            pos += 1
        return total

    best_order = None
    best_cost = INF
    best_tie = None
    if n <= 6:
        for perm in itertools.permutations(range(n)):
            order = [items[i] for i in perm]
            if not feasible(order):
                continue
            c = cost(order)
            tk = tie_key(order)
            if best_order is None or c < best_cost - 1e-9 or (abs(c - best_cost) <= 1e-9 and tk < best_tie):
                best_order = order
                best_cost = c
                best_tie = tk
    else:
        remaining = list(items)
        picked = set()
        prev = current
        while remaining:
            best_i = None
            best_d = INF
            for i, (kind, node, pid) in enumerate(remaining):
                if kind == "delivery" and pid not in picked and any(
                    k == "pickup" and q == pid for (k, q, _) in remaining
                ):
                    continue
                nd = _dist(prev, node)
                if nd < best_d:
                    best_d = nd
                    best_i = i
            leg = remaining.pop(best_i)
            if leg[0] == "pickup":
                picked.add(leg[2])
            best_order = (best_order or []) + [leg]
            prev = leg[1]
        best_cost = cost(best_order)
        best_tie = tie_key(best_order)
        best_order = list(best_order)

    times: Dict[str, float] = {}
    prev = current
    acc = 0.0
    for (kind, node, pid) in best_order:
        acc += _dist(prev, node)
        prev = node
        if kind == "delivery":
            times[pid] = acc
    return best_order, times, best_cost


def _tour_value(state: GraphState, unit_node: int, batch_ids: List[str], now: int) -> float:
    current = unit_node
    items = _build_legs(state, batch_ids, [])
    if not items:
        return 0.0
    _, times, _ = _order_legs(state, current, items, now)
    total = 0.0
    for pid, travel in times.items():
        p = state.get_pod(pid)
        if p is None:
            continue
        age = max(0, now - p.entry_time)
        total += 100.0 * math.exp(-(age + travel) / SCORE_SCALE)
    return total


def _marginal_gain(state: GraphState, unit, batch_ids: List[str], cand_id: str, now: int) -> float:
    base = _tour_value(state, unit.current_node, batch_ids, now)
    extended = _tour_value(state, unit.current_node, batch_ids + [cand_id], now)
    return extended - base


def _first_obligation(state: GraphState, unit) -> Optional[int]:
    carried = list(unit.carrying)
    batch = [pid for pid in (_BATCH.get(unit.id) or [])]
    items = _build_legs(state, batch, carried)
    if not items:
        return None
    order, _, _ = _order_legs(state, unit.current_node, items, state.current_time_step)
    for (kind, node, pid) in order:
        p = state.get_pod(pid)
        if p is None:
            continue
        if kind == "pickup":
            if p.carried_by == unit.id:
                continue
            if p.carried_by is None and p.current_node is not None:
                return node
            continue
        if pid in unit.carrying:
            return node
    return None


def _signature(state: GraphState) -> tuple:
    return tuple(sorted((p.id, p.current_node, p.carried_by) for p in state.active_pods))


def _recompute(state: GraphState) -> None:
    global _ASSIGN, _SEEN_SIG, _RECOMPUTE_STEP, _BATCH
    units = sorted(
        (u for u in state.drive_units if not u.in_transit and not u.carrying and u.has_capacity),
        key=lambda u: u.id,
    )
    pods = [p for p in state.active_pods if p.carried_by is None and p.current_node is not None]
    now = state.current_time_step
    new_assign: Dict[int, Optional[str]] = {u.id: None for u in state.drive_units}
    _BATCH = {u.id: [] for u in state.drive_units}
    if units and pods:
        rewards = [[_tour_value(state, u.current_node, [p.id], now) for p in pods] for u in units]
        matched = _hungarian_maximize(rewards, len(units), len(pods))
        claimed = set()
        for i, pod_idx in enumerate(matched):
            if pod_idx >= 0:
                pid = pods[pod_idx].id
                claimed.add(pid)
                _BATCH[units[i].id].append(pid)
                new_assign[units[i].id] = pid
        unclaimed = [p for p in pods if p.id not in claimed]
        improved = True
        while improved:
            improved = False
            for u in units:
                if len(_BATCH[u.id]) >= u.capacity:
                    continue
                best_pod = None
                best_gain = 0.0
                for p in unclaimed:
                    gain = _marginal_gain(state, u, _BATCH[u.id], p.id, now)
                    if gain > best_gain:
                        best_gain = gain
                        best_pod = p
                if best_pod is not None:
                    _BATCH[u.id].append(best_pod.id)
                    unclaimed.remove(best_pod)
                    improved = True

        def _batch_value(u, batch):
            return _tour_value(state, u.current_node, list(batch), now)

        improved = True
        while improved:
            improved = False
            for u_off in units:
                if not _BATCH[u_off.id]:
                    continue
                for p in list(_BATCH[u_off.id]):
                    for u_on in units:
                        if u_on.id == u_off.id or len(_BATCH[u_on.id]) >= u_on.capacity:
                            continue
                        off_left = [q for q in _BATCH[u_off.id] if q != p]
                        on_next = _BATCH[u_on.id] + [p]
                        cur = _batch_value(u_off, _BATCH[u_off.id]) + _batch_value(u_on, _BATCH[u_on.id])
                        new = _batch_value(u_off, off_left) + _batch_value(u_on, on_next)
                        if new > cur + 1e-9:
                            _BATCH[u_off.id] = off_left
                            _BATCH[u_on.id] = on_next
                            improved = True
    for u in units:
        if _BATCH.get(u.id):
            new_assign[u.id] = _BATCH[u.id][0]
    _ASSIGN = new_assign
    _SEEN_SIG = _signature(state)
    _RECOMPUTE_STEP = now


def _unit_wants_work(state: GraphState, u) -> bool:
    if u.carrying or not u.has_capacity:
        return False
    for pid in (_BATCH.get(u.id) or []):
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
    plan = _dijkstra(state, cur, best, unit.id)
    if plan:
        _PLANS[unit.id] = plan
        _GOALS[unit.id] = ("repos", best)


def _hop_valid(state: GraphState, unit, nxt: int) -> bool:
    edge = state.get_edge(unit.current_node, nxt)
    if edge is None:
        return False
    if edge.capacity is not None and state.edge_occupancy(unit.current_node, nxt) >= edge.capacity:
        return False
    node = state.get_node(nxt)
    if node is not None and node.capacity is not None and state.node_occupancy(nxt) >= node.capacity:
        return False
    return True


def _path_static_cost(path: List[int]) -> float:
    total = 0.0
    for i in range(len(path) - 1):
        total += _dist(path[i], path[i + 1])
    return total


def _block_wait(state: GraphState, unit, nxt: int) -> float:
    edge = state.get_edge(unit.current_node, nxt)
    if edge is not None and edge.capacity is not None and state.edge_occupancy(unit.current_node, nxt) >= edge.capacity:
        waits = [
            u.transit_remaining_time
            for u in state.drive_units
            if u.in_transit and edge.connects(u.current_node, u.transit_destination)
        ]
        return max(0.0, min(waits)) if waits else float(edge.weight)
    node = state.get_node(nxt)
    if node is not None and node.capacity is not None and state.node_occupancy(nxt) >= node.capacity:
        standing = [u for u in state.drive_units if not u.in_transit and u.current_node == nxt]
        if standing:
            return 1.0
        transit = [
            u.transit_remaining_time
            for u in state.drive_units
            if u.in_transit and u.transit_destination == nxt
        ]
        return max(0.0, min(transit)) if transit else 1.0
    return 0.0


def _try_hop(state: GraphState, unit) -> Optional[int]:
    plan = _PLANS.get(unit.id)
    if not plan or len(plan) < 2:
        return None
    cur = unit.current_node
    if plan[0] != cur:
        _PLANS[unit.id] = []
        _GOALS[unit.id] = None
        return None
    nxt = plan[1]
    if not _hop_valid(state, unit, nxt):
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
        goal = _first_obligation(state, unit)
        if goal is not None:
            if _GOALS.get(drive_unit_id) != goal or not _PLANS.get(drive_unit_id):
                plan = _dijkstra(state, unit.current_node, goal, drive_unit_id)
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
            if not unit.carrying and unit.has_capacity and not plan:
                _reposition(state, unit)
        hop = _try_hop(state, unit)
        if hop is None and goal is not None:
            plan = _PLANS.get(drive_unit_id, [])
            if len(plan) >= 2 and plan[0] == unit.current_node:
                wait = _block_wait(state, unit, plan[1])
                stay_cost = _path_static_cost(plan)
                replan = _dijkstra(state, unit.current_node, goal, drive_unit_id)
                if (
                    replan
                    and replan[-1] == goal
                    and len(replan) >= 2
                    and _hop_valid(state, unit, replan[1])
                    and _path_static_cost(replan) + 1e-9 < stay_cost + wait
                ):
                    _PLANS[drive_unit_id] = replan
                    _GOALS[drive_unit_id] = goal
                    hop = _try_hop(state, unit)
        return hop
    except Exception:
        return None
