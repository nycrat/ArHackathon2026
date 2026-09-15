"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the drive_unit_next_move function in this module.

*****IMPORTANT*****
Team name: Short Eric
Email address: xaviergbradford@gmail.com, eric.yoon4@gmail.com, avahxiao@gmail.com
*******************
"""

from typing import Optional, Dict, List, Tuple, Set
import heapq
import math
import time

from ar_hackathon.models.graph_state import GraphState
from ar_hackathon.models.drive_unit import DriveUnit

INF = float("inf")
PENALTY = 1_000_000.0

BATCH_EXTRA = 0.35      # max relative detour tolerated to batch-grab a pod
BATCH_ABS = 2.0         # max absolute detour tolerated to batch-grab a pod

SIM_HORIZON = 16        # how many steps each candidate move is rolled forward
MAX_CANDIDATES = 8      # cap on moves compared per call
CALL_BUDGET = 0.25      # wall-clock seconds allowed per call before giving up
TERM_W = 1.0            # weight on the terminal heuristic for undone pods

# ---- Persistent per-graph caches (rebuilt when the floor layout changes) ----
_graph_key = None
_adj = None
_distance_memo: Dict[Tuple[int, int], float] = {}


def _signature(state: GraphState):
    nodes = tuple(sorted((n.id, n.node_type, n.capacity) for n in state.nodes))
    edges = tuple(sorted(
        (e.from_node, e.to_node, e.weight, e.capacity, e.bidirectional)
        for e in state.edges))
    return (nodes, edges)


def _build_adjacency(state: GraphState) -> Dict[int, List[Tuple[int, object]]]:
    adj = {n.id: [] for n in state.nodes}
    for e in state.edges:
        if e.from_node not in adj or e.to_node not in adj:
            continue
        adj[e.from_node].append((e.to_node, e))
        if e.bidirectional:
            adj[e.to_node].append((e.from_node, e))
    return adj


def _adj_for(state: GraphState) -> Dict[int, List[Tuple[int, object]]]:
    global _graph_key, _adj
    key = _signature(state)
    if key != _graph_key:
        _graph_key = key
        _adj = _build_adjacency(state)
        _distance_memo.clear()
    return _adj


def _static_distance(state: GraphState, a: int, b: int) -> float:
    """Shortest travel time between nodes on the uncongested graph."""
    if a == b:
        return 0.0
    memo_key = (a, b) if a < b else (b, a)
    cached = _distance_memo.get(memo_key)
    if cached is not None:
        return cached

    adj = _adj_for(state)
    dist = {a: 0.0}
    pq = [(0.0, a)]
    visited = set()
    result = INF
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == b:
            result = d
            break
        for v, e in adj.get(u, ()):
            if v in visited:
                continue
            nd = d + e.weight
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    _distance_memo[memo_key] = result
    return result


def _path_to(state: GraphState, start: int, goal: int) -> List[int]:
    """
    Shortest path from start to goal that also avoids (as much as is cheaper)
    aisles and nodes that are currently at capacity. Returns the node list
    including both endpoints, or [] if the goal is unreachable.
    """
    if start == goal:
        return [start]
    adj = _adj_for(state)

    full_edges = set()
    for e in state.edges:
        if e.capacity is not None and state.edge_occupancy(e.from_node, e.to_node) >= e.capacity:
            full_edges.add(id(e))
    full_nodes = set()
    for n in state.nodes:
        if n.capacity is not None and state.node_occupancy(n.id) >= n.capacity:
            full_nodes.add(n.id)

    dist = {start: 0.0}
    prev = {start: None}
    pq = [(0.0, start)]
    visited = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == goal:
            path = []
            cur = u
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        for v, e in adj.get(u, ()):
            if v in visited:
                continue
            cost = e.weight
            if id(e) in full_edges:
                cost += PENALTY
            if v in full_nodes:
                cost += PENALTY
            nd = d + cost
            if nd < dist.get(v, INF):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return []


def _can_enter(state: GraphState, unit: DriveUnit, node_id: int) -> bool:
    """Mirror of the referee's is_valid_move against the current state."""
    if node_id == unit.current_node:
        return False
    edge = state.get_edge(unit.current_node, node_id)
    if edge is None:
        return False
    if edge.capacity is not None and \
            state.edge_occupancy(unit.current_node, node_id) >= edge.capacity:
        return False
    node = state.get_node(node_id)
    if node is not None and node.capacity is not None and \
            state.node_occupancy(node_id) >= node.capacity:
        return False
    return True


def _tsp_order(state: GraphState, start: int, stations: List[int]) -> List[int]:
    """Visiting order for several stations via greedy nearest-neighbour TSP."""
    remaining = list(stations)
    order = []
    cur = start
    while remaining:
        nxt = min(remaining, key=lambda s: (_static_distance(state, cur, s), s))
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return order


def _standby_target(state: GraphState, unit: DriveUnit, taken: Set[int]) -> Optional[int]:
    """
    Where an idle, empty drive unit should loiter. A unit parked on a station
    dock must vacate it. Otherwise camp at the nearest storage node that no
    other idle unit has already staked out, so upcoming pods are met quickly
    without several units piling onto the same storage.
    """
    node = state.get_node(unit.current_node)
    storages = [n.id for n in state.nodes if n.node_type == "storage"]
    if not storages:
        return None
    storages.sort(key=lambda s: (_static_distance(state, unit.current_node, s), s))

    def _free():
        for s in storages:
            if s not in taken:
                return s
        return None

    if node is not None and node.node_type == "station":
        # A full dock is precious: vacate immediately (reuse a taken storage
        # only if no free one remains).
        return _free() if _free() is not None else storages[0]
    return _free()


def _batch_pickup_target(state: GraphState, unit: DriveUnit,
                         dest_set: Set[int]) -> Optional[int]:
    """
    If this carrying unit still has free capacity, consider grabbing another
    active pod that shares one of its destinations when the detour is cheap.
    """
    go_straight = min(_static_distance(state, unit.current_node, d) for d in dest_set)
    best = None
    best_extra = INF
    for p in state.active_pods:
        if p.carried_by is not None or p.current_node is None:
            continue
        if p.destination_station not in dest_set:
            continue
        if p.current_node == unit.current_node:
            continue  # would already be auto-picked
        src_dist = _static_distance(state, unit.current_node, p.current_node)
        via = src_dist + _static_distance(state, p.current_node, p.destination_station)
        extra = via - go_straight
        if extra < best_extra and extra <= max(BATCH_ABS, BATCH_EXTRA * go_straight):
            best_extra = extra
            best = p.current_node
    return best


def _plan_all_targets(state: GraphState) -> Dict[int, Optional[int]]:
    """
    Assign every drive unit a destination (a station to deliver to, a pod's
    storage node to fetch from, or a standby location). Re-computed from the
    supplied snapshot each call so it stays consistent across all units.
    """
    units = sorted(state.drive_units, key=lambda u: u.id)
    targets: Dict[int, Optional[int]] = {}

    pods = [p for p in state.active_pods
            if p.carried_by is None and p.current_node is not None]
    pods.sort(key=lambda p: (p.entry_time, p.id))

    carrying = [u for u in units if u.carrying]
    empty = [u for u in units if not u.carrying]

    for u in carrying:
        dests: Set[int] = set()
        for pid in u.carrying:
            p = state.get_pod(pid)
            if p is not None:
                dests.add(p.destination_station)
        if not dests:
            targets[u.id] = _standby_target(state, u, set())
        else:
            order = _tsp_order(state, u.current_node, list(dests))
            target = order[0]
            if u.has_capacity:
                batch = _batch_pickup_target(state, u, dests)
                if batch is not None:
                    target = batch
            targets[u.id] = target

    assigned: Dict[int, Tuple[int, int]] = {}  # unit -> (source, count)
    for pod in pods:
        src = pod.current_node
        best_unit = None
        best_key = None
        for u in empty:
            info = assigned.get(u.id)
            if info is None:
                score = _static_distance(state, u.current_node, src)
            elif info[0] == src and info[1] < u.capacity:
                score = _static_distance(state, u.current_node, src)
            else:
                continue
            key = (score, u.id)
            if best_key is None or key < best_key:
                best_unit = u
                best_key = key
        if best_unit is None:
            continue
        info = assigned.get(best_unit.id)
        assigned[best_unit.id] = (src, 1 if info is None else info[1] + 1)
        targets[best_unit.id] = src

    claimed = set()
    for u in empty:
        if u.id not in targets:
            t = _standby_target(state, u, claimed)
            if t is not None:
                targets[u.id] = t
                claimed.add(t)

    return targets


def _plan_target(state: GraphState, drive_unit_id: int) -> Optional[int]:
    """Destination for a single drive unit (see _plan_all_targets)."""
    return _plan_all_targets(state).get(drive_unit_id)


def _start_move(state: GraphState, unit: DriveUnit, next_node: int) -> None:
    """Port of engine._move_drive_unit."""
    edge = state.get_edge(unit.current_node, next_node)
    unit.in_transit = True
    unit.transit_destination = next_node
    unit.transit_remaining_time = edge.weight


def _advance_units(state: GraphState) -> None:
    """Port of engine._advance_drive_units."""
    for unit in state.drive_units:
        if unit.in_transit:
            unit.transit_remaining_time -= 1
            if unit.transit_remaining_time <= 0:
                unit.current_node = unit.transit_destination
                unit.in_transit = False
                unit.transit_destination = None
                unit.transit_remaining_time = 0


def _resolve_dp(state: GraphState) -> None:
    """Port of engine._process_deliveries_and_pickups."""
    for unit in sorted(state.drive_units, key=lambda u: u.id):
        if unit.in_transit:
            continue
        for pod_id in list(unit.carrying):
            pod = state.get_pod(pod_id)
            if pod is not None and pod.destination_station == unit.current_node:
                unit.carrying.remove(pod_id)
                pod.carried_by = None
                pod.current_node = unit.current_node
                pod.delivery_time = state.current_time_step
                if pod in state.active_pods:
                    state.active_pods.remove(pod)
                state.delivered_pods.append(pod)
        if unit.has_capacity:
            waiting = [pod for pod in state.active_pods
                       if pod.carried_by is None and pod.current_node == unit.current_node]
            waiting.sort(key=lambda p: (p.entry_time, p.id))
            for pod in waiting:
                if not unit.has_capacity:
                    break
                pod.carried_by = unit.id
                pod.current_node = None
                unit.carrying.append(pod.id)


def _hop_to(state: GraphState, unit: DriveUnit, target: Optional[int]) -> Optional[int]:
    """First hop of the congestion-aware path toward target, or None."""
    if target is None or target == unit.current_node:
        return None
    path = _path_to(state, unit.current_node, target)
    return path[1] if len(path) >= 2 else None


def _sim_step(state: GraphState, spawn_map=None, threshold: Optional[int] = None) -> None:
    """
    Advance a simulated state by one full referee step.

    spawn_map: optional {entry_time: [Pod,...]} used only by offline validation
               harnesses (the live API never knows the pod schedule).
    threshold: if set, only idle units with id >= threshold are routed this
               step (lower ids were already polled by the referee).
    """
    if spawn_map:
        for pod in spawn_map.get(state.current_time_step, ()):
            state.active_pods.append(pod.deep_copy())

    _resolve_dp(state)

    targets = _plan_all_targets(state)
    for unit in sorted(state.drive_units, key=lambda u: u.id):
        if unit.in_transit:
            continue
        if threshold is not None and unit.id < threshold:
            continue
        hop = _hop_to(state, unit, targets.get(unit.id))
        if hop is not None and _can_enter(state, unit, hop):
            _start_move(state, unit, hop)

    _advance_units(state)
    _resolve_dp(state)
    state.current_time_step += 1


def _sim_rollout(state: GraphState, unit_id: int, spawn_map, horizon: int) -> float:
    """
    Finish the current step (routing only units with id >= unit_id, since the
    referee already committed the lower ids) and then roll `horizon` more
    steps. Returns the score of the resulting world.
    """
    _sim_step(state, spawn_map, threshold=unit_id)
    for _ in range(horizon - 1):
        _sim_step(state, spawn_map)
    return _score_state(state)


def _effective_horizon(state: GraphState) -> int:
    """
    Planning ahead helps most when the map is sparse; under heavy traffic the
    upstream decisions dominate, so a shorter lookahead is both cheaper and
    less myopic about congestion others will cause.
    """
    if len(state.drive_units) >= 4:
        return 12
    return SIM_HORIZON


def _score_state(state: GraphState) -> float:
    """Reward for a simulated world: delivered pods score their decay, and
    still-active pods get an optimistic terminal estimate, so the controller
    is nudged to finish carried work and keep short remaining legs."""
    total = 0.0
    t = state.current_time_step
    for pod in state.delivered_pods:
        total += math.exp(-(pod.delivery_time - pod.entry_time) / 50.0)
    for pod in state.active_pods:
        if pod.carried_by is not None:
            carrier = state.get_drive_unit(pod.carried_by)
            if carrier is not None:
                q = _static_distance(state, carrier.current_node, pod.destination_station)
            else:
                q = 0.0
        elif pod.current_node is not None:
            q = _static_distance(state, pod.current_node, pod.destination_station)
        else:
            q = 0.0
        total += TERM_W * math.exp(-(t + q - pod.entry_time) / 50.0)
    return total


def _greedy_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """The non-lookahead baseline: follow the global greedy assignment."""
    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None
    hop = _hop_to(state, unit, _plan_target(state, drive_unit_id))
    if hop is not None and _can_enter(state, unit, hop):
        return hop
    return None


def _choose_move(drive_unit_id: int, state: GraphState, t0: float) -> Optional[int]:
    """
    One-ply model-predictive control. Enumerate the enterable moves plus
    'wait', roll each forward under the greedy baseline policy, and pick the
    move with the best simulated score. Ties fall back to the greedy move.
    """
    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None

    greedy = _greedy_move(drive_unit_id, state)

    candidates: List[Optional[int]] = []
    seen = set()
    if greedy is not None:
        candidates.append(greedy)
        seen.add(greedy)

    others = []
    for nb in state.neighbors(unit.current_node):
        if nb in seen:
            continue
        edge = state.get_edge(unit.current_node, nb)
        if edge is not None and _can_enter(state, unit, nb):
            others.append((edge.weight, nb))
    others.sort(key=lambda x: (x[0], x[1]))
    for _, nb in others:
        if len(candidates) >= MAX_CANDIDATES:
            break
        candidates.append(nb)
        seen.add(nb)

    candidates.append(None)  # always offer "wait"

    best_move = greedy
    best_score = -INF
    horizon = _effective_horizon(state)
    for move in candidates:
        if time.monotonic() - t0 > CALL_BUDGET:
            break
        sim = state.deep_copy()
        sim_unit = sim.get_drive_unit(drive_unit_id)
        if move is not None:
            _start_move(sim, sim_unit, move)
        score = _sim_rollout(sim, drive_unit_id, None, horizon)
        if score > best_score + 1e-9:
            best_score = score
            best_move = move
    return best_move


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """
    Determine the next node for a drive unit to move to.

    Pickups and deliveries are automatic: a drive unit with free capacity
    that stops at a node with a waiting pod picks it up, and a drive unit
    that reaches a carried pod's destination station drops it off.

    Returns:
        next_node_id: ID of an adjacent node to move to, or None to wait
                      at the current node
    """
    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None
    t0 = time.monotonic()
    try:
        return _choose_move(drive_unit_id, state, t0)
    except Exception:
        return _greedy_move(drive_unit_id, state)
