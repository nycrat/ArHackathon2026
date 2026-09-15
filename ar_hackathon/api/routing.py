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

from ar_hackathon.models.graph_state import GraphState
from ar_hackathon.models.drive_unit import DriveUnit

INF = float("inf")
PENALTY = 1_000_000.0

BATCH_EXTRA = 0.35      # max relative detour tolerated to batch-grab a pod
BATCH_ABS = 2.0         # max absolute detour tolerated to batch-grab a pod

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


def _plan_target(state: GraphState, drive_unit_id: int) -> Optional[int]:
    """
    Assign every drive unit a destination (a station to deliver to, a pod's
    storage node to fetch from, or a standby location), then return the one
    for our unit. The assignment is recomputed each call from the supplied
    snapshot so it stays consistent across all units.
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

    return targets.get(drive_unit_id)


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

    target = _plan_target(state, drive_unit_id)
    if target is None or target == unit.current_node:
        return None

    path = _path_to(state, unit.current_node, target)
    if len(path) < 2:
        return None

    nxt = path[1]
    if _can_enter(state, unit, nxt):
        return nxt
    return None
