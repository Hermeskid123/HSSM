"""Finite-candidate segmental HSSM DP: Wang et al. (2008), Sec. 5, Eqs. 11–18.

A segment candidate supplies its internal feature ordering and cost (restricted
beta in Eq. 13). Keep t (total feature count), state, duration, and candidate
identity; minimize incoming costs and backtrack. Exact over supplied candidates,
not over all possible image-feature permutations. No learned probabilities.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Segment:
    state: int
    points: np.ndarray
    cost: float
    duration: int
    expected_start: np.ndarray
    expected_end: np.ndarray
    allowed_predecessors: tuple = ()
    variant: str = ''
    terms: dict = field(default_factory=dict)


def transition(previous, current):
    """Feature-transition cost: deviation from model inter-segment displacement."""
    observed = current.points[0] - previous.points[-1]
    expected = current.expected_start - previous.expected_end
    return float(.025 * np.sum((observed-expected)**2))


def decode(graph, candidates, start=1, end=20, transition_cost=transition):
    """Segment-boundary form of Eq. 12 with explicit registration length t.

    delta[t,j,l,v] = E(j,l,v)+D(j,l) +
      min_(i,k,u) {delta[t-l,i,k,u]+A(i,j)+Phi(u,v)}.
    A=0 for legal arcs and infinity otherwise. E includes restricted internal
    beta costs. Pi(start)=0; all other starts are forbidden. Durations are
    carried by candidates. DAG states replace the paper's bounded Tmax loop.
    """
    states = sorted(graph)
    if any(j <= i for i in graph for j in graph[i]):
        raise ValueError('Decoder requires a topologically numbered DAG')
    incoming = {j: [i for i in states if j in graph[i]] for j in states}
    table, back, entries = {}, {}, {j: [] for j in states}
    transitions = {(i,j): np.array([[transition_cost(a,b) for b in candidates[j]]
                                   for a in candidates[i]])
                   for i in states for j in graph[i]}
    for j in states:
        for v, segment in enumerate(candidates[j]):
            if segment.state != j or segment.duration != len(segment.points):
                raise ValueError('Inconsistent candidate state/duration')
            if j == start:
                key = (segment.duration, j, segment.duration, v)
                table[key], back[key] = segment.cost, None
                entries[j].append(key)
                continue
            best_by_t = {}
            for i in incoming[j]:
                if segment.allowed_predecessors and i not in segment.allowed_predecessors:
                    continue
                for old in entries[i]:
                    prev = candidates[i][old[3]]
                    t = old[0] + segment.duration
                    score = table[old] + transitions[i,j][old[3],v] + segment.cost
                    if t not in best_by_t or score < best_by_t[t][0]:
                        best_by_t[t] = (score, old)
            for t, (score, old) in best_by_t.items():
                key = (t, j, segment.duration, v)
                table[key], back[key] = score, old
                entries[j].append(key)
    if not entries[end]:
        raise ValueError('No legal complete registration')
    last = min(entries[end], key=table.get)
    keys, cursor = [], last
    while cursor is not None:
        keys.append(cursor)
        cursor = back[cursor]
    keys.reverse()
    return dict(cost=float(table[last]), keys=keys,
                segments=[candidates[k[1]][k[3]] for k in keys],
                states=[k[1] for k in keys], total_features=last[0])
