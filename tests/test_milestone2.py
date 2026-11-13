"""Decoder optimum checked independently by exhaustive complete-path enumeration."""
from itertools import product
import numpy as np
import pytest
from hssm.decoder import Segment, decode, transition
from hssm.hand_model import (GRAPH, CONFIGURATIONS, state_path, trace_configurations,
                            render_hand, geometry)
from hssm.milestone2 import (edge_lookup, make_candidates, infer, palm_coverage,
                             predicted_palm, pose_grid)


def enumerate_paths(graph, start, end):
    if start == end:
        return [[end]]
    return [[start]+tail for j in graph[start] for tail in enumerate_paths(graph,j,end)]


@pytest.mark.parametrize('seed',range(5))
def test_segmental_dp_against_exhaustive(seed):
    rng = np.random.default_rng(seed)
    graph = {1:(2,3),2:(4,),3:(4,),4:()}
    candidates = {}
    for state in graph:
        candidates[state] = []
        for duration in (2,3):
            points = rng.normal(size=(duration,2))
            candidates[state].append(Segment(state,points,float(rng.uniform(-2,2)),
                duration,np.array([state,0.]),np.array([state+1,0.])))
    scored = []
    for path in enumerate_paths(graph,1,4):
        for ids in product(range(2),repeat=len(path)):
            segments = [candidates[s][v] for s,v in zip(path,ids)]
            cost = sum(s.cost for s in segments)+sum(transition(a,b) for a,b in zip(segments,segments[1:]))
            scored.append((cost,path,ids,segments))
    expected = min(scored,key=lambda item:item[0])
    actual = decode(graph,candidates,end=4)
    assert actual['cost'] == pytest.approx(expected[0])
    assert actual['states'] == expected[1]
    assert tuple(k[3] for k in actual['keys']) == expected[2]
    assert actual['total_features'] == sum(s.duration for s in expected[3])
    assert actual['cost'] == pytest.approx(sum(s.cost for s in actual['segments'])+
        sum(transition(a,b) for a,b in zip(actual['segments'],actual['segments'][1:])))


def test_all_figure3_paths_and_shared_state20():
    paths = enumerate_paths(GRAPH,1,20)
    assert len(paths) == 16
    assert {tuple(p) for p in paths} == {tuple(state_path(b)) for b in CONFIGURATIONS}
    assert len(trace_configurations()) == 16
    assert state_path((1,1,1,1)) == [1,2,3,4,5,6,7,9,10,11,13,14,15,17,18,20]
    assert state_path((0,0,0,0)) == [1,2,3,4,8,12,16,19,20]
    assert GRAPH[18] == (20,) and GRAPH[19] == (20,)
    assert '20E' in geometry() and '20H' in geometry()


def test_disconnected_graph_rejected():
    with pytest.raises(ValueError,match='No legal complete'):
        decode({1:(),2:()},{1:[],2:[]},end=2)


@pytest.mark.parametrize('bits',[(0,0,0,0),(1,1,1,1),(1,0,1,0)])
def test_image_registration_without_truth_inputs(bits):
    sample = render_hand(bits,123)
    prediction = infer(sample['mask'],'fixed')
    assert prediction['bits'] == bits
    assert palm_coverage(sample['palm'],predicted_palm(prediction)) >= .75
    edges,nearest = edge_lookup(sample['mask'])
    for segment in prediction['segments']:
        xy = segment.points.astype(int)
        assert np.all(edges[xy[:,1],xy[:,0]] > 0)
    assert all(b in GRAPH[a] for a,b in zip(prediction['states'],prediction['states'][1:]))
    last = prediction['segments'][-1]
    assert last.variant == ('20E' if bits[3] else '20H')
    assert last.allowed_predecessors == ((18,) if bits[3] else (19,))


def test_localization_metric_and_empty_prediction():
    palm = [np.array([[0.,0.],[0.,20.]])]
    assert palm_coverage(palm,palm) == 1
    assert palm_coverage(palm,[palm[0]+[4,0]],tolerance=3) == 0


def test_pose_search_does_not_use_ground_truth():
    assert len(pose_grid('translation')) == 49
    assert len(pose_grid('rotation')) == 5
    assert len(pose_grid('scale')) == 5
    with pytest.raises(ValueError,match='no edges'):
        edge_lookup(np.zeros((20,20),np.uint8))


def test_full_hand_dp_against_all_16_structures():
    sample = render_hand((1,0,1,0),234)
    _,nearest = edge_lookup(sample['mask'])
    generated = make_candidates(nearest,(0,0,0,1))
    # One duration/offset per state, both legal realizations of shared state20.
    candidates = {s:([c[4],c[10]] if s == 20 else [c[4]]) for s,c in generated.items()}
    scored = []
    for bits in CONFIGURATIONS:
        path = state_path(bits)
        segments = [candidates[s][int(not bits[3]) if s == 20 else 0] for s in path]
        cost = sum(s.cost for s in segments)+sum(transition(a,b) for a,b in zip(segments,segments[1:]))
        scored.append((cost,path))
    expected = min(scored,key=lambda item:item[0])
    result = decode(GRAPH,candidates)
    assert result['cost'] == pytest.approx(expected[0])
    assert result['states'] == expected[1]


def test_clutter_is_separate_from_pose_variation():
    assert pose_grid('clutter') == pose_grid('fixed')
    clean = render_hand((1,0,1,0),123)
    noisy = render_hand((1,0,1,0),123,clutter=True)
    assert np.count_nonzero(noisy['mask']) > np.count_nonzero(clean['mask'])
    np.testing.assert_array_equal(noisy['clean'],clean['mask'])
    for a,b in zip(noisy['palm'],clean['palm']):
        np.testing.assert_array_equal(a,b)
