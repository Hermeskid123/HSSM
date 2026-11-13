"""Synthetic full-hand registration: finite pose grid and segmental graph DP.

Paper references: Figure3 (topology), Eq7 (additive score), Sec5 Eqs11–18
(segmental recurrence/backtracking). All geometry and costs here are project
adaptations. The candidate ordering restriction and state-balanced means differ
from the paper's pixel-level, learned probability model. See milestone2 readme.
"""
import argparse
import csv
import json
from itertools import product
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from .contours import resample_arc, write_overlay
from .decoder import Segment, decode
from .hand_model import (GRAPH, FINGERS, CONFIGURATIONS, STATE_MAPPING, geometry,
                         transform, render_hand, trace_configurations)

CONDITIONS = ('fixed','translation','rotation','scale','clutter')
TOLERANCE = 3.0


def edge_lookup(mask):
    """All component external edges; no endpoints, pose or labels as inputs."""
    contours,_ = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    edges = np.zeros_like(mask)
    cv2.drawContours(edges,contours,-1,255,1)
    if not np.any(edges):
        raise ValueError('Image has no edges')
    _, labels = cv2.distanceTransformWithLabels(255-edges,cv2.DIST_L2,5,
                                               labelType=cv2.DIST_LABEL_PIXEL)
    y,x = np.nonzero(edges)
    coords = np.zeros((int(labels.max())+1,2),float)
    coords[labels[y,x]] = np.column_stack((x,y))
    return edges, coords[labels]


def make_candidates(nearest, pose):
    """Restricted beta (Eq13): image-edge chains snapped from ordered proposals.

    Durations 6/8, three offset proposals per state. Different points can snap
    to the same edge pixel: uniqueness is NOT enforced. Pose is a search
    hypothesis, never evaluator truth. Emissions use pose-conditioned spatial
    residuals; no learned appearance/orientation classifier is used.
    """
    candidates = {s:[] for s in GRAPH}
    g = geometry()
    for key,curve in g.items():
        state = 20 if isinstance(key,str) else key
        extended20 = key == '20E'
        for duration in (6,8):
            canonical = resample_arc(curve,duration)
            expected = transform(canonical,pose)
            if state in (5,6,7,9,10,11,13,14,15,17,18):
                weights = np.full(duration,1/3)
            else:
                weights = np.ones(duration)
            if extended20:
                # Normalize the palm and shared little-side subsegments separately.
                is_palm = canonical[:,1] >= 140
                weights = np.where(is_palm, 1/max(1,is_palm.sum()),
                                   (1/3)/max(1,(~is_palm).sum()))*duration
            for dx in (-2,0,2):
                proposal = expected+[dx,0]
                xy = np.rint(proposal).astype(int)
                xy[:,0] = np.clip(xy[:,0],0,nearest.shape[1]-1)
                xy[:,1] = np.clip(xy[:,1],0,nearest.shape[0]-1)
                points = nearest[xy[:,1],xy[:,0]]
                residual = np.sum((points-expected)**2,axis=1)
                emission = float(np.mean(weights*residual)/16)
                internal = float(.01*np.mean(np.sum((np.diff(points,axis=0)-
                                                     np.diff(expected,axis=0))**2,axis=1)))
                # Duration cost is state-balanced too; no extended-path length bonus.
                duration_cost = float(.02*(duration-8)**2*np.mean(weights))
                candidates[state].append(Segment(state,points,emission+internal+duration_cost,
                    duration,expected[0],expected[-1],
                    (18,) if extended20 else ((19,) if key == '20H' else ()),
                    str(key),dict(emission=emission,internal=internal,duration=duration_cost,
                                  palm_indices=np.flatnonzero(canonical[:,1]>=140).tolist())))
    return candidates


def pose_grid(condition):
    if condition == 'translation':
        return [(dx,dy,0,1) for dx,dy in product(range(-24,25,8),repeat=2)]
    if condition == 'rotation':
        return [(0,0,a,1) for a in (-30,-15,0,15,30)]
    if condition == 'scale':
        return [(0,0,0,s) for s in (.75,.875,1,1.125,1.25)]
    if condition in ('fixed','clutter'):
        return [(0,0,0,1)]
    raise ValueError(condition)


def infer(mask, condition):
    """Public predictor: only image and declared experimental search condition."""
    started = perf_counter()
    edges,nearest = edge_lookup(mask)
    best = None
    for pose in pose_grid(condition):
        result = decode(GRAPH,make_candidates(nearest,pose))
        if best is None or result['cost'] < best['cost']:
            best = result
            best['pose'] = pose
    states = best['states']
    best['bits'] = tuple(int(s in states) for s in (5,9,13,17))
    best['runtime_seconds'] = perf_counter()-started
    best['edge_count'] = int(np.count_nonzero(edges))
    return best


def palm_coverage(truth, prediction, tolerance=TOLERANCE):
    """Directional true-to-predicted coverage, pooled equal samples per palm side."""
    gt = np.vstack([resample_arc(p,200) for p in truth])
    predicted = np.vstack([resample_arc(p,200) for p in prediction])
    distances = np.min(np.sum((gt[:,None,:]-predicted[None,:,:])**2,axis=2),axis=1)
    return float(np.mean(distances <= tolerance**2))


def predicted_palm(result):
    right = result['segments'][0].points
    left_segment = result['segments'][-1]
    # Recover the palm start on the registered state20 polyline; no GT supplied.
    if left_segment.variant == '20E':
        ids = left_segment.terms['palm_indices']
        # Include preceding sample to cover the transition near palm top.
        first = max(0,ids[0]-1) if ids else len(left_segment.points)-2
        left = left_segment.points[first:]
    else:
        left = left_segment.points
    return [right,left]


def actual_pose(condition,rng):
    if condition == 'translation':
        return (float(rng.uniform(-23,23)),float(rng.uniform(-23,23)),0,1)
    if condition == 'rotation':
        return (0,0,float(rng.uniform(-28,28)),1)
    if condition == 'scale':
        return (0,0,0,float(rng.uniform(.78,1.22)))
    return (0,0,0,1)


def run(output, per_configuration=2):
    if per_configuration < 1:
        raise ValueError('per_configuration must be positive')
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    (output/'graph.json').write_text(json.dumps(dict(mapping=STATE_MAPPING,
        transitions=GRAPH,traces=trace_configurations()),indent=2)+'\n')
    all_rows, summaries = [], {}
    first_failure = None
    for ci,condition in enumerate(CONDITIONS):
        rows = []; condition_output = output/condition
        condition_output.mkdir(exist_ok=True)
        failure_saved = False
        for config_id,bits in enumerate(CONFIGURATIONS):
            for repeat in range(per_configuration):
                # Same held-out shape/seed per configuration across conditions.
                seed = 20000+config_id*100+repeat
                rng = np.random.default_rng(seed+500000)
                pose = actual_pose(condition,rng)
                sample = render_hand(bits,seed,pose,clutter=condition=='clutter')
                result = infer(sample['mask'],condition)
                coverage = palm_coverage(sample['palm'],predicted_palm(result))
                localized = coverage >= .75
                exact = result['bits'] == bits
                row = dict(sample_id=f'{condition}_{config_id:02d}_{seed}',split='heldout',
                    seed=seed,condition=condition,true_bits=''.join(map(str,bits)),
                    predicted_bits=''.join(map(str,result['bits'])),
                    true_dx=pose[0],true_dy=pose[1],true_angle_degrees=pose[2],true_scale=pose[3],
                    predicted_dx=result['pose'][0],predicted_dy=result['pose'][1],
                    predicted_angle_degrees=result['pose'][2],predicted_scale=result['pose'][3],
                    palm_coverage=coverage,localized=int(localized),exact_configuration=int(exact),
                    cost=result['cost'],runtime_seconds=result['runtime_seconds'],
                    state_path=' '.join(map(str,result['states'])),edge_count=result['edge_count'])
                for f,truth,pred in zip(FINGERS,bits,result['bits']):
                    row[f+'_correct'] = int(truth==pred)
                rows.append(row)
                failed = not localized or not exact
                save_overlay = (config_id == 0 and repeat == 0) or (failed and not failure_saved)
                # Both constrained structure costs at the chosen pose; other fingers free.
                if save_overlay or (repeat == 0 and config_id in (0,15)):
                    _,nearest = edge_lookup(sample['mask'])
                    original = make_candidates(nearest,result['pose'])
                    for f,extended,hidden in zip(FINGERS,(5,9,13,17),(8,12,16,19)):
                        costs = []
                        for bit in (0,1):
                            constrained = {s:([] if s == (hidden if bit else extended) else c)
                                           for s,c in original.items()}
                            costs.append(decode(GRAPH,constrained)['cost'])
                        row[f+'_cost_hidden'],row[f+'_cost_extended'] = costs
                if save_overlay:
                    name = 'failure' if failed else 'example'
                    path = condition_output/f'overlay_{name}.png'
                    write_overlay(sample['mask'],sample['truth_curves'],
                        [s.points for s in result['segments']],
                        [row['sample_id'],f'truth={row["true_bits"]} predicted={row["predicted_bits"]}',
                         f'palm coverage={coverage:.3f}; localized={localized}; cost={result["cost"]:.3f}',
                         'bits: index middle ring little; 1=extended',
                         'Failure means localization or configuration failed'] +
                        [f'{f}: hidden={row[f+"_cost_hidden"]:.3f} extended={row[f+"_cost_extended"]:.3f}'
                         for f in FINGERS],path)
                    if failed:
                        failure_saved = True
                        if first_failure is None:
                            first_failure = str(path)
                cv2.imwrite(str(condition_output/f'{row["sample_id"]}.png'),sample['mask'])
                # Independent ground truth and decoded chains for reproducibility/auditing.
                (condition_output/f'{row["sample_id"]}.json').write_text(json.dumps(dict(
                    seed=seed,true_bits=bits,true_pose=pose,
                    truth_palm=[p.tolist() for p in sample['palm']],
                    truth_curves=[p.tolist() for p in sample['truth_curves']],
                    predicted_pose=result['pose'],states=result['states'],
                    segments=[dict(state=s.state,duration=s.duration,points=s.points.tolist(),
                        variant=s.variant,terms=s.terms) for s in result['segments']]),indent=2)+'\n')
        localized_rows = [r for r in rows if r['localized']]
        summaries[condition] = dict(examples=len(rows),
            localized=sum(r['localized'] for r in rows),
            palm_localization_accuracy=float(np.mean([r['localized'] for r in rows])),
            mean_palm_coverage=float(np.mean([r['palm_coverage'] for r in rows])),
            exact_configuration_accuracy=float(np.mean([r['exact_configuration'] for r in rows])),
            exact_configuration_on_localized=(float(np.mean([r['exact_configuration'] for r in localized_rows]))
                                             if localized_rows else None),
            per_finger_accuracy={f:float(np.mean([r[f+'_correct'] for r in rows])) for f in FINGERS},
            mean_runtime_seconds=float(np.mean([r['runtime_seconds'] for r in rows])),
            failure_overlay=str(condition_output/'overlay_failure.png') if failure_saved else None)
        all_rows.extend(rows)
        print(condition,json.dumps(summaries[condition]),flush=True)
    fields = list(dict.fromkeys(k for r in all_rows for k in r))
    with (output/'scores.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(all_rows)
    summary = dict(split='heldout synthetic',per_configuration=per_configuration,
        configurations=16,conditions=summaries,palm_tolerance_pixels=TOLERANCE,
        palm_required_coverage=.75,first_failure_overlay=first_failure,
        reference='fixed analytic geometry; no held-out fitting',
        protocol='restricted segmental HSSM adaptation, not paper reproduction')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/milestone2')
    parser.add_argument('--per-configuration',type=int,default=2)
    args = parser.parse_args()
    run(args.output,args.per_configuration)


if __name__ == '__main__':
    main()
