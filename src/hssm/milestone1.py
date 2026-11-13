"""Fixed-view synthetic diagnostic inspired by Wang et al. (2008), Fig. 3.

Known endpoints, synthetic rendering, and ordered arc MSE are project adaptations.
This does not implement the probabilistic model or registration in Sections 4–6.
Coordinates are (x,y) pixels; arrays use [row,column]. No angular parameters.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

LABELS = ('extended', 'folded')


def render(label, seed):
    """Return silhouette and deliberate known endpoints, independent of evaluation."""
    if label not in LABELS:
        raise ValueError(label)
    rng = np.random.default_rng(seed)
    width = int(rng.integers(20, 29))
    left, base = 100-width//2, 115
    right = left+width
    height = int(rng.integers(56, 81) if label == 'extended' else rng.integers(4, 13))
    # Fixed palm, thumb and three fingers. Only index boundary varies.
    mask = np.zeros((256, 256), np.uint8)
    outline = [(65,215),(65,160),(38,138),(33,115),(45,111),
               (65,131),(65,115),(85,115),(115,115),(123,115),
               (123,49),(129,40),(138,40),(145,49),(145,115),
               (153,115),(153,61),(160,53),(168,53),(175,61),
               (175,126),(182,126),(182,89),(189,82),(198,86),
               (202,96),(202,166),(190,215)]
    cv2.fillPoly(mask, [np.array(outline, np.int32)], 255)
    t = np.linspace(0, 1, 41)
    # Smooth arch with independent asymmetric boundary variation, fixed endpoints.
    x = left + (right-left)*t + (width-25)*0.16*np.sin(2*np.pi*t)
    y = base - height*np.sin(np.pi*t)**0.55
    y += rng.uniform(-2,2)*np.sin(2*np.pi*t) + rng.uniform(-1,1)*np.sin(4*np.pi*t)
    finger = np.vstack((np.column_stack((x,y)), [right,base], [left,base]))
    cv2.fillPoly(mask, [np.rint(finger).astype(np.int32)], 255)
    return mask, np.array([[left,base],[right,base]], dtype=float)


def external_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 1:
        raise ValueError('Expected one connected external contour')
    return contours[0][:,0,:].astype(float)


def select_arc(contour, endpoints):
    """Restore endpoint order; choose arc staying in the index's x corridor.

    The complementary arc visits the fixed palm/other fingers. This geometry
    rule is identical for both poses and does not consume a class label.
    """
    contour = np.asarray(contour, dtype=float)
    endpoints = np.asarray(endpoints, dtype=float)
    # Break nearest-pixel ties by coordinates, independent of contour traversal.
    indices = [int(np.lexsort((contour[:,1], contour[:,0],
                              np.sum((contour-p)**2, axis=1)))[0]) for p in endpoints]
    if indices[0] == indices[1]:
        raise ValueError('Endpoints must map to distinct contour points')
    n = len(contour)
    arcs = [contour[(indices[0] + step*np.arange((step*(indices[1]-indices[0])) % n+1)) % n]
            for step in (1,-1)]
    low, high = np.sort(endpoints[:,0])
    def penalty(arc):
        return float(np.mean(np.maximum(low-arc[:,0],0)**2 + np.maximum(arc[:,0]-high,0)**2))
    return min(arcs, key=penalty)


def normalize(arc, samples=128):
    """Equal arc-length samples in endpoint frame; retain height/base-width."""
    arc = np.asarray(arc, dtype=float)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(arc,axis=0),axis=1))]
    if samples < 2 or distance[-1] == 0:
        raise ValueError('Need a nonzero arc and at least two samples')
    keep = np.r_[True, np.diff(distance)>0]
    points = np.column_stack([np.interp(np.linspace(0,distance[-1],samples),distance[keep],arc[keep,i]) for i in (0,1)])
    delta = arc[-1]-arc[0]
    width = np.linalg.norm(delta)
    if width == 0:
        raise ValueError('Coincident endpoints')
    u = delta/width
    v = np.array([-u[1],u[0]])
    return (points-arc[0]) @ np.column_stack((u,v))/width


def predict(contour, endpoints, references):
    """Predict using only observed geometry, known endpoints and reference arcs."""
    arc = select_arc(contour,endpoints)
    observed = normalize(arc)
    scores = {label: float(np.mean(np.sum((observed-ref)**2,axis=1))) for label,ref in references.items()}
    return min(scores,key=scores.get), scores, arc


def overlay(mask, truth_arc, predicted_reference, endpoints, title, path):
    canvas = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    canvas = cv2.copyMakeBorder(canvas,0,90,0,360,cv2.BORDER_CONSTANT,value=(245,245,245))
    cv2.polylines(canvas,[np.rint(truth_arc).astype(np.int32)],False,(255,0,0),3)
    delta = endpoints[1]-endpoints[0]
    u = delta/np.linalg.norm(delta)
    mapped = predicted_reference @ np.column_stack((u,[-u[1],u[0]])).T*np.linalg.norm(delta)+endpoints[0]
    cv2.polylines(canvas,[np.rint(mapped).astype(np.int32)],False,(0,0,255),1)
    for p in endpoints:
        cv2.circle(canvas,tuple(np.rint(p).astype(int)),4,(0,180,0),-1)
    for i,(text,color) in enumerate([(title,(0,0,0)),('Blue: ground-truth index boundary',(255,0,0)),('Red: predicted reference match',(0,0,255)),('Green: known endpoints',(0,150,0))]):
        cv2.putText(canvas,text,(10,275+20*i),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1,cv2.LINE_AA)
    if not cv2.imwrite(str(path),canvas):
        raise OSError(path)


def evaluate(output, per_class=100):
    if per_class < 1:
        raise ValueError('per_class must be positive')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    references = {}
    for label,seed in zip(LABELS,(11,12)):
        mask, endpoints = render(label,seed)
        references[label] = normalize(select_arc(external_contour(mask),endpoints))
        cv2.imwrite(str(output/f'reference_{label}.png'),mask)
    np.savez(output/'references.npz',**references)
    rows, matrix = [], np.zeros((2,2),int)
    failure_saved = False
    for class_id,label in enumerate(LABELS):
        for i in range(per_class):
            seed = 10000+class_id*100000+i
            sample_id = f'heldout_{label}_{seed}'
            mask,endpoints = render(label,seed)
            prediction,scores,arc = predict(external_contour(mask),endpoints,references)
            matrix[class_id,LABELS.index(prediction)] += 1
            cv2.imwrite(str(output/f'{sample_id}.png'),mask)
            rows.append(dict(sample_id=sample_id,split='heldout',seed=seed,true_label=label,prediction=prediction,
                             cost_extended=scores['extended'],cost_folded=scores['folded'],
                             endpoint_0_x=endpoints[0,0],endpoint_0_y=endpoints[0,1],
                             endpoint_1_x=endpoints[1,0],endpoint_1_y=endpoints[1,1]))
            failed = prediction != label
            if i == 0 or (failed and not failure_saved):
                name = 'failure' if failed else label
                overlay(mask,arc,references[prediction],endpoints,
                        f'{sample_id}: {prediction}',output/f'overlay_{name}.png')
                failure_saved |= failed
    with (output/'scores.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = dict(split='heldout',examples=len(rows),labels=list(LABELS),confusion_matrix=matrix.tolist(),
                   accuracy=float(np.trace(matrix)/matrix.sum()),reference_seeds=[11,12],
                   heldout_seed_ranges=[[10000,10000+per_class-1],[110000,110000+per_class-1]],
                   failures=int(matrix.sum()-np.trace(matrix)),protocol='synthetic project adaptation; not paper reproduction')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/milestone1')
    parser.add_argument('--per-class',type=int,default=100)
    args = parser.parse_args()
    evaluate(args.output,args.per_class)


if __name__ == '__main__':
    main()
