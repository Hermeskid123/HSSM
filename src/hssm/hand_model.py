"""Figure 3 topology and synthetic boundary geometry, in (x,y) pixels.

Hand is oriented as Fig. 3 (thumb at image right). This is a synthetic polygonal
adaptation; no paper-trained emissions or duration statistics are available.
"""
from itertools import product
import cv2
import numpy as np
from .contours import resample_arc
from .milestone1 import external_contour

FINGERS = ('index', 'middle', 'ring', 'little')
# Fig. 3: start s1; thumb s2,s3,s4 fixed extended. Hidden-thumb shortcuts
# 1->5/8 are deliberately excluded. Extended little uses 17,18,20, NOT 17,18,19.
GRAPH = {1:(2,), 2:(3,), 3:(4,), 4:(5,8), 5:(6,), 6:(7,),
         7:(9,12), 8:(9,12), 9:(10,), 10:(11,), 11:(13,16),
         12:(13,16), 13:(14,), 14:(15,), 15:(17,19), 16:(17,19),
         17:(18,), 18:(20,), 19:(20,), 20:()}
STATE_MAPPING = {1:'right palm',2:'thumb outer side',3:'thumb tip',4:'thumb inner side',
                 5:'index outer side',6:'index tip',7:'index inner side',8:'index hidden boundary',
                 9:'middle outer side',10:'middle tip',11:'middle inner side',12:'middle hidden boundary',
                 13:'ring outer side',14:'ring tip',15:'ring inner side',16:'ring hidden boundary',
                 17:'little outer side',18:'little tip',19:'little hidden boundary',
                 20:'left palm plus little inner side when extended'}
CONFIGURATIONS = list(product((0,1), repeat=4))  # 1 = extended, 0 = hidden


def state_path(bits):
    path = [1,2,3,4]
    for bit, extended, hidden in zip(bits, ((5,6,7),(9,10,11),(13,14,15),(17,18)), (8,12,16,19)):
        path.extend(extended if bit else (hidden,))
    return path+[20]


def trace_configurations():
    traces = []
    for bits in CONFIGURATIONS:
        path = state_path(bits)
        assert path[0] == 1 and path[-1] == 20
        assert all(b in GRAPH[a] for a,b in zip(path,path[1:]))
        traces.append(dict(configuration=list(bits),states=path))
    return traces


def geometry(jitter=None):
    """Return every possible state boundary in canonical coordinates.

    Two realizations of state20 preserve its shared finger/palm role; the
    decoder restricts them to predecessors18 and19 respectively.
    """
    jitter = np.zeros(4) if jitter is None else np.asarray(jitter)
    g = {1:np.array([[195,215],[195,150]],float),
         2:np.array([[195,150],[225,110]],float),
         3:np.array([[225,110],[220,100],[215,100]],float),
         4:np.array([[215,100],[180,130]],float)}
    starts = [(180,130),(160,125),(130,125),(100,130)]
    outer = [180,150,120,90]; inner = [160,130,100,70]
    bases = [125,125,130,140]; tops = [55,40,55,80]
    for f,(start,xo,xi,base,top) in enumerate(zip(starts,outer,inner,bases,tops)):
        s = 5+4*f
        top += jitter[f]
        g[s] = np.array([start,[xo,start[1]],[xo,top+5]],float)
        g[s+1] = np.array([[xo,top+5],[xo-4,top],[xi+4,top],[xi,top+5]],float)
        if f < 3:
            g[s+2] = np.array([[xi,top+5],[xi,base]],float)
        g[(8,12,16,19)[f]] = np.array([start,[(xo+xi)/2,start[1]-5],[xi,base]],float)
    g['20E'] = np.array([[70,tops[3]+jitter[3]+5],[70,140],[65,215]],float)
    g['20H'] = np.array([[70,140],[65,215]],float)
    return g


def transform(points, pose):
    """pose=(dx pixels,dy pixels,angle degrees,uniform scale), center=(145,145)."""
    dx,dy,degrees,scale = pose
    angle = np.deg2rad(degrees)
    rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    return (np.asarray(points)-[145,145]) @ rotation.T*scale+[145+dx,145+dy]


def render_hand(bits, seed, pose=(0,0,0,1), clutter=False):
    """Generate observation and independent evaluator-only state/palm annotations."""
    rng = np.random.default_rng(seed)
    g = geometry(rng.uniform(-12,12,4))
    curves = [transform(g[('20E' if bits[3] else '20H') if s == 20 else s],pose)
              for s in state_path(bits)]
    mask = np.zeros((300,300),np.uint8)
    polygon = np.vstack(curves)
    cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],255)
    # Reuse milestone1's connected-silhouette validation before adding clutter.
    external_contour(mask)
    clean = mask.copy()
    if clutter:
        # Reproducible foreground strokes can merge with/occlude hand edges.
        for _ in range(28):
            a = rng.integers(25,275,2); b = a+rng.integers(-65,66,2)
            cv2.line(mask,tuple(a),tuple(b),255,int(rng.integers(1,5)))
    palm = [transform(g[1],pose),transform(np.array([[70,140],[65,215]]),pose)]
    return dict(mask=mask,clean=clean,truth_curves=curves,palm=palm,
                bits=tuple(bits),seed=seed,pose=pose)
