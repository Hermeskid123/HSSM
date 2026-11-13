import cv2
import numpy as np
import pytest
from hssm.milestone1 import LABELS, render, external_contour, select_arc, normalize, predict, evaluate


def refs():
    return {label: normalize(select_arc(external_contour(mask),endpoints))
            for label,seed in zip(LABELS,(11,12))
            for mask,endpoints in [render(label,seed)]}


@pytest.mark.parametrize('label,seed', [('extended',11),('folded',12)])
def test_self_match(label,seed):
    mask,endpoints = render(label,seed)
    prediction,scores,_ = predict(external_contour(mask),endpoints,refs())
    assert prediction == label
    assert scores[label] < 1e-20


@pytest.mark.parametrize('label',LABELS)
def test_traversal_and_contour_start_invariance(label):
    mask,endpoints = render(label,999)
    contour = external_contour(mask)
    expected, scores, arc = predict(contour,endpoints,refs())
    for transformed in (contour[::-1],np.roll(contour,37,axis=0)):
        actual,other,selected = predict(transformed,endpoints,refs())
        assert actual == expected
        np.testing.assert_allclose(selected,arc)
        np.testing.assert_allclose(list(other.values()),list(scores.values()))
    assert np.max(arc[:,0]) <= endpoints[1,0]+1
    assert np.min(arc[:,0]) >= endpoints[0,0]-1
    assert np.max(arc[:,1]) <= 115


def test_registration_preserves_height_and_similarity():
    arc = np.array([[0,0],[0,-2],[1,-2],[1,0]],float)
    np.testing.assert_allclose(normalize(arc),normalize(arc*3+[17,21]),atol=1e-14)
    assert normalize(arc)[:,1].min() < -1.9


def test_heldout_and_saved_evidence(tmp_path):
    summary = evaluate(tmp_path,10)
    assert summary['examples'] == 20
    assert summary['confusion_matrix'] == [[10,0],[0,10]]
    assert len((tmp_path/'scores.csv').read_text().splitlines()) == 21
    for label in LABELS:
        assert cv2.imread(str(tmp_path/f'overlay_{label}.png')) is not None


def test_render_repeatability_and_pose_difference():
    a,endpoints = render('extended',123)
    b,_ = render('extended',123)
    folded,_ = render('folded',123)
    np.testing.assert_array_equal(a,b)
    assert np.count_nonzero(a) > np.count_nonzero(folded)
    assert set(np.unique(a)) == {0,255}
