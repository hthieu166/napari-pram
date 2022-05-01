import json
import numpy as np
from scipy.spatial import distance

def read_annot_file(json_file):
    """"Read annotations from CSV file"""
    with open(json_file, "r") as fi:
        annot    = json.load(fi)
        img_ids  = list(annot.keys())
        pts_annt = annot[img_ids[0]]['regions']
        pts_arr  = []
        for i, pnt in enumerate(pts_annt):
            cx = pts_annt[pnt]['shape_attributes']['cx']
            cy = pts_annt[pnt]['shape_attributes']['cy']
            pts_arr.append([cx, cy])
    return np.array(pts_arr)

def eval_pred(gt, preds, thr_dst = 20):
    dst = distance.cdist(gt, preds)
    matches = dst < thr_dst
    tps     = matches.any(axis=0)
    fns     = 1-matches.any(axis=1)
    tp = tps.sum()
    fn = fns.sum()
    fp = len(preds) - tp
    recall= tp/(tp + fn)
    prec  = tp/(tp + fp)
    return prec, recall, tps.astype(np.bool), fns.astype(np.bool)