"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/plugins/stable/guides.html#readers
"""
from tkinter import N
import numpy as np
from skimage.io import imread
import json
from .utils import *
from .config import CONFIG


def read_pram_image(path):
    data = imread(path)
    return [(data, {"name": CONFIG.layer_name_pram_img}, "image")]

def read_vgg_annot(path):
    data = read_annot_file(path)
    data = data[:,[1,0]]
    return [(data, {"name": CONFIG.layer_name_label, "size": CONFIG.point_size, "edge_color": CONFIG.label_color, "face_color": CONFIG.point_face_color}, "points")]

def read_napari_pram_annot(path):
    out_list = []
    with open(path) as fi:
        data = json.load(fi)
        if CONFIG.json_labels in data:
            d = np.array(data[CONFIG.json_labels])
            out_list.append(
               (d, 
                    {"name": CONFIG.layer_name_label, 
                    "size": CONFIG.point_size, 
                    "edge_color": CONFIG.label_color, 
                    "face_color": CONFIG.point_face_color}, 
               "points"))
        if CONFIG.json_preds in data:
            d = np.array(data[CONFIG.json_preds])
            out_list.append(
               (d, 
                    {"name": CONFIG.layer_name_preds, 
                    "size": CONFIG.point_size, 
                    "edge_color": CONFIG.pred_color, 
                    "face_color": CONFIG.point_face_color}, 
               "points"))
    return out_list
def reader_pram_image(path):
    return read_pram_image

def reader_vgg_annot(path):
    if not path.endswith(".json"):
        return None
    # Read json from VGG annotator
    with open(path) as fi:
        data = json.load(fi)
        if "img_name" in data:
            return read_napari_pram_annot
        else:
            return read_vgg_annot