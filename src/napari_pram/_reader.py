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

def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]

    # if we know we cannot read the file, we immediately return None.
    if not path.endswith(".npy"):
        return None

    # otherwise we return the *function* that can read ``path``.
    return reader_function


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

def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of layer.
        Both "meta", and "layer_type" are optional. napari will default to
        layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    paths = [path] if isinstance(path, str) else path
    return None 
    # open_layers = []
    # for path in paths:
    # # optional kwargs for the corresponding viewer.add_* method
    # add_kwargs = {"name": "PRAM Image"}
    # layer_type = "image"  # optional, default is "image"
    # return [(data, add_kwargs, layer_type)]
