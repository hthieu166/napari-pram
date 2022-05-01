from napari import layers 
from skimage.io import imread
from qtpy.QtWidgets import QWidget, QFileDialog
from magicgui.widgets import PushButton, Slider, Label, FileEdit, Container
import os.path as osp
import os
from .utils import *
from .pram import pram_detect
from .config import CONFIG

def plot_circles(layer, data, color):
    layer.data = data  
    layer.events.set_data()
    layer.selected_data = list(range(len(layer.data)))
    layer.size = CONFIG.point_size
    layer.edge_color = color
    layer.face_color = "transparent"
    layer.selected_data = []
    layer.refresh()


class PramQWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.v = napari_viewer
        self.c_controller = Container(layout = "horizontal", widgets= [
            Slider(value =2, min=1, max=10, label = "Contrast Threshold", name="sld_thrs"),
            PushButton(value=True, text="Annotate",    name="btn_annotate"),
            PushButton(value=True, text="Run Detector",name="btn_detect"),
            PushButton(value=True, text="Evaluate",    name="btn_eval"),
            PushButton(value=True, text="Clear All",   name="btn_clear"),
            PushButton(value=True, text="Save to File",name="btn_save"),
            Label(value="--.--%", label="Precision: ", name="lbl_prec"),
            Label(value="--.--%", label="Recall: "   , name="lbl_recall"),
            ])
        # self.c_file_manager = Container(layout="horizontal", widgets=[
        #     FileEdit(value="dataset/", label="Image File: ",      name="txt_img_file",  filter="*.png"),
        #     FileEdit(value="dataset/", label="Annotation File: ", name="txt_annt_file", filter="*.json"),
        # ])
        container = Container(widgets=[
            # self.c_file_manager, 
            self.c_controller], labels=False)
        self.v.window.add_dock_widget(container, area= "bottom")
        
        # self.c_file_manager.txt_img_file.changed.connect(self.load_img_file)
        # self.c_file_manager.txt_annt_file.changed.connect(self.load_annot_file)
        self.c_controller.btn_eval.clicked.connect(self.run_evaluate)
        self.c_controller.btn_detect.clicked.connect(self.run_pram_detect)
        self.c_controller.btn_clear.clicked.connect(self.clear_window)
        self.c_controller.btn_annotate.clicked.connect(self.annotate)
        self.c_controller.btn_save.clicked.connect(self.save)
        
    def run_evaluate(self):
        if CONFIG.layer_name_preds not in self.v.layers:
            pass
        # Hide prediction layer
        self.v.layers[CONFIG.layer_name_preds].hidden = True
        preds = self.v.layers[CONFIG.layer_name_preds].data
        gts   = self.v.layers[CONFIG.layer_name_label].data
        self.prec, self.recall, tps, fns = eval_pred(gts, preds)
        self.tps = tps
        self.fns = fns
        self.c_controller.lbl_prec.value  = "%.02f" % (self.prec * 100)   + "%"
        self.c_controller.lbl_recall.value= "%.02f" % (self.recall * 100) + "%"
        
        if CONFIG.layer_name_tp not in self.v.layers:
            self.v.add_layer(layers.Points(name = CONFIG.layer_name_tp))
            self.v.add_layer(layers.Points(name = CONFIG.layer_name_fp))
            self.v.add_layer(layers.Points(name = CONFIG.layer_name_fn))
            self.v.layers[CONFIG.layer_name_tp].editable = False
        
        plot_circles(self.v.layers[CONFIG.layer_name_tp],preds[tps] , CONFIG.tp_color)
        plot_circles(self.v.layers[CONFIG.layer_name_fp],preds[~tps], CONFIG.fp_color)
        plot_circles(self.v.layers[CONFIG.layer_name_fn],gts[fns],    CONFIG.fn_color)      

    def run_pram_detect(self):
        img_data = self.v.layers[CONFIG.layer_name_pram_img].data.copy()
        thr_ctrs = self.c_controller.sld_thrs.value
        preds = pram_detect(img_data, thr_ctrs)
        preds = preds[:,[1,0]]
        if CONFIG.layer_name_preds not in self.v.layers:
            self.v.add_layer(layers.Points(name = CONFIG.layer_name_preds))
        plot_circles(self.v.layers[CONFIG.layer_name_preds], preds, "red")

    def load_annot_file(self, file):
        annt_pts = read_annot_file(file)
        annt_pts = annt_pts[:,[1,0]]
        if CONFIG.layer_name_label not in self.v.layers:
            self.v.add_layer(layers.Points(name = CONFIG.layer_name_label))
        plot_circles(self.v.layers[CONFIG.layer_name_label], annt_pts, "blue")

    def load_img_file(self, file):
        img_data = imread(file)
        self.v.add_image(img_data, name = CONFIG.layer_name_pram_img)

    def annotate(self):
        self.v.add_layer(layers.Points(name = CONFIG.layer_name_label, face_color = CONFIG.point_face_color,
                                       edge_color = CONFIG.label_color, size = CONFIG.point_size))
    
    def clear_window(self):
        self.c_controller.lbl_prec.value  = "--.--%" 
        self.c_controller.lbl_recall.value= "--.--%"
        while len(self.v.layers) != 0:
            self.v.layers.pop(0)

    def save(self):
        inp_path    = self.v.layers[CONFIG.layer_name_pram_img].source.path
        outFile, _  = QFileDialog.getSaveFileName(self,caption = "Save to JSON file", 
        directory = osp.dirname(inp_path), filter= "JSON files (*.json)")
        data  = {
            "img_name"      : osp.basename(inp_path)    
            }
        if CONFIG.layer_name_preds in self.v.layers:
            data[CONFIG.json_preds]  = self.v.layers[CONFIG.layer_name_preds].data.tolist()
        if CONFIG.layer_name_label in self.v.layers:
            data[CONFIG.json_labels] = self.v.layers[CONFIG.layer_name_label].data.tolist()
        if CONFIG.layer_name_tp    in self.v.layers:
            data[CONFIG.json_prec]   = self.prec
            data[CONFIG.json_recall] = self.recall
            data[CONFIG.json_tp]     = self.tps.tolist()
            data[CONFIG.json_fn]     = self.fns.tolist()
        with open(outFile, "w") as fo:
            json.dump(data, fo)