from napari import layers 
from skimage.io import imread
from qtpy.QtWidgets import QWidget
from magicgui.widgets import PushButton, Slider, Label, FileEdit, Container

from .utils import *
from .pram import pram_detect

def plot_circles(layer, data, color):
    layer.data = data  
    layer.events.set_data()
    layer.selected_data = list(range(len(layer.data)))
    layer.size = 30
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
            PushButton(value=True, text="Run Detector",name="btn_detect"),
            Label(value="--.--%", label="Precision: ", name="lbl_prec"),
            Label(value="--.--%", label="Recall: "   , name="lbl_recall"),
            PushButton(value=True, text="Evaluate",    name="btn_eval")
            ])

        self.c_file_manager = Container(layout="horizontal", widgets=[
            FileEdit(value="dataset/", label="Image File: ",      name="txt_img_file",  filter="*.png"),
            FileEdit(value="dataset/", label="Annotation File: ", name="txt_annt_file", filter="*.json"),
        ])

        container = Container(widgets=[self.c_file_manager, self.c_controller], labels=False)
        self.v.window.add_dock_widget(container, area= "bottom")
        
        self.c_file_manager.txt_img_file.changed.connect(self.load_img_file)
        self.c_file_manager.txt_annt_file.changed.connect(self.load_annot_file)
        self.c_controller.btn_eval.clicked.connect(self.run_evaluate)
        self.c_controller.btn_detect.clicked.connect(self.run_pram_detect)
    
    def run_evaluate(self):
        if "Predictions" not in self.v.layers:
            pass
        # Hide prediction layer
        self.v.layers["Predictions"].hidden = True
        preds = self.v.layers["Predictions"].data
        gts   = self.v.layers["Labels"].data
        prec, recall, tps, fns = eval_pred(gts, preds)
        self.c_controller.lbl_prec.value  = "%.02f" % (prec * 100)
        self.c_controller.lbl_recall.value= "%.02f" % (recall * 100)
        
        if "True Positive" not in self.v.layers:
            self.v.add_layer(layers.Points(name = "True Positive"))
            self.v.add_layer(layers.Points(name = "False Positive"))
            self.v.add_layer(layers.Points(name = "False Negative"))
        
        plot_circles(self.v.layers["True Positive"],preds[tps] , "green")
        plot_circles(self.v.layers["False Positive"],preds[~tps], "yellow")
        plot_circles(self.v.layers["False Negative"],gts[fns],    "red")

    def run_pram_detect(self):
        img_data = self.v.layers["PRAM Image"].data.copy()
        thr_ctrs = self.c_controller.sld_thrs.value
        preds = pram_detect(img_data, thr_ctrs)
        preds = preds[:,[1,0]]
        if "Predictions" not in self.v.layers:
            self.v.add_layer(layers.Points(name = "Predictions"))
        plot_circles(self.v.layers["Predictions"], preds, "red")

    def load_annot_file(self, file):
        annt_pts = read_annot_file(file)
        annt_pts = annt_pts[:,[1,0]]
        if "Labels" not in self.v.layers:
            self.v.add_layer(layers.Points(name = "Labels"))
        plot_circles(self.v.layers["Labels"], annt_pts, "blue")

    def load_img_file(self, file):
        img_data = imread(file)
        self.v.add_image(img_data, name = "PRAM Image")


# def setup_panel():
#     container = Container(widgets=[c_file_manager, c_controller], labels=False)
#     v.window.add_dock_widget(container, area= "bottom")

# if __name__ == "__main__":
#     # start the event loop and show the viewer
#     napari.run()