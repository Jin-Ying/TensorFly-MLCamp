from __future__ import division

import cv2
import torch
import sys


from PyQt5 import QtWidgets, QtGui
from pet import Ui_MainWindow
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from detection.models import *
from detection.utils.utils import *
from detection.utils.datasets import *
import qdarkstyle

from reid.network import *
from reid.find_closest_index import find_closest_index

sys.path.append("reid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class QLabel_alterada(QLabel):
    clicked=pyqtSignal()
    def __init__(self, parent=None):
        QLabel.__init__(self, parent)

    def mousePressEvent(self, ev):
        self.clicked.emit()


class MyMainForm(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.upload_img)

        self.scroVertLayout = QtWidgets.QVBoxLayout()
        self.scrollAreaWidgetContents.setLayout(self.scroVertLayout)

        self.reids_ql = []
        self.resize(844, 555)

        # detection model config
        self.model_def = "./detection/config/yolov3-custom.cfg"
        self.pretrained_weights = "./detection/checkpoints/yolov3_ckpt_99.pth"
        self.class_path = "./detection/data/custom/classes.names"
        self.img_size = 416

        self.model = Darknet(self.model_def).to(device)
        self.model.apply(weights_init_normal)
        self.model.load_state_dict(torch.load(self.pretrained_weights, map_location=torch.device('cpu')))
        self.model.eval()
        self.classes = load_classes(self.class_path)

        self.conf_thres = 0.8
        self.nms_thres = 0.4

        # reid model config
        self.reid_feat_model = torch.load("./reid/snapshot/finetune_pet/iter_01000_model.pth.tar", map_location=torch.device('cpu'))[0]
        self.reid_cls_model = torch.load(os.path.join("./reid/snapshot/reid_bulldog_new", "iter_05000_classifier.pth.tar"), map_location=torch.device('cpu'))[0]
        self.reid_img_list = np.load("./reid/snapshot/reid_bulldog_new/image_list.npy")
        self.gallery = np.load(os.path.join("./reid/snapshot/reid_bulldog_new/", "2499_feature.npy"))

        self.topk = 7

    @staticmethod
    def get_img(path, img_size=416):
        img = transforms.ToTensor()(Image.open(path).convert('RGB'))
        img, _ = pad_to_square(img, 0)
        img = resize(img, img_size)

        return img

    def large_img(self):
        label = self.sender()

        img = cv2.imread(label.ori_path)
        height, width, bytesPerComponent = img.shape
        bytesPerLine = 3 * width
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg).scaled(self.label_3.width()-10, self.label_3.height()-10, Qt.KeepAspectRatio)
        self.label_3.setPixmap(pixmap)
        self.label_3.setAlignment(Qt.AlignCenter)

    def detect_helper(self, imgName):
        img = np.array(Image.open(imgName))
        input_img = self.get_img(imgName).unsqueeze(0)
        input_img = Variable(input_img.type(Tensor))

        with torch.no_grad():
            detections = self.model(input_img)
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)

        if detections is not None:
            detection = rescale_boxes(detections[0], self.img_size, img.shape[:2])
            x1, y1, x2, y2, conf, cls_conf, cls_pred = detection[0] # TODO

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            return x1, y1, x2, y2
        return None

    def reid_helper(self, img):

        img = torch.tensor(np.array(img)).permute(2,0,1).unsqueeze(0)

        feature, _ = self.reid_feat_model(img)
        feature, _ = self.reid_cls_model(feature)
        feature_npy = feature.detach().numpy()

        dis_index = find_closest_index(feature_npy[0], self.gallery, top_k=self.topk)
        print(dis_index)

        imgs = []
        for i in range(min(self.topk, len(dis_index))):
            imgs.append(self.reid_img_list[dis_index[i]])

        return imgs

    def upload_img(self):


        for label in self.reids_ql:
            label.deleteLater()
        self.reids_ql.clear()

        imgName, imgType = QFileDialog.getOpenFileName(self, "Upload", "", "*.jpg;;*.png;;s")
        if os.path.exists(imgName):
            jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
            self.label.setPixmap(jpg)
            self.label.setAlignment(Qt.AlignCenter)


            # detect
            bbox = self.detect_helper(imgName)
            if bbox:
                x1, y1, x2, y2 = bbox
                img = cv2.imread(imgName)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=2)
                # img = img
                img4reid = img.copy()[y1:y2, x1:x2, :]


                height, width, bytesPerComponent = img.shape
                bytesPerLine = 3 * width
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(QImg).scaled(self.label_2.width(), self.label_2.height(), Qt.KeepAspectRatio)
                self.label_2.setPixmap(pixmap)
                self.label_2.setAlignment(Qt.AlignCenter)


            # reid
            # img = cv2.imread(imgName)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (224, 224))

            img4reid = cv2.cvtColor(img4reid, cv2.COLOR_BGR2RGB)
            img4reid = cv2.resize(img4reid, (224, 224))

            reid_res = self.reid_helper(img4reid)
            for fn in reid_res:

                fn = "_".join(fn.split("_")[1:])

                ori_path = os.path.join("test/ori_bbox", fn)
                det_img = cv2.imread(os.path.join("test/det", fn))

                height, width, bytesPerComponent = det_img.shape
                bytesPerLine = 3 * width
                det_img = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
                QImg = QImage(det_img.data, width, height, bytesPerLine, QImage.Format_RGB888)

                qlabel = QLabel_alterada(self.scrollAreaWidgetContents)
                # qlabel.resize(self.scrollAreaWidgetContents.width()-5, height-10)
                qlabel.setFixedSize(self.scrollAreaWidgetContents.width()-25, (height/width)*self.scrollAreaWidgetContents.width()-25)
                pixmap = QPixmap.fromImage(QImg).scaled(self.scrollAreaWidgetContents.width()-25, (height/width)*self.scrollAreaWidgetContents.width()-25)#, Qt.KeepAspectRatio)
                qlabel.setPixmap(pixmap)
                qlabel.setAlignment(Qt.AlignCenter)
                qlabel.ori_path = ori_path
                qlabel.clicked.connect(self.large_img)
                self.scroVertLayout.addWidget(qlabel)
                self.reids_ql.append(qlabel)






if __name__ == '__main__':


    app = QApplication(sys.argv)

    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    myWin = MyMainForm()
    myWin.setWindowTitle("Embrace Your Pet")
    myWin.show()
    sys.exit(app.exec_())
