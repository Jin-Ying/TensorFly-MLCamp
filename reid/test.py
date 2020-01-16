import torch
import os
import numpy as np
from find_closest_index import find_closest_index

topk=7

img_list = np.load("./snapshot/reid_bulldog_new/image_list.npy")
feat_model = torch.load("./snapshot/finetune_pet/iter_01000_model.pth.tar", map_location=torch.device('cpu'))[0]
cls_model = torch.load(os.path.join("./snapshot/reid_bulldog_new", "iter_05000_classifier.pth.tar"), map_location=torch.device('cpu'))[0]

image = torch.rand(1,3,224,224)
image = image.cuda()

feature,_=feat_model(image)
feature,_=cls_model(feature)
feature_npy = feature.detach().cpu().numpy()

print(feature_npy[0])

gallery = np.load(os.path.join("./snapshot/reid_bulldog_new/", "2499_feature.npy"))

print(find_closest_index(feature_npy[0], gallery))
dis = []

for i in range(gallery.shape[0]):
    dis.append(np.linalg.norm(np.abs(feature_npy-gallery[i])))

dis = np.array(dis)
dis_index = np.argsort(dis)
print(dis_index)

imgs = []
for i in range(topk):
    imgs.append(img_list[dis_index[i]])

print(imgs)
