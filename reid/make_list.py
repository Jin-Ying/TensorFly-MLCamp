import numpy as np
import os

root_dir = '/data/finetune/pets_google/american_bulldog/test'
img_list = []

for i in range(18):
    sub_dir = os.path.join(root_dir, str(i+44))
    file_list = sorted(os.listdir(sub_dir))
    for file in file_list:
        img_list.append(file)

img_list = np.array(img_list)
np.save("./snapshot/reid_bulldog_new/image_list.npy", img_list)