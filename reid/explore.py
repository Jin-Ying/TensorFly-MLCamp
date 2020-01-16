import numpy as np
import os

root='./snapshot/reid_bulldog_new/'

feature = np.load(os.path.join(root, "2499_feature.npy"))
label = np.load(os.path.join(root, "2499_label.npy"))

# print(feature)
# print(label)
for i in range(feature.shape[0]):
    print(np.linalg.norm(np.abs(feature[0]-feature[i])))
    # print(np.dot(feature[1],feature[i])/(np.linalg.norm(feature[0]) * np.linalg.norm(feature[i])))

