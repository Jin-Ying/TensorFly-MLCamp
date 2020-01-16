import numpy as np
import matplotlib.pyplot as plt

prediction = np.load("./snapshot/dann_d_a_explore/99999_output.npy")
labels = np.load("./snapshot/dann_d_a_explore/99999_label.npy")

predict_label = np.argmax(prediction, axis=1)
print(predict_label)
for i in range(prediction.shape[0]):
    if predict_label[i] != labels[i]:
        print('correct: ', labels[i], 'prediction: ', predict_label[i])
        plt.bar(range(prediction.shape[1]), prediction[i])
        plt.savefig('./temp/'+str(i)+'.jpg')
        plt.clf()