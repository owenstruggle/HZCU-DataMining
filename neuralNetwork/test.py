import numpy as np
import pandas as pd


def softmax(x):
    m = np.max(x)
    return np.exp(x - m) / np.sum(np.exp(x - m))  # 防止出现 overflow


def feedforward(weights, x):
    return softmax(np.dot(np.concatenate((x, np.ones((x.shape[0], 1))), axis=1), weights.T))


def score(x, y, theta1, theta2):
    x_hidden_data = feedforward(theta1, x)
    pred = feedforward(theta2, x_hidden_data)
    s = 0.0
    for i in range(pred.shape[0]):
        if y[i] == np.argmax(pred[i])+1:
            s += 1
    return s / pred.shape[0]


if __name__ == '__main__':
    x_data = pd.read_csv('data/X_data.csv').to_numpy()
    y_label = pd.read_csv('data/y_label.csv').to_numpy()
    theta1 = pd.read_csv('data/Theta1.csv', header=None).to_numpy()
    theta2 = pd.read_csv('data/Theta2.csv', header=None).to_numpy()
    print("准确率:", score(x_data, y_label, theta1, theta2))
