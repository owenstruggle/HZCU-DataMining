import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.utils import shuffle


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class NN:
    def __init__(self, x, y, classes, hidden_features):
        self.x = x
        self.y = y
        self.classifications = classes
        self.samples, self.features = x.shape
        self.hidden_features = hidden_features

        self.weights1 = np.random.rand(self.features, self.hidden_features)
        self.weights2 = np.random.rand(self.hidden_features, self.classifications)
        self.bias1 = 1
        self.bias2 = 1

        # self.weights1 = np.array([[0.15, 0.25], [0.2, 0.3]])
        # self.weights2 = np.array([[0.4, 0.5], [0.45, 0.55]])
        # self.bias1 = np.array([0.35])
        # self.bias2 = np.array([0.6])

        self.net_hidden = None
        self.sigmoid_hidden = None
        self.net_out = None
        self.sigmoid_out = None

    def fit(self, epochs=10000, eta=1e-2):
        print("--------------训练开始--------------")
        min_score = np.inf
        for i in np.arange(1, epochs + 1):
            self.x, self.y = shuffle(self.x, self.y)
            self._forward_propagation()
            self._back_propagation(eta)

            s = self.score()
            if s < min_score:
                min_score = s

            if i % 10 == 0:
                print(i, self._calculate_square_error(), )
        print("最好的准确率:", min_score)

    def predict(self, x_test):
        self._forward_propagation(x_test)
        pred_zeros = np.zeros((self.samples, self.classifications))
        for i in range(len(self.sigmoid_out)):
            pred_zeros[i, np.argmax(self.sigmoid_out[i])] = 1
        return pred_zeros

    def score(self, x_test=None, y_test=None):
        if x_test is None and y_test is None:
            x_test, y_test = self.x, self.y
        pred = [np.argmax(i) for i in self.predict(x_test)]
        label = [np.argmax(i) for i in y_test]
        s = 0.0
        for i in range(len(pred)):
            if label[i] == pred[i]:
                s += 1
        return s / x_test.shape[0]

    def _forward_propagation(self, x_test=None):
        """前向传播"""
        # 输入层---->隐含层
        self.net_hidden = np.dot(self.x if x_test is None else x_test, self.weights1) + self.bias1
        self.sigmoid_hidden = sigmoid(self.net_hidden)
        # 隐含层---->输出层
        self.net_out = np.dot(self.sigmoid_hidden, self.weights2) + self.bias2
        self.sigmoid_out = sigmoid(self.net_out)

    def _back_propagation(self, eta):
        """反向传播"""
        delta_out = -(self.y - self.sigmoid_out) * (self.sigmoid_out * (1 - self.sigmoid_out))
        theta_weight2 = np.dot(self.sigmoid_hidden.T, delta_out)

        delta_hidden = np.sum(np.concatenate(
            [(delta_out * self.weights2[i]).reshape(self.samples, 1, self.classifications) for i in
             range(self.hidden_features)], 1), axis=2) * (self.sigmoid_hidden * (1 - self.sigmoid_hidden))
        theta_weight1 = np.dot(self.x.T, delta_hidden)

        self.weights1 -= eta * theta_weight1  # 隐含层---->输出层的偏置项更新
        self.bias1 -= eta * np.sum(delta_hidden)  # 隐含层---->输出层的权值更新
        self.weights2 -= eta * theta_weight2  # 输入层---->隐含层的偏置项更新
        self.bias2 -= eta * np.sum(delta_out)  # 输入层---->隐含层的权值更新

    def _calculate_square_error(self, x_test=None, y_test=None):
        if x_test is None and y_test is None:
            x_test, y_test = self.x, self.y
        return 0.5 * np.sum((y_test - self.sigmoid_out) ** 2)


if __name__ == '__main__':
    x_data = pd.read_csv('data/X_data.csv').to_numpy()
    y_label = pd.read_csv('data/y_label.csv').to_numpy()

    np.random.seed(64)
    x_data = scale(x_data)
    y_label_oh = OneHotEncoder(sparse=False).fit_transform(y_label)

    nn = NN(x_data, y_label_oh, 10, 25)
    print("未训练的模型预测准确率:", nn.score())

    nn.fit(eta=0.001)
    print("训练后的模型预测准确率:", nn.score())

    # nn = NN(np.array([[0.05, 0.1]]), np.array([[0.01, 0.99]]), 2, 2)
    # nn.fit(eta=0.5)
