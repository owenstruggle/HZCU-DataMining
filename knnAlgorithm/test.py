import pandas as pd
import numpy as np
import operator

from sklearn.preprocessing import OneHotEncoder

amazon_X = pd.read_csv("data/amazon_X.csv").to_numpy()
amazon_y = pd.read_csv("data/amazon_y.csv")
np.random.seed(64)


class SoftmaxRegression:
    def __init__(self):
        self.n_samples = None
        self.n_features = None
        self.n_classes = None
        self.weights = None
        self.all_loss = list()

    def fit(self, x, y, iters=1000, alpha=0.1, lam=0.01):
        self.n_samples, self.n_features = x.shape
        self.n_classes = y.shape[1]
        # self.weights = np.random.rand(self.n_classes, self.n_features)  # 使用随机初始化会造成运算异常
        self.weights = np.zeros((self.n_classes, self.n_features))

        for i in range(iters):
            probs = self._softmax(x)
            loss = - (1 / self.n_samples) * np.sum(y * np.nan_to_num(np.log(probs)))
            self.all_loss.append(loss)

            dw = -(0.5 / self.n_samples) * np.dot((y - probs).T, x) + lam * self.weights
            dw[:, 0] -= lam * self.weights[:, 0]
            self.weights -= alpha * dw

    def predict(self, test_x):
        probs = self._softmax(test_x)
        return np.argmax(probs, axis=1).reshape((-1, 1))

    def score(self, test_x, test_y):
        score = 0.0
        for i in range(len(test_y)):
            if (self.predict([test_x[i]]) == np.argmax(test_y[i])).any():
                score += 1
        return score / len(test_y)

    def _softmax(self, x):
        scores = np.dot(x, self.weights.T)
        sm = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        sm = np.nan_to_num(sm)
        return sm


if __name__ == '__main__':
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    amazon_y = enc.fit_transform(amazon_y)

    reg = SoftmaxRegression()
    reg.fit(amazon_X, amazon_y, alpha=0.01)
    print(reg.score(amazon_X, amazon_y))

