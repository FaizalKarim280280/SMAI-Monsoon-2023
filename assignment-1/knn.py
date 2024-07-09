import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse
import sklearn
import time
from collections import Counter

def reshape_data(data):
    temp = []
    for d in data:
        temp.append(d.reshape(-1, 1))
    temp = np.array(temp)
    temp = temp - np.mean(temp)
    temp = temp/(np.var(temp) ** 0.5)
    return temp
def load_data(path):
    data = np.load(path, allow_pickle=True)
    resnet, vit, y = reshape_data(data[:, 1]), reshape_data(data[:, 2]), data[:, 3]
    return {
        'resnet': {'X': resnet, 'y': y},
        'vit': {'X': vit, 'y': y}
    }

def euclidean(x, y, optimized=False):
    return np.sqrt(np.sum((x - y) ** 2, axis=1)) if not optimized else (
        np.sqrt(np.sum((x - y[:, np.newaxis, :]) ** 2, axis=2)).squeeze(-1))

def manhattan(x, y, optimized=False):
    return np.sum(np.abs(x - y), axis=1) if not optimized else (
        np.sum(np.abs(x - y[:, np.newaxis, :]), axis=2).squeeze(-1))

def cosine(x, y, optimized=False):
    if len(x.shape) == 3:
        x, y = x.squeeze(axis=-1), y.squeeze(axis=-1)
    return 1 - (np.sum(x * y, axis=-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)))


features_scalers = {
    'standardization': sklearn.preprocessing.StandardScaler,
    'min_max': sklearn.preprocessing.MinMaxScaler,
    'robust': sklearn.preprocessing.RobustScaler
}

class KNNModel:
    def __init__(self, train_data, encoder_type, k, distance_metric, scaler, metric_average, optimized=False):
        self.encoder = train_data[encoder_type]
        self.X = self.encoder['X']
        self.y = self.encoder['y']
        self.k = k
        self.distance_metric = distance_metric
        self.scaler = features_scalers[scaler]()
        self.X = self.scaler.fit_transform(self.X.squeeze(-1))
        self.metric_average = metric_average
        self.optimized = optimized

    def forward_optimized(self, x):

        self.X, x = np.expand_dims(self.X, axis=-1), np.expand_dims(x, axis=-1)
        distances = np.array([self.distance_metric(self.X, i) for i in x])
        if len(distances.shape) == 3:
            distances = distances.squeeze(-1)
        min_k_idx = self.y[np.argsort(distances, axis=-1)[:, :self.k]]
        # pred = [pd.Series(row).value_counts().keys()[0] for row in min_k_idx]
        pred = [Counter(row).most_common(1)[0][0] for row in min_k_idx]
        return pred

    def forward(self, x):
        # x : (512, 1) self.X: (1500, 512, 1)
        # print(x.shape)
        distances = self.distance_metric(self.X, x)
        min_k_idx = np.argsort(distances.ravel())[:self.k]
        # pred = pd.Series(self.y[min_k_idx]).value_counts().keys()[0]
        pred = Counter(self.y[min_k_idx]).most_common(1)[0][0]
        return pred

    def get_scores(self, x_test, y_test):

        assert self.X.shape[1] == x_test.shape[1], (
            "Number of features in x_test={} not equal to original data features={}".format(x_test.shape[1],
                                                                                            self.X.shape[1]))
        assert len(x_test) == len(y_test), (
            'Number of x_test={} not equal to y_test={}'.format(len(x_test), len(y_test)))

        start = time.time()

        x_test = self.scaler.transform(x_test.squeeze(-1))
        y_pred = []
        if self.optimized:
            y_pred = self.forward_optimized(x_test)
        else:
            for x in x_test:
                y_pred.append(self.forward(x))

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=self.metric_average)
        recall = recall_score(y_test, y_pred, average=self.metric_average)
        f1 = f1_score(y_test, y_pred, average=self.metric_average)

        start = time.time() - start

        return {
            # 'pred' : y_pred,
            'acc': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'time': start
        }

def print_output(output: dict, encoder_name):
    print("Encoder:", encoder_name)

    for k, v in output.items():
        print("{} : {:.3f}".format(k, v))
    print("="*50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path of the input file')

    args = parser.parse_args()

    input_path = args.input_path

    train_data = load_data('./data.npy')
    val_data = load_data(input_path)


    model1 = KNNModel(train_data=train_data,
                      encoder_type='vit',
                      k=1,
                      distance_metric=euclidean,
                      scaler='standardization',
                      metric_average='micro',
                      optimized=True)

    model2 = KNNModel(train_data=train_data,
                      encoder_type='resnet',
                      k=1,
                      distance_metric=euclidean,
                      metric_average='micro',
                      scaler='standardization',
                      optimized=True)

    output1 = model1.get_scores(*val_data['vit'].values())
    output2 = model2.get_scores(*val_data['resnet'].values())

    print_output(output1, 'vit')
    print_output(output2, 'resnet')

if __name__ == "__main__":
    main()
