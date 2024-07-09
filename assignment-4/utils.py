import numpy as np
import os
from tqdm.notebook import tqdm
from collections import Counter
import copy
    
class DataLoader:
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.i = 0
        
    def __iter__(self):
        return self
    
    def __len__(self):
        return int(len(self.X)/self.batch_size)
    
    def __next__(self):     
        start, end = self.batch_size * self.i, self.batch_size * (self.i + 1)
        if end > len(self.X):
            self.i = 0
            raise StopIteration

        self.i = (self.i + 1)                  
        return self.X[start: end, ], self.y[start : end, ]
            
class Relu:
    def __call__(self, x):
        return np.maximum(np.zeros_like(x), x)    
    def __str__(self):
        return "ReLU"
    def grad(self, x):
        return (x >= 0).astype(np.float32)

class Sigmoid:
    def __call__(self, x):
        return 1/(1 + np.exp(-x))
    def __str__(self):
        return "Sigmoid"
    def grad(self, x):
        return self.__call__(x) * (1 - self.__call__(x))
    
class Linear:
    def __call__(self, x):
        return x
    def __str__(self):
        return "Linear"
    def grad(self, x):
        return np.ones_like(x)

class Tanh:
    def __call__(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    def __str__(self):
        return "Tanh"
    def grad(self, x):
        return 1 - (self.__call__(x) ** 2)
    
class Softmax:
    def __call__(self, x):
        return (np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)) 
    def __str__(self):
        return "Softmax"
    def grad(self, x):
        return np.ones_like(x)
    
class MSELoss:
    def __call__(self, y, y_pred):
        self.y, self.y_pred = y, y_pred
        return np.mean((self.y - self.y_pred) ** 2)
    def grad(self):
        return -2 * (self.y - self.y_pred)
    
class BCELoss:
    def __call__(self, y, y_pred):
        self.y, self.y_pred = y, y_pred
    def grad(self):
        pass
    
class CrossEntropyLoss:
    def __call__(self, y, y_pred):
        self.y, self.y_pred = y, y_pred
        return np.mean(np.sum(-y  * np.log(y_pred), axis=-1))
    def grad(self):
        return self.y_pred - self.y


class Neuron:
    def __init__(self, dim_in, activation):
        self.dzw, self.dzx, self.daz = 0, 0, 0
        self.dim_in = dim_in
        self.activation = activation
    
    def get_grads(self):
        return [self.dzw, self.dzx, self.daz]
        
    def calculate_grad(self, x, z, w, index):
        self.dzw = x
        self.dzx = w[index]
        self.daz = self.activation.grad(z[:, index])
        # print(self.daz.shape, self.dzw.shape, self.daz.shape)
        
        return [self.dzw, self.dzx, self.daz]
    
class Layer:
    def __init__(self, dim_in, dim_out, activation):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activation = activation
        
        self.W = np.random.randn(self.dim_out, self.dim_in)
        self.b = np.random.randn(self.dim_out)
        
        self.neurons = [Neuron(self.dim_in, activation) for _ in range(self.dim_out)]
        
        self.dzw, self.dzx, self.daz = [], [], []
        
    
    def get_grads(self):
        grads = [np.stack(self.dzw, axis=1),
                np.stack(self.dzx, axis=-1), 
                np.stack(self.daz, axis=-1)]

        self.dzw.clear()
        self.dzx.clear()
        self.daz.clear()
        return grads
        
    def __str__(self):
        return(f"Layer: [in:{self.dim_in}] [out:{self.dim_out}] [activation:{self.activation}]")
        
    def __call__(self, x):
        '''
            x: (bs, dim_in)
        '''
        
        if x.shape[1] != self.dim_in:
            raise TypeError(f'Input should have dimension {self.dim_in} but found {x.shape[1]}')
        
        z = x @ self.W.T + self.b
        self.a = self.activation(z)

        self.daz.clear()
        self.dzx.clear()
        self.dzw.clear()

        for i, neuron in enumerate(self.neurons):
            dzw, dzx, daz = neuron.calculate_grad(x, z, self.W, i)
            self.dzw.append(dzw)
            self.dzx.append(dzx)
            self.daz.append(daz)
            
        return self.a


from sklearn.metrics import accuracy_score, r2_score

class Model:
    def __init__(self, logger, 
                 loss_fxn=MSELoss(), 
                 lr=1e-3, 
                 type='regression',
                 epochs=1000, 
                 verbose=False):
        
        self.loss_fxn = loss_fxn
        self.layers = []
        self.lr = lr
        self.dW, self.dB = [], []
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc':[]
        }
        self.accuracy = self.classification_accuracy if type=='classification' else self.regression_accuracy
        self.logger = logger
        self.epochs = epochs
        self.verbose = verbose
        
    def classification_accuracy(self, y_pred, y):
        return accuracy_score(np.argmax(y, axis=-1), np.argmax(y_pred, axis=-1))
        
    def regression_accuracy(self, y_pred, y):
        return r2_score(y, y_pred)
        
    def __str__(self):
        out = ""
        for layer in self.layers:
            out += layer.__str__() + "\n"
        return out

    def add(self, layer):
        self.layers.append(layer)

    def __call__(self, x):
        '''
            x: (bs, dim_in)
        '''
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self):
        dLy = self.loss_fxn.grad()
        common = dLy

        for i in range(len(self.layers)-1, -1, -1):
            dzw, dzx, daz = self.layers[i].get_grads()
            if i != len(self.layers) - 1:
                common = common @ self.layers[i + 1].W
            common = common * daz
            dw = common[:, :, None] * dzw
            db = common[:, :] * 1
            
            self.dW.append(np.mean(dw, axis=0))
            self.dB.append(np.mean(db, axis=0))
            
    def update_gradients(self):
        for i, (dw, db) in enumerate(zip(reversed(self.dW), reversed(self.dB))):
            self.layers[i].W += - self.lr * dw
            self.layers[i].b += - self.lr * db
            
        self.dW.clear()
        self.dB.clear()
        
    def training_step(self, loader):
        loss, acc = 0, 0
        for x, y in loader:
            y_pred = self.__call__(x)
            loss += self.loss_fxn(y, y_pred)
            acc += self.accuracy(y_pred, y)
            
            self.backward()
            self.update_gradients()
            
        return loss/len(loader), acc/len(loader)
    
    def validate_step(self, loader):
        loss, acc = 0, 0
        for x, y in loader:
            y_pred = self.__call__(x)
            loss += self.loss_fxn(y, y_pred)
            acc += self.accuracy(y_pred, y)
            
        return loss/len(loader), acc/len(loader)

    def train(self, X, y):
        
        train_loader = DataLoader(X, y, batch_size=len(X))
        
        for epoch in range(self.epochs):
            
            train_loss, train_acc = self.training_step(train_loader)
            # val_loss, val_acc = self.validate_step(val_loader)
        
            self.history['train_loss'].append(train_loss)
            # self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            # self.history['val_acc'].append(val_acc)
            
            if epoch % 20 == 0 and self.verbose:
                print(f"epoch: {epoch} \tTrain:[loss:{train_loss:.4f} acc:{train_acc:.4f}]]")
    
            if self.logger is not None and self.verbose:
                self.logger.log({"train_acc": train_acc, "train_loss": train_loss, 'val_acc': val_acc, 'val_loss': val_loss})
        if not self.verbose:
            print(f"epoch: {epoch} \tTrain:[loss:{train_loss:.4f} acc:{train_acc:.4f}]]")
            


class Bagging:
    def __init__(self, 
                 model_list, 
                 X, y,
                 num_estimators, 
                 frac:float, 
                 bootstrap:bool, 
                 voting_mechanism:str,
                 task:str) -> None:
        
        self.model_list = model_list
        self.num_estimators = num_estimators
        self.frac = frac
        self.bootstrap = bootstrap
        self.voting_mechanism = voting_mechanism
        self.task = task
        self.X, self.y = X, y
        self.num_samples = len(self.X)
        self.model_list = [copy.deepcopy(model) for model in self.model_list for _ in range(self.num_estimators)]
        self.model_weights = None
        
    def hard_vote(self, preds):
        if self.task == 'classification':
            preds = np.array([np.argmax(i, axis=-1).tolist() for i in preds])
            return [Counter(i).most_common(1)[0][0] for i in preds.T]
        else:
            return np.mean(preds, axis=0)
        
    def soft_vote(self, preds):
        if self.task == 'classification':
            preds = np.array(preds) * self.model_weights[:, None, None]
            preds = np.sum(preds, axis=0)
            return np.argmax(preds, axis=-1)
        else:
            preds = np.array(preds) * self.model_weights[:, None, None]
            return np.sum(preds, axis=0)
              
    def __call__(self, x):
        self.preds = []
        for model in self.model_list:
            self.preds.append(model(x))
            
        if self.voting_mechanism == 'hard':
            return self.hard_vote(self.preds)
        else:
            print("Prediction using soft vote")
            return self.soft_vote(self.preds)
    
    def validate(self, x, y):
        self.preds = [model(x) for model in self.model_list]
        if self.task == 'classification':
            acc = np.array([accuracy_score(np.argmax(y, -1), np.argmax(pred, -1)) for pred in self.preds])
            return acc/np.sum(acc)
        else:
            losses = np.array([MSELoss()(y, pred) for pred in self.preds])
            losses = 1/losses
            return losses/np.sum(losses)
        
    def get_training_data(self, i):
        if self.bootstrap:
            indices = np.random.randint(0, self.num_estimators, int(self.frac * self.num_samples))
            return self.X[indices], self.y[indices]
        else:
            start, end = i * int((self.num_samples)/self.num_estimators), (i + 1) * (int((self.num_samples)/self.num_estimators))
            return self.X[start:end, :], self.y[start:end, ]
    
    def train(self):
        for model in self.model_list:
            indices = np.random.randint(0, self.num_samples, int(self.frac * self.num_samples))
            model.train(self.X[indices], self.y[indices])
            
        print("Training done!")

        if self.voting_mechanism == 'soft':
            self.model_weights = self.validate(self.X, self.y)
            print("Model weights loaded")
            
            
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

class DecisionTree:
    def __init__(self,
                 criterion,
                 max_depth,
                 task):
        self.criterion = criterion
        self.max_depth = max_depth
        self.task = task
        self.model = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth)  if task == 'classification' \
                    else DecisionTreeRegressor(criterion=self.criterion, max_depth=self.max_depth)
        
    def __call__(self, x):
        pred = self.model.predict(x) 
        return np.expand_dims(pred, -1) if self.task == 'regression' else pred
        
    def train(self, X, y):
        self.model.fit(X, y)
        print("Accuracy of decision tree:", self.model.score(X, y))
        
          
def main():
    
    model = Model(logger=None, loss_fxn=MSELoss())
    model.add(Layer(32, 64, Relu()))
    model.add(Layer(64, 10, Linear()))
    
    print(model)
      
if __name__ == "__main__":
    main()