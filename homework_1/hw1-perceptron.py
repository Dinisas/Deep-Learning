#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import time
import pickle
import json

import numpy as np

import utils

class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))
    
    def save(self, path):
        """
        Save perceptron to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load perceptron from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def update_weight(self, x_i, y_i):
        raise NotImplementedError
    
    def train_epoch(self, X, y):
        """
        X (n_examples, n_features): features for the whole dataset
        y (n_examples,): labels for the whole dataset
        """
        # Todo: Q1 1(a)
        for i in range(X.shape[0]):
            self.update_weight(X[i],y[i])

    def predict(self, X):
        """
        X (n_examples, n_features)
        returns predicted labels y_hat, whose shape is (n_examples,)
        """
        # Todo: Q1 1(a)
        scores = np.dot(X, self.W.T)
        return np.argmax(scores, axis=1)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels

        returns classifier accuracy
        """
        # Todo: Q1 1(a)
        y_predict = self.predict(X)
        return np.mean(y==y_predict)
    
class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i):
        """
        x_i (n_features,): a single training example
        y_i (scalar): the gold label for that example
        """
        # Todo: Q1 1(a)
        y_predict = np.dot(self.W,x_i)
        predicted_label = np.argmax(y_predict)
        if predicted_label != y_i:
            self.W[y_i,:] += x_i
            self.W[predicted_label,:] -= x_i

class LogisticRegression(LinearModel):
    def __init__ (self, n_classes, n_features, learning_rate=0.0001, reg_lambda =0.00001):
        self.W = np.zeros((n_classes, n_features))
        self.lr = learning_rate
        self.reg_lambda = reg_lambda

    def softmax(self, y_predict):
        max = np.max(y_predict)
        return np.exp(y_predict - max) / np.sum(np.exp(y_predict - max))
       
    def update_weight(self,x_i,y_i):
        # Calculate the predicted scores
        y_predict = np.expand_dims(self.W.dot(x_i), axis=1)

        # One-hot encode true label
        y_i_one_hot = np.zeros((self.W.shape[0], 1))
        y_i_one_hot[int(y_i)] = 1

        # Get predicted probabilities
        predicted_probs = self.softmax(y_predict)

        # Calculate the error
        error = predicted_probs - y_i_one_hot

        # Calculate loss gradient: error ⊗ x_i^T
        loss_gradient = np.dot(error, np.expand_dims(x_i, axis=1).T)

        # Add L2 regularization gradient: λW
        loss_gradient += self.reg_lambda * self.W

        # Update weights using SGD
        self.W -= self.lr * loss_gradient

        # Loss
        loss = -np.sum(y_i_one_hot * np.log(predicted_probs))
        return loss

def run_grid_search(X_train, y_train, X_valid, y_valid, X_test, y_test, n_classes):
    learning_rates = [1e-4, 5e-4, 1e-3]
    penalties = [1e-5, 1e-4]
    feature_types = ["original"]  # Add "custom" when you implement it
    
    results = []
    best_overall = {"valid_acc": 0.0}
    
    for lr in learning_rates:
        for lam in penalties:
            for feat in feature_types:
                # Apply features
                if feat == "original":
                    X_train_feat, X_valid_feat, X_test_feat = X_train, X_valid, X_test
                
                # Create model
                model = LogisticRegression(n_classes, X_train_feat.shape[1], 
                                          learning_rate=lr, reg_lambda=lam)
                
                best_valid = 0.0
                best_w = None
                
                # Train 20 epochs
                for epoch in range(20):
                    order = np.random.permutation(X_train_feat.shape[0])
                    model.train_epoch(X_train_feat[order], y_train[order])
                    
                    val_acc = model.evaluate(X_valid_feat, y_valid)
                    if val_acc > best_valid:
                        best_valid = val_acc
                        best_w = model.W.copy()
                
                # Test accuracy with best checkpoint
                model.W = best_w
                test_acc = model.evaluate(X_test_feat, y_test)
                
                print(f"LR={lr}, L2={lam}, Feat={feat} | Valid={best_valid:.4f}, Test={test_acc:.4f}")
                
                # Store result
                config = {
                    "lr": lr, "l2": lam, "features": feat,
                    "valid_acc": best_valid, "test_acc": test_acc
                }
                results.append(config)
                
                # Track best overall
                if best_valid > best_overall["valid_acc"]:
                    best_overall = config
    
    # Save to JSON (for homework submission)
    with open("Q2c-grid-search-results.json", "w") as f:
        json.dump({"best": best_overall, "all": results}, f, indent=2)
    
    print(f"\nBest config: LR={best_overall['lr']}, L2={best_overall['l2']}, "
          f"Features={best_overall['features']}, Test={best_overall['test_acc']:.4f}")
    
    return results
class MLP(object):
    def __init__ (self,n_classes, n_features, hidden_size):
        #loc is the mean, scale is the standard deviation, and size is the number of elements you want to generate.
        self.W1 = np.random.normal(loc=0.1, scale=0.1, size=(hidden_size, n_features))
        self.b1 = np.zeros((hidden_size)) #hidden bias, number of hidden units
        self.W2 = np.random.normal(loc=0.1, scale=0.1, size=(n_classes, hidden_size))
        self.b2 = np.zeros((n_classes)) #output bias, number of output classes
        self.weights = [self.W1, self.W2]
        self.biases = [self.b1, self.b2]
        self.n_classes = n_classes
        self.n_features = n_features
        self.hidden_size = hidden_size
 
    def save(self, path):
        """
        Save perceptron to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load perceptron from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def predict(self, X):
        # Compute the forward pass of the network.
        predicted_labels = []
        for x_train in X:
            output, _ = self.forward_propagation(x_train, self.weights, self.biases)
            predicted_probs = self.softmax(output)
            label = predicted_probs.argmax(axis=0).tolist()
            predicted_labels.append(label)
        return np.array(predicted_labels) 
    
    def one_hot(self, y):
        one_hot = np.zeros((np.size(y, 0), self.W2.shape[0]))
        for i in range(np.size(y, 0)):
            one_hot[i, y[i]] = 1
        return one_hot
    
    def relu(self,z):
        return np.maximum(0,z)
    
    def derivative_relu(self,z):
        return (z > 0).astype(float)

    def softmax(self, y_predict):
        max = np.max(y_predict)
        return np.exp(y_predict - max) / np.sum(np.exp(y_predict - max))
    
    def evaluate(self, X, y):
        #identical to evaluate from Linear Model
        y_predict = self.predict(X)
        return np.mean(y==y_predict)
    
    def compute_loss(self, output, y):
        # compute loss
        probs = self.softmax(output)
        loss = -y.dot(np.log(probs + 1e-8))
        return loss 

    def forward_propagation(self,x, weights, biases):
        num_layers=len(weights)#2
        hiddens=[]
        for i in range (num_layers):
            h = x if i == 0 else hiddens[i-1]
            z = weights[i].dot(h) + biases[i]
            if i < num_layers - 1: # Assuming the output layer has no activation.
                hiddens.append(self.relu(z))
        output=z
        return output, hiddens
    
    def backward_propagation(self, x, y, output, hiddens, weights):
        num_layers = len(weights)
        z = output

        probs = self.softmax(z)
        grad_z = probs - y
        
        grad_weights = []
        grad_biases = []
        
        # Backpropagate gradient computations 
        for i in range(num_layers-1, -1, -1): #(start, stop, step)
            
            # Gradient of hidden parameters.
            h = x if i == 0 else hiddens[i-1]
            grad_weights.append(grad_z[:, None].dot(h[:, None].T))
            grad_biases.append(grad_z)
            
            # Gradient of hidden layer below.
            grad_h = weights[i].T.dot(grad_z)
            
            # Gradient of hidden layer below before activation.
            derivative_relu = (h > 0).astype(float) #relu derivative is 1 if h > 0, 0 otherwise
            grad_z = grad_h * derivative_relu
          
        # Making gradient vectors have the correct order
        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases
    
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        y_one_hot = np.zeros(self.n_classes)
        y_one_hot[int(y_i)] = 1.0
        
        output, hiddens = self.forward_propagation(x_i, self.weights, self.biases)
        loss = self.compute_loss(output, y_one_hot)
        grad_weights, grad_biases = self.backward_propagation(x_i, y_one_hot, output, hiddens, self.weights)
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * grad_weights[i]
            self.biases[i] -= learning_rate * grad_biases[i]
        
        return loss
    
    def train_epoch(self, X, y, learning_rate=0.001):
        total_loss = 0.0
        for i in range(X.shape[0]):
            loss = self.update_weight(X[i], y[i], learning_rate)
            total_loss += loss
        return total_loss / X.shape[0]  # Return average loss


def main(args):
    utils.configure_seed(seed=args.seed)

    data = utils.load_dataset(data_path=args.data_path, bias=True)
    X_train, y_train = data["train"]
    X_valid, y_valid = data["dev"]
    X_test, y_test = data["test"]
    n_classes = np.unique(y_train).size
    n_feats = X_train.shape[1]

    # initialize the model
    model = Perceptron(n_classes, n_feats)

    epochs = np.arange(1, args.epochs + 1)

    valid_accs = []
    train_accs = []

    start = time.time()

    best_valid = 0.0
    best_epoch = -1
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(X_train.shape[0])
        X_train = X_train[train_order]
        y_train = y_train[train_order]

        model.train_epoch(X_train, y_train)

        train_acc = model.evaluate(X_train, y_train)
        valid_acc = model.evaluate(X_valid, y_valid)

        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        print('train acc: {:.4f} | val acc: {:.4f}'.format(train_acc, valid_acc))

        # Todo: Q1(a)
        # Decide whether to save the model to args.save_path based on its
        # validation score
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_epoch = i 
            model.save(args.save_path)

    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))

    print("Reloading best checkpoint")
    best_model = Perceptron.load(args.save_path)
    test_acc = best_model.evaluate(X_test, y_test)

    print('Best model test acc: {:.4f}'.format(test_acc))

    utils.plot(
        "Epoch", "Accuracy",
        {"train": (epochs, train_accs), "valid": (epochs, valid_accs)},
        filename=args.accuracy_plot
    )

    with open(args.scores, "w") as f:
        json.dump(
            {"best_valid": float(best_valid),
             "selected_epoch": int(best_epoch),
             "test": float(test_acc),
             "time": elapsed_time},
            f,
            indent=4
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int,
                        help="""Number of epochs to train for.""")
    parser.add_argument('--data-path', type=str, default="emnist-letters.npz")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--accuracy-plot", default="Q1-perceptron-accs.pdf")
    parser.add_argument("--scores", default="Q1-perceptron-scores.json")
    args = parser.parse_args()
    main(args)
