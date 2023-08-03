import logging
import os
import torch
import pickle
import torch.optim as optim
from sklearn.model_selection import train_test_split

def save_log(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def read_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def accuracy(outputs, labels):
    """
    Compute the accuracy
    outputs, labels: (tensor)
    return: (float) accuracy in [0, 100]
    """
    pre = torch.max(outputs.cpu(), 1)[1].numpy()
    y = labels.data.cpu().numpy()
    acc = ((pre == y).sum() / len(y)) * 100
    return acc

def save_model(model, args):
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")
    if not args.fft:
        torch.save(model.state_dict(), "./checkpoints/{}.tar".format(args.backbone))
    else:
        torch.save(model.state_dict(), "./checkpoints/{}FFT.tar".format(args.backbone))

def optimizer(args, parameter_list):
    # define optimizer
    if args.optimizer == "sgd":
        optimizer = optim.SGD(parameter_list, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(parameter_list, lr=args.lr)
    else:
        raise Exception("optimizer not implement")

    return optimizer
