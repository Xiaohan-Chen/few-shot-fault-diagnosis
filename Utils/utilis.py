import logging
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

def save_model(model, args):
    if not args.fft:
        torch.save(model.state_dict(), "./checkpoints/{}_checkpoint.tar".format(args.backbone))
    else:
        torch.save(model.state_dict(), "./checkpoints/{}FFT_checkpoint.tar".format(args.backbone))

def optimizer(args, parameter_list):
    # define optimizer
    if args.optimizer == "sgd":
        optimizer = optim.SGD(parameter_list, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(parameter_list, lr=args.lr)
    else:
        raise Exception("optimizer not implement")

    # Define the learning rate decay
    if args.lr_scheduler == 'step':
        steps = [int(step) for step in args.steps.split(',')]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=args.gamma)
    elif args.lr_scheduler == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    elif args.lr_scheduler == 'stepLR':
        steps = int(args.steps.split(",")[0])
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, args.gamma)
    elif args.lr_scheduler == 'cos':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, 0)
    elif args.lr_scheduler == 'fix':
        lr_scheduler = None
    else:
        raise Exception("lr schedule not implement")

    return optimizer, lr_scheduler
