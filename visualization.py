import matplotlib.pyplot as plt
from Utils.utilis import *

def plot(meters):
    for key in meters:
        plt.figure(figsize=(9,6))
        plt.style.use('seaborn')
        plt.plot(meters[key], linewidth=2.0, marker='x', markersize=20.0)
        plt.xlabel("Epoch", fontsize=15)
        plt.ylabel("{}".format(key), fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title("{}".format(key), fontsize=15)
        plt.grid(visible=True)
        plt.savefig("./History/{}.png".format(key))

if __name__ == "__main__":
    root = "./History/Prototypical.pkl"
    meters = read_pkl(root)
    plot(meters)