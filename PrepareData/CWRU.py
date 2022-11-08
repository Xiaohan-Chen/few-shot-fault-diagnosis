"""
@Author: Xiaohan Chen
@Email: cxh_bb@outlook.com
"""

import numpy as np
from scipy.io import loadmat

# datanames in every working conditions
dataname_dict= {0:[97, 109, 122, 135, 173, 189, 201, 213, 226, 238],  # 1797rpm
                1:[98, 110, 123, 136, 175, 190, 202, 214, 227, 239],  # 1772rpm
                2:[99, 111, 124, 137, 176, 191, 203, 215, 228, 240],  # 1750rpm
                3:[100,112, 125, 138, 177, 192, 204, 217, 229, 241]}  # 1730rpm

axis = "_DE_time"
data_length = 1024


def transformation(sub_data, fft, normalization, backbone):

    if fft:
        sub_data = np.fft.fft(sub_data)
        sub_data = np.abs(sub_data) / len(sub_data)
        sub_data = sub_data[:int(sub_data.shape[0] / 2)].reshape(-1,)                

    if normalization == "0-1":
        sub_data = (sub_data - sub_data.min()) / (sub_data.max() - sub_data.min())
    elif normalization == "mean-std":
        sub_data = (sub_data - sub_data.mean()) / sub_data.std()

    if backbone in ("ResNet1D", "CNN1D"):
        sub_data = sub_data[np.newaxis, :]
    elif backbone == "ResNet2D":
        n = int(np.sqrt(sub_data.shape[0]))
        if fft:
            sub_data = sub_data[:n*n]
        sub_data = np.reshape(sub_data, (n, n))
        sub_data = sub_data[np.newaxis, :]
        sub_data = np.concatenate((sub_data, sub_data, sub_data), axis=0)

    return sub_data

def CWRU(datadir, load, labels, window, normalization, backbone, fft, number):
    """
    loading the hole dataset
    """
    path = datadir + "/CWRU/" + "Drive_end_" + str(load) + "/"
    dataset = {label: [] for label in labels}
    for label in labels:
        fault_type = dataname_dict[load][label]
        if fault_type < 100:
            realaxis = "X0" + str(fault_type) + axis
        else:
            realaxis = "X" + str(fault_type) + axis
        mat_data = loadmat(path+str(fault_type)+".mat")[realaxis]
        start, end = 0, data_length

        # set the endpoint of data sequence
        length = mat_data.shape[0]
        endpoint = data_length + number * window
        if endpoint > length:
            raise Exception("Sample number {} exceeds signal length.".format(number))

        # split the data and transformation
        while end < endpoint:
            sub_data = mat_data[start : end].reshape(-1,)

            sub_data = transformation(sub_data, fft, normalization, backbone)

            dataset[label].append(sub_data)
            start += window
            end += window
        
        dataset[label] = np.array(dataset[label], dtype="float32")

    return dataset

def CWRUloader(args, load, label_set, number):
    """
    args: arguments
    number: the numbers of training samples, "all" or specific numbers (string type)
    """
    dataset = CWRU(args.datadir, load, label_set, args.window, args.normalization, args.backbone, args.fft, number)

    # DATA, LABEL = [], []

    # datan = int(number)
    # for key in dataset.keys():
    #     LABEL.append(np.tile(key, datan))
    #     DATA.append(dataset[key][:datan])
    
    # DATA, LABEL = np.array(DATA, dtype="float32"), np.array(LABEL, dtype="int32")
    # DATA, LABEL = np.concatenate(DATA, axis=0), np.concatenate(LABEL, axis=0)

    return dataset