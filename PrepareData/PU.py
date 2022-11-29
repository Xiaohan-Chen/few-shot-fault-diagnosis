"""
@Author: Xiaohan Chen
@Email: cxh_bb@outlook.com
"""

import numpy as np
from scipy.io import loadmat

data_length = 1024

#1 Undamaged (healthy) bearings(6X)
HBdata = ['K001',"K002",'K003','K004','K005','K006']

#2 Artificially damaged bearings(12X)
ADBdata = ['KA01','KA03','KA05','KA06','KA07','KA08','KA09','KI01','KI03','KI05','KI07','KI08']

#3 Bearings with real damages caused by accelerated lifetime tests(13x)
RDBdata = ['KA04','KA15','KA16','KA22','KA30','KB23','KB24','KB27','KI14','KI16','KI17','KI18','KI21']
#RDBdata = ['KA16','KA22','KA30','KB23','KB27','KI14','KI17','KI18']

#working condition
WC = ["N15_M07_F10","N09_M07_F10","N15_M01_F10","N15_M07_F04"]
#state = WC[0] #WC[0] can be changed to different working conditions


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

def read_file(path, filename):
    data = loadmat(path)[filename][0][0][2][0][6][2]
    return data.reshape(-1,)

def PU(datadir, load, labels, window, normalization, backbone, fft, number):
    """
    loading the whole dataset
    """
    dataset_dir = datadir + "/PU/"
    state = WC[load]

    dataset = {label: [] for label in labels}

    for label in labels:
        filename = state + '_' + RDBdata[label] + '_' + '1'
        subset_path = dataset_dir + RDBdata[label] + '/' + filename + '.mat'
        mat_data = read_file(subset_path, filename)

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

def PUloader(args, load, label_set, number):
    """
    args: arguments
    number: the numbers of training samples, "all" or specific numbers (string type)

    Return:
        A dictionary, labels are keys, values are the corresponding fault signals
    """
    dataset = PU(args.datadir, load, label_set, args.window, args.normalization, args.backbone, args.fft, number)

    # DATA, LABEL = [], []

    # datan = int(number)
    # for key in dataset.keys():
    #     LABEL.append(np.tile(key, datan))
    #     DATA.append(dataset[key][:datan])
    
    # DATA, LABEL = np.array(DATA, dtype="float32"), np.array(LABEL, dtype="int32")
    # DATA, LABEL = np.concatenate(DATA, axis=0), np.concatenate(LABEL, axis=0)

    return dataset