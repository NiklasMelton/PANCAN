import numpy as np


def load_data():
    data = []
    init = False
    for line in open('TCGA-PANCAN/data.csv','r').readlines():
        if not init:
            init = True
        else:
            data.append(np.array(list(map(float,line.split(',')[1:]))))
    return np.vstack(data)

def load_labels():
    labels = []
    init = False
    for line in open('TCGA-PANCAN/labels.csv','r').readlines():
        if not init:
            init = True
        else:
            labels.append(np.array(list(map(str,line.split(',')[1:])),dtype=str))
    return np.vstack(labels)

def normalize(data):
    data -= np.min(data,axis=0)
    data /= np.max(data,axis=0)
    return data



