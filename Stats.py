import matplotlib.pyplot as plt
from HelperFuncs import *
from sklearn.decomposition import PCA

if __name__ == '__main__':
    data = load_data()
    labels = load_labels()
    u_labels = np.unique(labels).tolist()
    y = np.array([u_labels.index(l) for l in labels])

    print('{} samples in total'.format(data.shape[0]))
    print('{} genes in total'.format(data.shape[1]))
    for i, u_label in enumerate(u_labels):
        print('{} samples labeled {}'.format(sum(y==i),u_label.strip()))