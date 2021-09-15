import matplotlib.pyplot as plt
from HelperFuncs import *
from sklearn.decomposition import PCA

if __name__ == '__main__':
    data = load_data()
    labels = load_labels()
    u_labels = np.unique(labels).tolist()
    y = np.array([u_labels.index(l) for l in labels])

    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(data)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    colors = ['r','b','g','m','c']

    for i in range(len(u_labels)):
        ax.scatter(x_pca[y==i,0],x_pca[y==i,1],x_pca[y==i,2],c=colors[i],label=u_labels[i])
    ax.legend()
    plt.show()


