import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from HelperFuncs import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

def find_num_clusters(data,low=2,high=10):
    Ks = []
    ar_results = []
    for k in range(low,high+1):
        # print(k)
        Ks.append(k)
        ar_results.append([])
        for rs in range(0,100,5):
            clf = KMeans(k, random_state=rs)
            y_pred = clf.fit_predict(data)
            ar_results[-1].append(silhouette_score(data,y_pred))
        ar_results[-1] = np.mean(ar_results[-1])
    return Ks[np.argmax(ar_results)]


if __name__ == '__main__':
    data = load_data()
    labels = load_labels()
    u_labels = np.unique(labels).tolist()
    y = np.array([u_labels.index(l) for l in labels])

    print(data.shape)
    print('{} unique labels'.format(len(u_labels)))

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size = 0.33, random_state = 42)

    # find number of clusters blindly
    k = find_num_clusters(X_train)
    print('Estimated Number of clusters:',k)

    clf = KMeans(5,random_state=42)
    clf.fit(X_train)

    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print("Adjusted rand index for cluster prediction is: ",adjusted_rand_score(y_test,y_pred))