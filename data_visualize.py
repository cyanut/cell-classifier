import numpy as np
import matplotlib.pyplot as plt
from cell_classifier import get_raw_data, get_fields, get_args
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    rand_state = 42
    
    args = get_args()
    raw_titles, raw_data = get_raw_data(args.input)
    print(raw_titles)
    feature_title, X = get_fields(raw_titles, raw_data, args.features)
    target_title, y = get_fields(raw_titles, raw_data, [args.target])
    X = X.astype(float)
    y = y.astype(float)

    y_prime = np.zeros_like(y)
    if args.benchmark:
        benchmark_title, y_prime = get_fields(\
                raw_titles, raw_data, [args.benchmark])
        le = LabelEncoder()
        le.fit(["False", "True"])
        y_prime = le.transform(y_prime[:,0])
    else:
        y_prime = y_prime[:,0] # added by PS to make shapes match

    y = y[:,0]


    normalizer = StandardScaler()
    normalizer.fit(X)
    X_norm = normalizer.transform(X)
    
    pca = PCA(n_components=2)
    pca.fit(X_norm[y>-1])
    print(pca.components_)
    print(pca.explained_variance_ratio_)
    fig = plt.figure()
    pca3 = PCA(n_components=3)
    pca3.fit(X_norm)
    ax = fig.add_subplot(111, projection="3d")
    t = pca3.transform(X_norm[y==1])
    ax.scatter(t[:,0], t[:,1], t[:,2], s=1, c="r")
    f = pca3.transform(X_norm[y==0])
    ax.scatter(f[:,0], f[:,1], f[:,2], s=1, c="b")
    n = pca3.transform(X_norm[y==-1])
    ax.scatter(n[:,0], n[:,1], n[:,2], s=1, c="gray")
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.set_title("LDA trained on 1 vs 0")
    lda = LDA()
    lda = lda.fit(X_norm[y>-1], y[y>-1])
    print(zip(feature_title,lda.coef_[0]))
    t = lda.transform(X_norm[y==1])
    print(t.shape)
    f = lda.transform(X_norm[y==0])
    n = lda.transform(X_norm[y==-1])
    ax1.hist(t,50,facecolor="r",alpha=.5)
    ax1.hist(f,50,facecolor="b",alpha=.5)
    ax1.hist(n,50,facecolor="k",alpha=.5)
    ax1.legend()

    ax2 = fig.add_subplot(312)
    ax2.set_title("LDA trained on 1 vs (0 + -1)")
    lda = LDA()
    lda = lda.fit(X_norm, y>0)
    print(zip(feature_title,lda.coef_[0]))
    t = lda.transform(X_norm[y==1])
    print(t.shape)
    f = lda.transform(X_norm[y==0])
    n = lda.transform(X_norm[y==-1])
    ax2.hist(t,50,facecolor="r",alpha=.5)
    ax2.hist(f,50,facecolor="b",alpha=.5)
    ax2.hist(n,50,facecolor="k",alpha=.5)

    ax3 = fig.add_subplot(313)
    ax3.set_title("Projection of benchmark on LDA axis")
    print(y_prime)
    t = lda.transform(X_norm[y_prime==1])
    f = lda.transform(X_norm[y_prime==0])
    ax3.hist(t,50,facecolor="r",alpha=.5)
    ax3.hist(f,50,facecolor="b",alpha=.5)
    plt.tight_layout()
    plt.show()

    
