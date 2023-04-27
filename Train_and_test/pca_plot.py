def pca_plot(data1, data2, figure_num = 1, title_name = "Default"):
    X = np.vstack((data1, data2))
    sample_pair = [("PET", 1), ("PP", 5)]
    Y = np.hstack((np.repeat(1, data1.shape[0]), np.repeat(5, data2.shape[0])))

    fig = plt.figure(figure_num, figsize=(4, 3))
    fig.suptitle(title_name)
    ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    for name, label in sample_pair:
        ax.text3D(
            X[Y == label, 0].mean(),
            X[Y == label, 1].mean() + 1.5,
            X[Y == label, 2].mean(),
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
        )
    # Reorder the labels to have colors matching the cluster results
    # y = np.choose(Y, [1, 2, 0]).astype(float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, edgecolor="k")

    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    # ax.zaxis.set_ticklabels([])


    X2 = np.vstack((data1/5, data2))
    fig2 = plt.figure(2, figsize=(4, 3))
    ax2 = fig2.add_subplot(111, projection="3d", elev=48, azim=134)
    pca = decomposition.PCA(n_components=3)
    pca.fit(X2)
    X2 = pca.transform(X2)

    for name, label in sample_pair:
        ax.text3D(
            X2[Y == label, 0].mean(),
            X2[Y == label, 1].mean() + 1.5,
            X2[Y == label, 2].mean(),
            name,
            horizontalalignment="center",
            bbox=dict(alpha=0.5, edgecolor="w", facecolor="w"),
        )
    # Reorder the labels to have colors matching the cluster results
    # y = np.choose(Y, [1, 2, 0]).astype(float)
    ax2.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c=Y, edgecolor="k")

    # ax.xaxis.set_ticklabels([])
    # ax.yaxis.set_ticklabels([])
    # ax.zaxis.set_ticklabels([])