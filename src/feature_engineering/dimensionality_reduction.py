from sklearn.decomposition import PCA
def perform_dimensionality_reduction(X_train, X_test):
    pca = PCA(n_components=5)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    return X_train_reduced, X_test_reduced