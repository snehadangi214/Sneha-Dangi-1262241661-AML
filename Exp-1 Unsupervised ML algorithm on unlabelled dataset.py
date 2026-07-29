import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = pd.read_csv("Wine dataset.csv")

X = wine.drop("target", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_data = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

print("Original Shape:", X.shape)
print("Reduced Shape:", X_pca.shape)

print(pca_data)
