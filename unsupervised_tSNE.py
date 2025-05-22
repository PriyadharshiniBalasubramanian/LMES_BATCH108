# Import libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import seaborn as sns

# 1. Load the MNIST small digits dataset
digits = load_digits()
X = digits.data         # Features (64 features: 8x8 image flattened)
y = digits.target       # Labels (0-9)

print("Shape of X:", X.shape)  # (1797, 64)
print("Shape of y:", y.shape)  # (1797,)

# 2. Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X)

print("Shape after t-SNE:", X_embedded.shape)  # (1797, 2)

# 3. Plot the results
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y, palette="tab10", s=60, legend="full")
plt.title('t-SNE visualization of MNIST Digits')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(title='Digit', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
