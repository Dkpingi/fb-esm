
import torch
import esm
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


# Load ESM-1b model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

for param in model.parameters():
    if param.size() == torch.Size((33,1280)):
        last_matrix = param


last_matrix = last_matrix[4:24,:]
X = np.array(last_matrix.detach())
#scaler = StandardScaler()
#X = scaler.fit_transform(X)

pca = PCA(n_components=19)
X_embedded = pca.fit_transform(X)
X_embedded = TSNE(n_components=2,perplexity=2).fit_transform(X_embedded) #1.5-3 was best so far

tokens = alphabet.standard_toks[:20]
fig, ax = plt.subplots()

ax.scatter(X_embedded[:,0],X_embedded[:,1])

for i, txt in enumerate(tokens):
    ax.annotate(txt, (X_embedded[i,0], X_embedded[i,1]))

plt.savefig("tsne.png")


