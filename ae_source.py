#%%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score


def noiser(X, sigma):
    # bruite X avec un bruit gaussien de variance sigma
    n = X.shape[0]
    X_noisy = X.copy()
    for idx in range(n):
        noise = np.random.normal(0.5, sigma, 28*28)
        for i in range(28):
            for j in range(28):
                X_noisy[idx][i * 28 + j] += noise[i * 28 + j]
    return(X_noisy)


def plot_reconst(idx, X_noisy, X_reconst, X_original):
    plt.figure(figsize = (15,8))
    plt.subplot(1,3,1)
    plt.imshow(X_noisy[idx].reshape(28,28), 'gray')
    plt.title('Image en entrée', fontsize = 15)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,2)
    plt.imshow(X_reconst.reshape(28,28), 'gray')
    plt.title('Image reconstruite', fontsize = 15)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,3,3)
    plt.imshow(X_original[idx].reshape(28,28), 'gray')
    plt.title('Image originale', fontsize = 15)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_reconst_comparaison(idx, X_noisy, X_reconst1, X_reconst2, X_original):
    plt.figure(figsize = (15,8))
    plt.subplot(1,4,1)
    plt.imshow(X_noisy[idx].reshape(28,28), 'gray')
    plt.title('Image en entrée', fontsize = 15)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,4,2)
    plt.imshow(X_reconst1.reshape(28,28), 'gray')
    plt.title('Image reconstruite \n 50 neurones', fontsize = 15)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,4,3)
    plt.imshow(X_reconst2.reshape(28,28), 'gray')
    plt.title('Image reconstruite \n 4 neurones', fontsize = 15)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1,4,4)
    plt.imshow(X_original[idx].reshape(28,28), 'gray')
    plt.title('Image originale', fontsize = 15)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_image(X):
    idx = np.random.randint(X.shape[0])
    img = X[idx].reshape(28,28)

    plt.figure(figsize = (6,6))
    plt.imshow(img,'gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def DAEClassifier(n_input, n_encoder1, n_encoder2, n_latent, n_decoder2, n_decoder1):
    # 
    DAE = MLPRegressor(hidden_layer_sizes = (n_encoder1, n_encoder2, n_latent, n_decoder2, n_decoder1), 
                   activation = 'tanh', 
                   solver = 'adam', 
                   learning_rate_init = 0.0001, 
                   max_iter = 20, 
                   tol = 0.0000001, 
                   verbose = True)
    return(DAE)
# %%
