
#%%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from ae_source import *
import time


#%%
# Charge les données MNIST avec seulement les 1, 5 et 6
X_train = np.load('./data_files/mnist_train_images.npy')
X_train_clean = np.load('./data_files/mnist_train_images.npy')
X_test = np.load('./data_files/mnist_test_images.npy')
X_test_clean = np.load('./data_files/mnist_test_images.npy')

#%%
# bruite les images
n_train = X_train.shape[0]
n_test = X_test.shape[0]

# TODO : fonction
for idx in range(n_train):
    noise = np.random.normal(0,0.2,28*28)
    for i in range(28):
        for j in range(28) :
            X_train[idx][i*28+j]+=noise[i*28+j]

for idx in range(n_test):
    noise = np.random.normal(0,0.2,28*28)
    for i in range(28):
        for j in range(28) :
            X_test[idx][i*28+j]+=noise[i*28+j]


#%%
plot_image(X_train)

# %%
# Structure de l'autoencodeur

# Shape de l'entrée
n_input = 28*28

# Structure de l'encodeur
n_encoder1 = 500
n_encoder2 = 300

# dimension de la couche cachée, goulot d'étranglement
n_latent = 4 # dimension beaucoup plus petite

# Strucure du decodeur
n_decoder2 = 300
n_decoder1 = 500

# encodeur et decodeur = "meme dimension"

#%%
# Autoencodeur débruiteur 
DAE = DAEClassifier(n_input, n_encoder1, n_encoder2, n_latent, n_decoder2, n_decoder1)

# Training
start_time = time.time()
DAE.fit(X_train, X_train_clean) 
# images bruitées en entrée et non bruitées en sortie "attendues"
end_time = time.time()

time_training = end_time - start_time

print("Durée pour l'entrainement")
print(time_training/60)


#%%
# prend un indice aléatoirement pour avoir une images aléatoires
# parmi les données
idx = np.random.randint(X_test.shape[0])

# prediction : image reconstruite à partir d'une image floutée
X_reconst = DAE.predict(X_test[idx].reshape(-1,784))

# Plot des images : bruitée, reconstruite et originale
plot_reconst(idx, X_test, X_reconst, X_test_clean)

# %%
idx = np.random.randint(X_test.shape[0])

# prediction : image reconstruite à partir d'une image nette (originale)
X_reconst = DAE.predict(X_test_clean[idx].reshape(-1,784))

# Plot des images : originale, reconstruite et originale
plot_reconst(idx, X_test_clean, X_reconst, X_test_clean)
# %%
X_test = np.load('./data_files/mnist_test_images.npy')

for idx in range(n_test):
    noise = np.random.normal(0,1,28*28)
    for i in range(28):
        for j in range(28) :
            X_test[idx][i*28+j]+=noise[i*28+j]

#%%
idx = np.random.randint(X_test.shape[0])

# prediction : image reconstruite à partir d'une image très bruitée
X_reconst = DAE.predict(X_test[idx].reshape(-1,784))

# Plot des images : bruitée, reconstruite et originale
plot_reconst(idx, X_test, X_reconst, X_test_clean)
# %%
