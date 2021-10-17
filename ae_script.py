
#%%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from ae_source import *
import time


#%%
# Charge les données MNIST avec seulement les 1, 5 et 6
#X_train = np.load('./data_files/mnist_train_images.npy')
X_train_clean = np.load('./data_files/mnist_train_images.npy')
#X_test = np.load('./data_files/mnist_test_images.npy')
X_test_clean = np.load('./data_files/mnist_test_images.npy')

#%%
# # bruite les images

X_train = noiser(X_train_clean, 0.2)
X_test = noiser(X_test_clean, 0.2)

#%%
plot_image(X_train)

# %%
# Structure de l'autoencodeur

# Shape de l'entrée
n_input = 28*28 # 784

# Structure de l'encodeur
n_encoder1 = 500
n_encoder2 = 300

# dimension de la couche cachée, goulot d'étranglement
# dimension plus petite 
n_latent1 = 50  # pour débruiter l'image

n_latent2 = 4 # pour déterminer la classe

# Strucure du decodeur
n_decoder2 = 300
n_decoder1 = 500

# encodeur et decodeur = "meme dimension"

#%%
# Autoencodeur débruiteur 
DAE_50 = DAEClassifier(n_input, n_encoder1, n_encoder2, n_latent1, n_decoder2, n_decoder1)
DAE_4 = DAEClassifier(n_input, n_encoder1, n_encoder2, n_latent2, n_decoder2, n_decoder1)

#%%
# Training avec couche cachée de 50 neurones
start_time = time.time()
DAE_50.fit(X_train, X_train) 
# images bruitées en entrée et non bruitées en sortie "attendues"
end_time = time.time()

time_training = end_time - start_time

print("Durée pour l'entrainement")
print(time_training/60)

#%%
# Training avec couche cachée de 4 neurones
start_time = time.time()
DAE_4.fit(X_train, X_train) 
# images bruitées en entrée et non bruitées en sortie "attendues"
end_time = time.time()

time_training = end_time - start_time

print("Durée pour l'entrainement")
print(time_training/60)


#%%
# prend un indice aléatoirement pour avoir une image aléatoire
# parmi les données
idx = np.random.randint(X_test.shape[0])

# prediction : image reconstruite à partir d'une image floutée
X_reconst = DAE_50.predict(X_test[idx].reshape(-1,784))

# Plot des images : bruitée, reconstruite et originale
plot_reconst(idx, X_test, X_reconst, X_test_clean)


#%%
# prend un indice aléatoirement pour avoir une image aléatoire
# parmi les données
idx = np.random.randint(X_test.shape[0])

# prediction : image reconstruite à partir d'une image floutée
X_reconst = DAE_4.predict(X_test[idx].reshape(-1,784))

# Plot des images : bruitée, reconstruite et originale
plot_reconst(idx, X_test, X_reconst, X_test_clean)

#%%
# comparaison pour une meme image
idx = np.random.randint(X_test.shape[0])

X_reconst1 = DAE_50.predict(X_test[idx].reshape(-1,784))

X_reconst2 = DAE_4.predict(X_test[idx].reshape(-1,784))

plot_reconst_comparaison(idx, X_test, X_reconst1, X_reconst2, X_test_clean)

# %%
idx = np.random.randint(X_test.shape[0])

# prediction : image reconstruite à partir d'une image nette (originale)
X_reconst1 = DAE_50.predict(X_test_clean[idx].reshape(-1,784))

# prediction : image reconstruite à partir d'une image nette (originale)
X_reconst2 = DAE_4.predict(X_test_clean[idx].reshape(-1,784))

# Plot comparaison des images : originale, reconstruite et originale
plot_reconst_comparaison(idx, X_test_clean, X_reconst1, X_reconst2, X_test_clean)

# %%

# images test avec un bruit trop important
X_test = noiser(X_test_clean, 1)

#%%
idx = np.random.randint(X_test.shape[0])

# prediction : image reconstruite à partir d'une image très bruitée
X_reconst1 = DAE_50.predict(X_test[idx].reshape(-1,784))

# prediction : image reconstruite à partir d'une image très bruitée
X_reconst2 = DAE_4.predict(X_test[idx].reshape(-1,784))

# Plot comparaison des images : bruitée, reconstruite et originale
plot_reconst_comparaison(idx, X_test, X_reconst1, X_reconst2, X_test_clean)


# %%
