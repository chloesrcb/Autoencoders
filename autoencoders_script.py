

# Some ideas
# blurry image to real image
# noisy digit data to unoisy digit


# sklearn 

import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score


# # Shape of input and latent variable

# n_input = 28*28

# # Encoder structure
# n_encoder1 = 500
# n_encoder2 = 300

# n_latent = 2

# # Decoder structure
# n_decoder2 = 300
# n_decoder1 = 500

# reg = MLPRegressor(
#     hidden_layer_sizes = (n_encoder1, n_encoder2, 
#                           n_latent, n_decoder2,
#                           n_decoder1), 
#     activation = 'tanh', 
#     solver = 'adam', 
#     learning_rate_init = 0.0001, 
#     max_iter = 20, 
#     tol = 0.0000001, 
#     verbose = True
#     )

# MLP : Multi-layer Perceptron classifier
# This model optimizes the log-loss function
# using LBFGS or stochastic gradient descent.

# Encoder
# simple ANN (MLP) model
# use tanh for a nonlinear activation function
# latent is not applied with a nonlinear activation function

# Decoder
# Simple ANN (MLP) model
# use tanh for a nonlinear activation function
# reconst is not applied with a nonlinear activation function