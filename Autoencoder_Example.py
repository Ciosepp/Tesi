#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eliafavarelli

"""
import sys

import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Generare dati di esempio (dati normali)
def generate_normal_data(n_samples=1000):
    X = np.random.normal(loc=0, scale=1, size=(n_samples, 10))
    return X

# Generare anomalie (dati anomali)
def generate_anomalies(n_samples=100):
    anomalies = np.random.normal(loc=5, scale=1, size=(n_samples, 10))  # Anomalie con distribuzione diversa
    return anomalies

plt.close('all')


# Dati di addestramento (solo dati normali)
X_train = generate_normal_data(1000)

# Dati di test (dati normali + anomalie)
X_test_normal = generate_normal_data(200)
X_test_anomalies = generate_anomalies(50)
X_test = np.concatenate([X_test_normal, X_test_anomalies])

# Etichette vere dei dati di test (0 per normale, 1 per anomalia)
y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_test_anomalies))])

# Normalizzare i dati
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Costruzione dell'autoencoder
input_dim = X_train_scaled.shape[1]
encoding_dim = 5  # Dimensione del codice latente

autoencoder = models.Sequential([
    layers.InputLayer(input_shape=(input_dim,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(encoding_dim, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(input_dim, activation="sigmoid")  # Output riconstruito
])

autoencoder.compile(optimizer='adam', loss='mse')

# Addestramento dell'autoencoder
history = autoencoder.fit(X_train_scaled, X_train_scaled, epochs=200, batch_size=32, validation_split=0.2, verbose=0)

# Funzione per calcolare l'errore di ricostruzione
def compute_reconstruction_error(X, model):
    X_pred = model.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=1)
    return mse

# Calcolare gli errori di ricostruzione su dati normali e anomali
train_error = compute_reconstruction_error(X_train_scaled, autoencoder)
test_error = compute_reconstruction_error(X_test_scaled, autoencoder)

# Definire una soglia per il rilevamento delle anomalie (ad esempio, il 99° percentile dell'errore sui dati di training)
threshold = np.percentile(train_error, 99)

# Classificare le anomalie
test_predictions = (test_error > threshold).astype(int)

plt.figure()

# Visualizzare i risultati
plt.hist(test_error, bins=50, alpha=0.7, label='Test Error')
plt.hist(train_error, bins=50, alpha=0.7, label='Training Error')
plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
plt.legend()
plt.show()


# Visualizzare la matrice di confusione
conf_matrix = confusion_matrix(y_test, test_predictions)

# Plot della matrice di confusione
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normale', 'Anomalia'], yticklabels=['Normale', 'Anomalia'])
plt.title('Matrice di Confusione')
plt.xlabel('Predizioni')
plt.ylabel('Valori Veri')
plt.show()

# Calcolare le metriche di valutazione
accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions)
recall = recall_score(y_test, test_predictions)
f1 = f1_score(y_test, test_predictions)

print(f"Accuratezza: {accuracy:.2f}")
print(f"Precisione: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")