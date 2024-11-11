import sys

import numpy as np
import tensorflow as tf
from keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio
import cartopy.crs as ccrs

from scipy import signal

import time

#creazione dataset di elaborazione
percentAcqTrain = 0.4 # %/100
nSamples = 100
nPWelch =100
fs = 100

saveFlag =True

showLoss = False
showErr = False

# Carica i file .mat
#data = sio.loadmat('U1_S1_M1load.mat')
#U1 = Utente 1
#U2 = Utente 2
#S1 = Strada 1 (dissestata)
#S2 = Strada 2 (liscia)
#M1 = Monopattino 1 (sano)
#M2 = Monopattino 2 (danneggiato)
inDataConstrainSize = 63000
Accelerations = np.zeros((2,2,2,inDataConstrainSize,4))

for u in range(1,3):
    for s in range(1,3):
        for m in range(1,3):
            fileName = "analisys\\U" + str(u) + "_S" + str(s) + "_M" + str(m) + "load.mat"

            print(fileName + ' loaded')
            
            data = sio.loadmat(fileName)
            Acceleration = data['acceleration']
            
            print(fileName+' size: '+str(data['acceleration'].shape[0]))
            
            Accelerations[u-1,s-1,m-1,:,:] = Acceleration[:63000,:]

#print(data.keys())
#Acceleration = data['acceleration']
#print(data['acceleration'].shape)
#print(Acceleration[3])


#dataset di traing
trainData=[]
trainDataMeta=[]
nPackets = int(inDataConstrainSize/nSamples)

print('nPackets: '+str(nPackets))
print(f'dataset width:{(nPWelch/2+1)*3}')
for u in range(0,2):
	for i in range(0, int(nPackets*percentAcqTrain)):
		acqPower= []
		
		for j in range(3): # X Y Z
		
			acc = Accelerations[u, 1, 0, i*nSamples:nSamples*(i+1), j+1]
			f, Pacc = signal.welch(acc, fs, nperseg=nPWelch)   
			acqPower = np.append(acqPower, Pacc)

		trainDataMeta.append(acqPower)

trainData = np.stack(trainDataMeta, axis=0)

print(str(int(nPackets*percentAcqTrain))+'->'+str(int(nPackets*percentAcqTrain)*2))
print(trainData.shape)

##dataset di test
percentAcqTest=1-percentAcqTrain
testDataNormal =[]
testDataNormalMeta =[]


for u in range(2):
	for i in range(int(nPackets*percentAcqTrain), nPackets):
		acqPower= []
		
		for j in range(3): # X Y Z
			acc = Accelerations[u, 1, 0, i*nSamples:nSamples*(i+1), j+1]
			f, Pacc = signal.welch(acc, fs, nperseg=nPWelch)   
			acqPower = np.append(acqPower, Pacc)
		
		testDataNormalMeta.append(acqPower)

testDataNormal = np.stack(testDataNormalMeta)

print(str(int(nPackets*percentAcqTest))+'->'+str(int(nPackets*percentAcqTest)*2))
print(testDataNormal.shape)

#dataset anomalo
percentAcqAnom = 1
testDataAnomalies =[]
testDataAnomaliesMeta =[]

for u in range(2):
	for i in range(0,int(nPackets*percentAcqAnom)):
		acqPower= []
		for j in range(3): # X Y Z
			acc = Accelerations[u, 0, 0, i*nSamples:nSamples*(i+1), j+1]
			f, Pacc = signal.welch(acc, fs, nperseg=nPWelch)   
			acqPower = np.append(acqPower, Pacc)
		
		testDataAnomaliesMeta.append(acqPower)

testDataAnomalies = np.stack(testDataAnomaliesMeta)
print(str(int(nPackets*percentAcqAnom))+'->'+str(int(nPackets*percentAcqAnom)*2))
print(testDataAnomalies.shape)

#creazione dataset di test concatenando dati normali e anomali
testData = np.concatenate([testDataNormal, testDataAnomalies])

#etichette dati test
#(0 per normale, 1 per anomalia)
dataLabel = np.concatenate([np.zeros(len(testDataNormal)), np.ones(len(testDataAnomalies))])

# Normalizzare i dati
scaler = MinMaxScaler()
trainData_scaled = scaler.fit_transform(trainData)
testData_scaled = scaler.transform(testData)

# Costruzione dell'autoencoder
input_dim = trainData_scaled.shape[1]
encoding_dim = 5  # Dimensione del codice latente 5

autoencoder = models.Sequential([
    layers.InputLayer(input_shape=(input_dim,)), 
	layers.Dense(8, activation="relu"),
    layers.Dense(encoding_dim, activation="relu"),
    layers.Dense(5, activation="relu"),
    layers.Dense(input_dim, activation="sigmoid")  # Output riconstruito
])
autoencoder.compile(optimizer='adam', loss='mse')

# Addestramento dell'autoencoder
history = autoencoder.fit(trainData_scaled, trainData_scaled, epochs=200, batch_size=32, validation_split=0.2, verbose=0)

# Funzione per calcolare l'errore di ricostruzione
def compute_reconstruction_error(X, model):
    X_pred = model.predict(X)
    mse = np.mean(np.power(X - X_pred, 2), axis=1)
    return mse

# Calcolare gli errori di ricostruzione su dati normali e anomali
train_error = compute_reconstruction_error(trainData_scaled, autoencoder)
test_error = compute_reconstruction_error(testData_scaled, autoencoder)

# Definire una soglia per il rilevamento delle anomalie (ad esempio, il 99° percentile dell'errore sui dati di training)
threshold = np.percentile(train_error, 99)
print(threshold)

# Classificare le anomalie
test_predictions = (test_error > threshold).astype(int)


if(showLoss):
	# Grafico della loss
	plt.figure()
	plt.plot(history.history['loss'], label='Training Loss')
	plt.plot(history.history['val_loss'], label='Validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.title('Andamento della Loss durante il Training')

'''
NormalSampletest = np.expand_dims(testData_scaled[0], axis=0)
AnomalieSampletest = np.expand_dims(testData_scaled[int(testData_scaled.shape[0] *0.9)], axis=0)
print(int(testData_scaled.shape[1] *0.9))
NormalSampleNet = autoencoder.predict(NormalSampletest)
AnomalieSampleNet = autoencoder.predict(AnomalieSampletest)

mseN = np.mean(np.power(NormalSampletest - NormalSampleNet, 2), axis=1)
mseA = np.mean(np.power(AnomalieSampletest - AnomalieSampleNet, 2), axis=1)

NormalSampletest = NormalSampletest.squeeze()
AnomalieSampletest = AnomalieSampletest.squeeze()
NormalSampleNet = NormalSampleNet.squeeze()
AnomalieSampleNet = AnomalieSampleNet.squeeze()

# Crea la figura con due grafici affiancati
fig, axes = plt.subplots(2, 1, figsize=(14, 14))

# Grafico per il campione normale
axes[0].plot(NormalSampletest, label="Ingresso - Normale", color="blue")
axes[0].plot(NormalSampleNet, label="Predizione - Normale", color="orange")
axes[0].set_title("Campione Normale")
axes[0].set_xlabel("Indice")
axes[0].set_ylabel("Valore")
axes[0].legend()

# Grafico per il campione anomalo
axes[1].plot(AnomalieSampletest, label="Ingresso - Anomalia", color="blue")
axes[1].plot(AnomalieSampleNet, label="Predizione - Anomalia", color="orange")
axes[1].set_title("Campione Anomalo")
axes[1].set_xlabel("Indice")
axes[1].legend()

# Mostra i grafici
plt.tight_layout()
plt.show()

#plt.plot(train_error)
#plt.plot(test_error)
'''

if(showErr):
	# Visualizzare i risultati
	plt.figure()
	plt.hist(test_error, bins=80, alpha=0.4, label='Test Error')
	plt.hist(train_error, bins=80, alpha=0.8, label='Training Error')
	plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
	plt.legend()



# Visualizzare la matrice di confusione
conf_matrix = confusion_matrix(dataLabel, test_predictions)

# Plot della matrice di confusione
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normale', 'Anomalia'], yticklabels=['Normale', 'Anomalia'])
plt.title(f"Matrice di Confusione n:{nSamples}  w:{nPWelch}")
plt.xlabel('Predizioni')
plt.ylabel('Valori Veri')
if(saveFlag):
	plt.savefig(f"MC n_{nSamples}  w_{nPWelch}")


# Calcolare le metriche di valutazione
accuracy = accuracy_score(dataLabel, test_predictions)
precision = precision_score(dataLabel, test_predictions)
recall = recall_score(dataLabel, test_predictions)
f1 = f1_score(dataLabel, test_predictions)

print(f"Accuratezza: {accuracy:.2f}")
print(f"Precisione: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
recall_array = []
p_fd = []

'''
start_time = time.time()
for th in np.arange(0, threshold, threshold/10000):
    
    test_predictions = (test_error > th).astype(int)
    recall_array.append( recall_score(dataLabel, test_predictions))
    conf_matrix = confusion_matrix(dataLabel, test_predictions)
    p_fd.append(conf_matrix[0,1]/(conf_matrix[0,1]+conf_matrix[1,1]))

execution_time = time.time() - start_time
print(f"Tempo di esecuzione: {execution_time} secondi")
'''

# Calcola i valori di FPR e TPR
fpr, tpr, thresholds = roc_curve(dataLabel, test_predictions)

# Calcola l'AUC
auc_score = roc_auc_score(dataLabel, test_predictions)

plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC n:"+str(nSamples)+'  w:'+str(nPWelch))
plt.legend(loc="lower right")

if(saveFlag):
	plt.savefig(f"ROC n{nSamples} w{nPWelch}.png")
plt.show()

##p_fd = np.array(p_fd)
#recall_array= np.array(recall_array)


