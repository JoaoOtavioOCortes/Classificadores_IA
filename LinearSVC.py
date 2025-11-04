import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC  
(XTreino, yTreino), (XTeste, yTeste) = keras.datasets.cifar10.load_data()

XTreino_reashaped = XTreino.reshape((XTreino.shape[0], -1))
XTeste_reashaped = XTeste.reshape((XTeste.shape[0], -1))

XTreino_normalized = XTreino_reashaped / 255.0
XTeste_normalized = XTeste_reashaped / 255.0

yTreino_flat = yTreino.flatten()
yTeste_flat = yTeste.flatten()

clf2 = LinearSVC(verbose=1)

clf2.fit(XTreino_normalized, yTreino_flat)

previsto= clf2.predict(XTeste_normalized)
esperado = yTeste_flat
nome = ["Avião", "Automóvel", "Pássaro", "Gato", "Veado", "Cachorro", "Sapo", "Cavalo", "Navio", "Caminhão"]
print(classification_report(esperado, previsto, target_names=nome))


