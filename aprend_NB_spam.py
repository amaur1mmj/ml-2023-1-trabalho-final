import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Carregar os dados de treinamento
data = pd.read_csv('spam_ham_dataset.csv')

# Separar as features (mensagens de e-mail) e os rótulos (spam ou não spam)
X = data['text']
y = data['label_num']

# Dividir os dados em conjunto de treinamento e conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pré-processamento dos dados: vetorização das mensagens de e-mail
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Inicializar listas para armazenar dados complementares
train_accuracies = []
test_accuracies = []

# Treinar o modelo Naive Bayes e acompanhar o desempenho durante o treinamento
model = MultinomialNB()
for i in range(1, 11):
    model.partial_fit(X_train_vectorized, y_train, classes=[0, 1])
    train_pred = model.predict(X_train_vectorized)
    train_accuracy = accuracy_score(y_train, train_pred)
    train_accuracies.append(train_accuracy)

    X_test_vectorized = vectorizer.transform(X_test)
    test_pred = model.predict(X_test_vectorized)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_accuracies.append(test_accuracy)

    print("Epoch", i, "- Acurácia de treinamento:", train_accuracy, "Acurácia de teste:", test_accuracy)

# Plotar o gráfico de linha com as acurácias
epochs = range(1, 11)
plt.plot(epochs, train_accuracies, label='Treinamento')
plt.plot(epochs, test_accuracies, label='Teste')
plt.xlabel('Epoch')
plt.ylabel('Acurácia')
plt.title('Acurácia durante o treinamento')
plt.legend()
plt.show()

# Plotar a matriz de confusão do conjunto de teste
from sklearn.metrics import confusion_matrix
import seaborn as sns

confusion_mtx = confusion_matrix(y_test, test_pred)
sns.heatmap(confusion_mtx, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Classe Prevista')
plt.ylabel('Classe Verdadeira')
plt.title('Matriz de Confusão')
plt.show()

# Exemplo de previsão para uma mensagem de teste
mensagem_teste = ["Oferta imperdível! Ganhe dinheiro rápido!"]
mensagem_teste_vectorized = vectorizer.transform(mensagem_teste)
previsao = model.predict(mensagem_teste_vectorized)
probabilidades = model.predict_proba(mensagem_teste_vectorized)[0]
chance_spam = probabilidades[1]  # Probabilidade de ser spam
chance_nao_spam = probabilidades[0]  # Probabilidade de não ser spam

print("Previsão:", "spam" if previsao[0] == 1 else "não spam")
print("Chances de ser spam:", chance_spam)
print("Chances de não ser spam:", chance_nao_spam)
