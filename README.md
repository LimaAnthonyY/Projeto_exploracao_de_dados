# Projeto de Modelagem Preditiva: Nestlé, Santander e Diversos Bancos

## 1. Projeto: Predição de Preço de Ações da Nestlé

### 1.1 Coleta e Carregamento dos Dados

Utilizando os dados históricos de preços das ações da Nestlé a partir da API ou Kaggle. Neste caso de um csv do Kaggle.

```python
import pandas as pd

# Carregando os dados da Nestlé
df_nestle = pd.read_csv('nestle_stock_data.csv')

# Exibindo as primeiras linhas do dataset
print(df_nestle.head())
```

### 1.2 Análise Exploratória dos Dados (EDA)

Fazemos uma análise exploratória dos dados para entender as tendências de preço e comportamento das ações.

```python
# Estatísticas descritivas
print(df_nestle.describe())

# Visualização da tendência de preço ao longo do tempo
import matplotlib.pyplot as plt

plt.plot(df_nestle['Date'], df_nestle['Close Price'])
plt.title('Preço de Fechamento das Ações da Nestlé')
plt.xlabel('Data')
plt.ylabel('Preço de Fechamento')
plt.show()
```

### 1.3 Pré-processamento dos Dados

Realizamos o tratamento de valores ausentes e ajuste as variáveis de interesse.

```python
# Verificando valores nulos e preenchendo-os, se necessário
df_nestle.fillna(method='ffill', inplace=True)

# Convertendo datas
df_nestle['Date'] = pd.to_datetime(df_nestle['Date'])
```

### 1.4 Modelagem com Regressão Linear

Utilizaremos a Regressão Linear para prever o preço das ações com base nas variáveis históricas.

Primeiro corrigimos as colunas para melhor visualização

```python
# Renomeando as colunas para remover espaços e torná-las minúsculas
df_nestle.columns = df_nestle.columns.str.strip().str.replace(' ', '_').str.lower()
df_nestle.columns = df_nestle.columns.str.strip().str.replace(' ', '_').str.lower()
df_nestle.columns = df_nestle.columns.str.replace('.', '_').str.replace('_of_', 'of_')  # Substitui o ponto e padroniza

# Renomeando a coluna para um formato mais amigável
df_nestle.rename(columns={'%_deli__qty_to_traded_qty': 'percent_deliverable_qty'}, inplace=True)
```

E com isso fazemos a Regressão Linear

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Variáveis independentes e dependente
X = df_nestle[['open_price', 'high_price', 'low_price', 'noof_shares']]
y = df_nestle['close_price']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciando o modelo de regressão linear
modelo_lr = LinearRegression()
modelo_lr.fit(X_train, y_train)

# Prevendo preços
y_pred = modelo_lr.predict(X_test)

# Avaliando o modelo
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')
```

### 1.5 Visualização das Previsões

Comparamos as previsões do modelo com os valores reais.

```python
# Criando duas subplots
fig, axs = plt.subplots(2, 1, figsize=(14, 6))

# Gráfico 1: Valores reais
axs[0].plot(y_test.values, label='Real', color='blue')
axs[0].set_title('Preços Reais da Nestlé')
axs[0].set_xlabel('Observações')
axs[0].set_ylabel('Preço de Fechamento')
axs[0].legend()

# Gráfico 2: Valores previstos
axs[1].plot(y_pred, label='Previsto', color='orange')
axs[1].set_title('Preços Previsto da Nestlé')
axs[1].set_xlabel('Observações')
axs[1].set_ylabel('Preço de Fechamento')
axs[1].legend()
plt.tight_layout()
plt.show()

# Gráfico 3: Fusão dos 2
plt.figure(figsize=(14, 6))
plt.plot(y_test.values, label='Real')
plt.plot(y_pred, label='Previsto')
plt.title('Previsão de Preços da Nestlé')
plt.xlabel('Observações')
plt.ylabel('Preço de Fechamento')
plt.legend()
plt.show()

```

### 1.6 Salvando o train em um CSV

E por fim salvamos o modelo Train, para poder utilizar em outro CSV ou em uma API

```python
# Criando um DataFrame para salvar os dados
train_data = pd.DataFrame(X_train, columns=['open_price', 'high_price', 'low_price', 'noof_shares'])
train_data['close_price'] = y_train.values

test_data = pd.DataFrame(X_test, columns=['open_price', 'high_price', 'low_price', 'noof_shares'])
test_data['predicted_close_price'] = y_pred

# Concatenando os dados de treinamento e teste
combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)

# Salvando em um arquivo CSV
combined_data.to_csv('modelo_train_data_nestle.csv', index=False)
```

---

## 2. Projeto: Predição de Churn de Clientes Santander

### 2.1 Coleta e Carregamento dos Dados

Carregamos um dataset de churn do Santander para prever a saída de clientes.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Carregando o dataset de churn do Santander
df = pd.read_csv('santander_churn.csv')

# Exibindo as primeiras linhas do dataset
df.head()
```

### 2.2 Análise do DF Santander

Verificamos as principais informações contidas no DF

```python
# Exibir as 5 primeiras linhas do dataset
print("Primeiras 5 linhas do dataset:")
print(df.head())

# Informações gerais sobre o dataset
print("\nInformações gerais do dataset:")
print(df.info())

# Resumo estatístico das variáveis numéricas
print("\nResumo estatístico das variáveis numéricas:")
print(df.describe())

# Verificar se há valores ausentes
print("\nValores ausentes por coluna:")
print(df.isnull().sum())

# Distribuição das classes no target (por exemplo, churn)
print("\nDistribuição das classes no target:")
print(df['Exited'].value_counts())

```

### 2.3 Análise Exploratória dos Dados (EDA)

Verificamos as características dos clientes e o comportamento relacionado ao churn.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualização da distribuição de churn
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Exited', data=df)
plt.title('Distribuição de Churn')
ax.set_xticklabels(['Não', 'Sim'])
ax.set_xlabel('')
ax.set_ylabel('') 

plt.show()


# Visualização das correlações entre as variáveis
plt.figure(figsize=(12, 8))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()  # Seleciona apenas colunas numéricas
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação')
plt.show()

# Histogramas das variáveis numéricas
df.hist(bins=30, figsize=(15, 10), color='blue', edgecolor='black')
plt.suptitle('Distribuição das Variáveis Numéricas', fontsize=16)
plt.show()

# Boxplots para detectar outliers nas variáveis numéricas
plt.figure(figsize=(15, 8))
sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']))
plt.title('Boxplots das Variáveis Numéricas para Detecção de Outliers')
plt.xticks(rotation=90)
plt.show()

# Análise da relação entre churn e outras variáveis categóricas
categorical_columns = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

plt.figure(figsize=(14, 10))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=col, hue='Exited', data=df)
    plt.title(f'Distribuição de Churn por {col}')
    plt.legend(title='Churn', loc='upper right', labels=['Não', 'Sim'])
plt.tight_layout()
plt.show()

```

### 2.4 Pré-processamento dos Dados

Limpamos e Tranformamos as variáveis para enfim preparar os dados para a modelagem.

```python
# Removendo colunas desnecessárias
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
df.head()

# Codificando variáveis categóricas
df = pd.get_dummies(df, columns=['Geography', 'Gender', 'Card Type'], drop_first=True)
df.head()

# Normalizando variáveis numéricas
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']] = scaler.fit_transform(df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']])
df.head()
```

### 2.5 Construindo o Modelo SKLEARN

Utilizamos o train_test_split para realizar a predição de churn no DF.

```python
# Dividindo os dados em conjunto de treinamento e teste
from sklearn.model_selection import train_test_split
X = df.drop('Exited', axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape

# Treinando um modelo de Regressão Logística
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

```

### 2.6 Avaliação e Visualização dos Resultados

Por fim realizamos a avaliação e visualizaçao dos resultados para interpretar o nosso modelo.

```python

## Avaliação do Modelo (AUC-ROC, Matriz de Confusão e Relatório de Classificação)
from sklearn.model_selection import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier 

# Cálculo do AUC-ROC
auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f'AUC-ROC: {auc_roc:.2f}')

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
print('Matriz de Confusão:')
print(cm)

# Relatório de Classificação
report = classification_report(y_test, y_pred)
print('Relatório de Classificação:')
print(report)

# Cálculo das métricas detalhadas
precision_0 = precision_score(y_test, y_pred, pos_label=0)
recall_0 = recall_score(y_test, y_pred, pos_label=0)
f1_0 = f1_score(y_test, y_pred, pos_label=0)
support_0 = cm[0, 0] + cm[0, 1]  # Total da classe 0

precision_1 = precision_score(y_test, y_pred, pos_label=1)
recall_1 = recall_score(y_test, y_pred, pos_label=1)
f1_1 = f1_score(y_test, y_pred, pos_label=1)
support_1 = cm[1, 0] + cm[1, 1]  # Total da classe 1

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.show()

# Realizar um cross val

model = RandomForestClassifier()

scores = cross_val_score(model, X_train, y_train, cv=5)  # cv é o número de folds

print(f'Scores de validação cruzada: {scores}')
print(f'Média das acurácias: {scores.mean():.2f}')

from sklearn.ensemble import RandomForestClassifier 

model = RandomForestClassifier()

scores = cross_val_score(model, X_train, y_train, cv=5)  # cv é o número de folds

print(f'Scores de validação cruzada: {scores}')
print(f'Média das acurácias: {scores.mean():.2f}')


# Obter a importância das features
if hasattr(model, 'coef_'):

    print("\nMétricas para a Classe 0:")
    print(f"Precisão: {precision_0:.2f}")
    print(f"Recall: {recall_0:.2f}")
    print(f"F1-Score: {f1_0:.2f}")
    print(f"Support: {support_0}")
  
    print("\nMétricas para a Classe 1:")
    print(f"Precisão: {precision_1:.2f}")
    print(f"Recall: {recall_1:.2f}")
    print(f"F1-Score: {f1_1:.2f}")
    print(f"Support: {support_1}\n\n")
  
    # Para a Regressão Logística, usamos coef_ ao invés de feature_importances_
    importancia_features = model.coef_[0]  # Acesso aos coeficientes do modelo

    # Obter os nomes das colunas
    colunas_totais = X.columns.tolist()

    # Visualizar a importância
    plt.figure(figsize=(10, 6))
    plt.barh(colunas_totais, importancia_features)
    plt.xlabel('Importância (Coeficientes)')
    plt.ylabel('Variáveis')
    plt.title('Importância das Variáveis no Modelo de Regressão Logística')
    plt.show()
else:
    print("O modelo não possui o atributo 'coef_'.")


"""
# Da para rodar um cross validation para visualizar mas vai da erro no cod kk
from sklearn.ensemble import RandomForestClassifier 

model = RandomForestClassifier()

scores = cross_val_score(model, X_train, y_train, cv=5)  # cv é o número de folds

print(f'Scores de validação cruzada: {scores}')
print(f'Média das acurácias: {scores.mean():.2f}')
"""

```

---

## 3. Projeto: Predição de Churn de Clientes Bancários

### 3.1 Coleta e Carregamento dos Dados

Carregamos um dataset de churn de múltiplos bancos para prever a saída de clientes.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Carregando o dataset de churn bancário
df_bancos = pd.read_csv('Bank_Customer_Churn_Prediction.csv')

# Exibindo as primeiras linhas do dataset
df.head()
```

### 3.2 Análise do DF

Verificamos as principais informações contidas no DF

```python
# Exibir as 5 primeiras linhas do dataset
print("Primeiras 5 linhas do dataset:")
print(df.head())

# Informações gerais sobre o dataset
print("\nInformações gerais do dataset:")
print(df.info())

# Resumo estatístico das variáveis numéricas
print("\nResumo estatístico das variáveis numéricas:")
print(df.describe())

# Verificar se há valores ausentes
print("\nValores ausentes por coluna:")
print(df.isnull().sum())

# Distribuição das classes no target (por exemplo, churn)
print("\nDistribuição das classes no target:")
print(df['Exited'].value_counts())

```

### 3.3 Análise Exploratória dos Dados (EDA)

Verificamos as características dos clientes e o comportamento relacionado ao churn.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Visualização da distribuição de churn
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Exited', data=df)
plt.title('Distribuição de Churn')
ax.set_xticklabels(['Não', 'Sim'])
ax.set_xlabel('')
ax.set_ylabel('') 

plt.show()


# Visualização das correlações entre as variáveis
plt.figure(figsize=(12, 8))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()  # Seleciona apenas colunas numéricas
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlação')
plt.show()

# Histogramas das variáveis numéricas
df.hist(bins=30, figsize=(15, 10), color='blue', edgecolor='black')
plt.suptitle('Distribuição das Variáveis Numéricas', fontsize=16)
plt.show()

# Boxplots para detectar outliers nas variáveis numéricas
plt.figure(figsize=(15, 8))
sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']))
plt.title('Boxplots das Variáveis Numéricas para Detecção de Outliers')
plt.xticks(rotation=90)
plt.show()

# Análise da relação entre churn e outras variáveis categóricas
categorical_columns = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

plt.figure(figsize=(14, 10))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=col, hue='Exited', data=df)
    plt.title(f'Distribuição de Churn por {col}')
    plt.legend(title='Churn', loc='upper right', labels=['Não', 'Sim'])
plt.tight_layout()
plt.show()

```

### 3.4 Pré-processamento dos Dados

Limpamos e Tranformamos as variáveis para enfim preparar os dados para a modelagem.

```python
# Removendo colunas desnecessárias
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
df.head()

# Codificando variáveis categóricas
df = pd.get_dummies(df, columns=['Geography', 'Gender', 'Card Type'], drop_first=True)
df.head()

# Normalizando variáveis numéricas
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']] = scaler.fit_transform(df[['CreditScore', 'Age', 'Balance', 'EstimatedSalary']])
df.head()
```

### 3.5 Construindo o Modelo SKLEARN

Utilizamos o train_test_split para realizar a predição de churn no DF.

```python
# Dividindo os dados em conjunto de treinamento e teste
from sklearn.model_selection import train_test_split
X = df.drop('Exited', axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape

# Treinando um modelo de Regressão Logística
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

```

### 3.6 Avaliação e Visualização dos Resultados

Por fim realizamos a avaliação e visualizaçao dos resultados para interpretar o nosso modelo.

```python

## Avaliação do Modelo (AUC-ROC, Matriz de Confusão e Relatório de Classificação)
from sklearn.model_selection import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier 

# Cálculo do AUC-ROC
auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f'AUC-ROC: {auc_roc:.2f}')

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)
print('Matriz de Confusão:')
print(cm)

# Relatório de Classificação
report = classification_report(y_test, y_pred)
print('Relatório de Classificação:')
print(report)

# Cálculo das métricas detalhadas
precision_0 = precision_score(y_test, y_pred, pos_label=0)
recall_0 = recall_score(y_test, y_pred, pos_label=0)
f1_0 = f1_score(y_test, y_pred, pos_label=0)
support_0 = cm[0, 0] + cm[0, 1]  # Total da classe 0

precision_1 = precision_score(y_test, y_pred, pos_label=1)
recall_1 = recall_score(y_test, y_pred, pos_label=1)
f1_1 = f1_score(y_test, y_pred, pos_label=1)
support_1 = cm[1, 0] + cm[1, 1]  # Total da classe 1

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.show()

# Realizar um cross val

model = RandomForestClassifier()

scores = cross_val_score(model, X_train, y_train, cv=5)  # cv é o número de folds

print(f'Scores de validação cruzada: {scores}')
print(f'Média das acurácias: {scores.mean():.2f}')

from sklearn.ensemble import RandomForestClassifier 

model = RandomForestClassifier()

scores = cross_val_score(model, X_train, y_train, cv=5)  # cv é o número de folds

print(f'Scores de validação cruzada: {scores}')
print(f'Média das acurácias: {scores.mean():.2f}')


# Obter a importância das features
if hasattr(model, 'coef_'):

    print("\nMétricas para a Classe 0:")
    print(f"Precisão: {precision_0:.2f}")
    print(f"Recall: {recall_0:.2f}")
    print(f"F1-Score: {f1_0:.2f}")
    print(f"Support: {support_0}")
  
    print("\nMétricas para a Classe 1:")
    print(f"Precisão: {precision_1:.2f}")
    print(f"Recall: {recall_1:.2f}")
    print(f"F1-Score: {f1_1:.2f}")
    print(f"Support: {support_1}\n\n")
  
    # Para a Regressão Logística, usamos coef_ ao invés de feature_importances_
    importancia_features = model.coef_[0]  # Acesso aos coeficientes do modelo

    # Obter os nomes das colunas
    colunas_totais = X.columns.tolist()

    # Visualizar a importância
    plt.figure(figsize=(10, 6))
    plt.barh(colunas_totais, importancia_features)
    plt.xlabel('Importância (Coeficientes)')
    plt.ylabel('Variáveis')
    plt.title('Importância das Variáveis no Modelo de Regressão Logística')
    plt.show()
else:
    print("O modelo não possui o atributo 'coef_'.")


"""
# Da para rodar um cross validation para visualizar mas vai da erro no cod kk
from sklearn.ensemble import RandomForestClassifier 

model = RandomForestClassifier()

scores = cross_val_score(model, X_train, y_train, cv=5)  # cv é o número de folds

print(f'Scores de validação cruzada: {scores}')
print(f'Média das acurácias: {scores.mean():.2f}')
"""

```

---

## Link para download dos csvs:

    - https://www.kaggle.com/datasets/mansigaikwad/nestle-india-historical-stock-price-data

    - https://www.kaggle.com/datasets/yorkyong/santander-churn-dataset

    - https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn

## Conclusão

Este arquivo apresenta três projetos preditivos utilizando dados da Nestlé, Santander e diversos bancos. Cada projeto foca em um problema de negócios específico: predição de preços de ações e churn de clientes. As técnicas incluem Regressão Linear e Random Forest, e o pipeline inclui EDA, pré-processamento, modelagem e avaliação dos resultados.
