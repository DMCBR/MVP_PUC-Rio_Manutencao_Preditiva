# MVP — Manutenção Preditiva
Análise exploratória e modelagem preditiva de falhas industriais com o dataset **AI4I 2020 Predictive Maintenance** (UCI Machine Learning Repository). Sprint Análise de Dados e Boas Práticas do curso de Ciência de Dados e Analytics da PUC-Rio.

---

## Descrição

Este projeto analisa as condições operacionais de equipamentos industriais para prever a ocorrência de falhas. Foram desenvolvidos três modelos independentes — um generalista (Machine failure) e dois especializados por modo de falha (HDF e OSF) — com o objetivo de comparar o desempenho entre abordagens.

O trabalho cobre desde a análise exploratória até o treinamento e avaliação dos modelos, seguindo boas práticas de engenharia de software para ciência de dados.

---

## Dataset

| Propriedade     | Valor                                      |
|-----------------|--------------------------------------------|
| Fonte           | UCI Machine Learning Repository            |
| ID              | 601                                        |
| Instâncias      | 10.000                                     |
| Atributos       | 14                                         |
| Tipo de problema| Classificação binária (supervisionado)     |

**Atributos principais:**

- `Type` — tipo do produto (L, M, H)
- `Air temperature (K)` — temperatura do ar
- `Process temperature (K)` — temperatura do processo
- `Rotational speed (rpm)` — velocidade de rotação
- `Torque (Nm)` — torque
- `Tool wear (min)` — desgaste da ferramenta

**Targets analisados:**

| Dataset      | Target          | Falhas | Proporção |
|--------------|-----------------|--------|-----------|
| ds_machine   | Machine failure | 339    | 3,39%     |
| ds_hdf       | HDF             | 115    | 1,15%     |
| ds_osf       | OSF             | 98     | 0,98%     |

---

## Hipóteses

1. O dataset apresenta desbalanceamento entre as classes.
2. Variáveis operacionais influenciam a ocorrência de falhas.
3. Certas combinações de variáveis estão mais associadas à falha.
4. É possível prever falhas a partir das variáveis operacionais.
5. Modelos especializados por modo de falha superam o modelo generalista.

---

## Estrutura do Projeto

```
MVP_Manutencao_Preditiva/
│
├── MVP_Manutencao_Preditiva.ipynb   # Notebook principal
├── README.md                        # Este arquivo
```

---

## Pipeline

```
Carga de dados (UCI)
        ↓
Análise Exploratória (EDA)
        ↓
Feature Engineering (Temperature delta — HDF)
        ↓
Split estratificado 70/30
        ↓
Padronização (StandardScaler + OneHotEncoder)
        ↓
Balanceamento (SMOTE — apenas treino)
        ↓
Treinamento (RandomForestClassifier)
        ↓
Avaliação (Accuracy, Precision, Recall, F1, Matriz de Confusão)
```

---

## Arquitetura do Código

O código tem a seguinte separação de responsabilidades por classe:

| Classe                    | Responsabilidade                                      |
|---------------------------|-------------------------------------------------------|
| `TargetValidator`         | Valida se o target pertence aos modos permitidos      |
| `DataLoader`              | Carrega o dataset do repositório UCI                  |
| `ColumnDropper`           | Remove colunas desnecessárias do DataFrame            |
| `PredictiveMaintenanceDS` | Encapsula o dataset preparado para um target          |
| `FeatureEngineer`         | Cria novas features (Temperature delta)               |
| `ExploratoryAnalysis`     | Executa toda a análise exploratória                   |
| `Preprocessor`            | Split, padronização e balanceamento                   |
| `ModelTrainer`            | Treinamento, predição e avaliação do modelo           |

---

## Resultados

| Target          | Precision | Recall | F1-Score |
|-----------------|-----------|--------|----------|
| Machine failure | 0,49      | 0,70   | 0,57     |
| HDF             | 0,81      | 0,76   | 0,79     |
| OSF             | 0,79      | 0,66   | 0,72     |

Métricas referentes à classe positiva (falha).

**Conclusão:** Os modelos especializados (HDF e OSF) superam o modelo generalista em F1-Score, confirmando a Hipótese 5. O modelo generalista aprende simultaneamente cinco padrões de falha distintos, o que dilui o sinal preditivo.

---

## Features Mais Importantes

**Machine failure:** Rotational speed (28%) · Torque (28%) · Tool wear (19%)

**HDF:** Rotational speed (42%) · Air temperature (28%) · Torque (15%)

**OSF:** Tool wear (32%) · Torque (29%) · Rotational speed (25%)

---

## Dependências

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
imbalanced-learn
ucimlrepo
```

Instalação:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn ucimlrepo
```

---

## Execução

O notebook foi desenvolvido e testado no **Google Colab**. O dataset é carregado diretamente do UCI via `ucimlrepo`, sem necessidade de download manual.

```python
# Instalação da dependência UCI
!pip install ucimlrepo

# Carga automática
df_raw = DataLoader.load_from_ucirepo()
```

---

## Referências

- AI4I 2020 Predictive Maintenance Dataset. UCI Machine Learning Repository, 2020. DOI: https://doi.org/10.24432/C5HS5C

- MATZKA, S. Explainable Artificial Intelligence for Predictive Maintenance Applications. In: International Conference on Artificial Intelligence for Industries (AI4I), 3., 2020. Proceedings [...]. IEEE, 2020. DOI: 10.1109/AI4I49448.2020.00023

- KALINOWSKI, M. et al. Engenharia de Software para Ciência de Dados. Casa do Código, 2023.

- VAN ROSSUM, G.; WARSAW, B. PEP 8 — Style Guide for Python Code. Python Software Foundation, 2001.

---

## Autor

Daniel Massari de Souza Coelho
Programa de Pós-Graduação em Data Science & Analytics — PUC-Rio
