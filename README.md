## 🏠 Immo Predictor

> Plateforme intelligente de valorisation et diagnostic immobilier basée sur le Machine Learning.
> 
## 🔗 Liens

| | Lien |
|---|---|
| 🖥️ **Interface Gradio** | [huggingface.co/spaces/NdeyeDibe/immo-predictor](https://huggingface.co/spaces/NdeyeDibe/immo-predictor) |
| 📡 **API FastAPI** | [NdeyeDibe-immo-predictor-api.hf.space/docs](https://NdeyeDibe-immo-predictor-api.hf.space/docs) |

---

## 📌 Description

Immo Predictor est une application de Machine Learning développée dans le cadre d'un projet d'examen. Elle permet d'effectuer deux tâches sur des données immobilières issues du dataset **House Prices (Kaggle)** :

- 💰 **Régression** : Estimer le prix de vente d'un bien immobilier
- 🏗️ **Classification** : Identifier le type de bâtiment

---

## 📊 Performances des modèles

### Régression (SalePrice)

| Modèle | MAE | RMSE | R² |
|--------|-----|------|----|
| Decision Tree | 21 946 $ | 31 419 $ | 0.788 |
| **Random Forest** ✅ | **14 706 $** | **20 811 $** | **0.907** |

### Classification (BldgType)

| Modèle | Accuracy | F1-Score |
|--------|----------|----------|
| SVM | 0.890 | 0.877 |
| **Random Forest** ✅ | **0.916** | **0.909** |

---

## 🛠️ Technologies

- **Langage** : Python 3.10
- **Prétraitement** : Pandas, StandardScaler, OneHotEncoder
- **ML** : Scikit-learn (Random Forest, SVM, Decision Tree)
- **Sérialisation** : Pickle
- **Interface** : Gradio
- **API** : FastAPI + Uvicorn
- **Déploiement** : Hugging Face Spaces + Docker

---

## 📁 Structure du projet

```
immo-predictor/
├── app.py              # Interface Gradio
├── main.py             # API FastAPI
├── Dockerfile          # Configuration Docker
├── requirements.txt    # Dépendances
├── notebook.ipynb      # Notebook d'entraînement
└── README.md
```

> ⚠️ Les fichiers `.pkl` (modèles entraînés) ne sont pas inclus dans ce repo en raison de leur taille. Ils sont hébergés directement sur Hugging Face Spaces.

---

## 📡 Utilisation de l'API

### Estimer un prix
```bash
POST https://NdeyeDibe-immo-predictor-api.hf.space/predict_price

{
  "GrLivArea": 1500,
  "TotalBsmtSF": 800,
  "LotArea": 8000,
  "OverallQual": 7,
  "OverallCond": 5,
  "YearBuilt": 2003,
  "YearRemodAdd": 2005,
  "BedroomAbvGr": 3,
  "FullBath": 2,
  "TotRmsAbvGrd": 7,
  "GarageCars": 2,
  "GarageArea": 500,
  "PoolArea": 0,
  "Fireplaces": 1,
  "Neighborhood": "NAmes"
}
```

### Classifier un type
```bash
POST https://NdeyeDibe-immo-predictor-api.hf.space/predict_type

{
  "GrLivArea": 1500,
  "TotRmsAbvGrd": 7,
  "OverallQual": 7,
  "YearBuilt": 2003,
  "GarageCars": 2,
  "Neighborhood": "NAmes",
  "HouseStyle": "1Story"
}
```

---

## 👩‍💻 Auteur

**Ndeye Dibe Faye** — Projet Examen Machine Learning 2026
