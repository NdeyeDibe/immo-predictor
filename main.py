from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# ── Chargement des modèles ──
with open('model_reg.pkl', 'rb') as f:
    model_reg = pickle.load(f)
with open('model_clf.pkl', 'rb') as f:
    model_clf = pickle.load(f)
with open('scaler_reg.pkl', 'rb') as f:
    scaler_reg = pickle.load(f)
with open('scaler_clf.pkl', 'rb') as f:
    scaler_clf = pickle.load(f)
with open('encoder_reg.pkl', 'rb') as f:
    encoder_reg = pickle.load(f)
with open('encoder_clf.pkl', 'rb') as f:
    encoder_clf = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

NUM_FEATURES_REG = [
    'GrLivArea', 'TotalBsmtSF', 'LotArea', 'BedroomAbvGr', 'FullBath',
    'TotRmsAbvGrd', 'OverallQual', 'OverallCond', 'YearBuilt',
    'YearRemodAdd', 'GarageCars', 'GarageArea', 'PoolArea', 'Fireplaces'
]
NUM_FEATURES_CLF = ['GrLivArea', 'TotRmsAbvGrd', 'OverallQual', 'YearBuilt', 'GarageCars']
COLS_REG = model_reg.feature_names_in_.tolist()
COLS_CLF = model_clf.feature_names_in_.tolist()

class RegressionInput(BaseModel):
    GrLivArea: float
    TotalBsmtSF: float
    LotArea: float
    BedroomAbvGr: int
    FullBath: int
    TotRmsAbvGrd: int
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    GarageCars: int
    GarageArea: float
    PoolArea: float
    Fireplaces: int
    Neighborhood: str

class ClassificationInput(BaseModel):
    GrLivArea: float
    TotRmsAbvGrd: int
    OverallQual: int
    YearBuilt: int
    GarageCars: int
    Neighborhood: str
    HouseStyle: str

app = FastAPI(
    title="Immo Predictor API",
    description="API de valorisation et diagnostic immobilier — Projet Examen ML 2026",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "Bienvenue sur l'API Immo Predictor",
        "endpoints": {
            "POST /predict_price": "Estimer le prix d'un bien immobilier",
            "POST /predict_type": "Classifier le type d'un bâtiment"
        }
    }

@app.post("/predict_price")
def predict_price(data: RegressionInput):
    num_values = [
        data.GrLivArea, data.TotalBsmtSF, data.LotArea, data.BedroomAbvGr,
        data.FullBath, data.TotRmsAbvGrd, data.OverallQual, data.OverallCond,
        data.YearBuilt, data.YearRemodAdd, data.GarageCars, data.GarageArea,
        data.PoolArea, data.Fireplaces
    ]
    num_df = pd.DataFrame([num_values], columns=NUM_FEATURES_REG)
    num_scaled_df = pd.DataFrame(scaler_reg.transform(num_df), columns=NUM_FEATURES_REG)
    cat_df = pd.DataFrame([[data.Neighborhood]], columns=['Neighborhood'])
    cat_encoded_df = pd.DataFrame(
        encoder_reg.transform(cat_df),
        columns=encoder_reg.get_feature_names_out(['Neighborhood'])
    )
    X = pd.concat([num_scaled_df, cat_encoded_df], axis=1).reindex(columns=COLS_REG, fill_value=0)
    prix = model_reg.predict(X)[0]
    return {"prix_estime": round(float(prix), 2), "prix_formate": f"${prix:,.0f}"}

@app.post("/predict_type")
def predict_type(data: ClassificationInput):
    num_values = [data.GrLivArea, data.TotRmsAbvGrd, data.OverallQual, data.YearBuilt, data.GarageCars]
    num_df = pd.DataFrame([num_values], columns=NUM_FEATURES_CLF)
    num_scaled_df = pd.DataFrame(scaler_clf.transform(num_df), columns=NUM_FEATURES_CLF)
    cat_df = pd.DataFrame([[data.Neighborhood, data.HouseStyle]], columns=['Neighborhood', 'HouseStyle'])
    cat_encoded_df = pd.DataFrame(
        encoder_clf.transform(cat_df),
        columns=encoder_clf.get_feature_names_out(['Neighborhood', 'HouseStyle'])
    )
    X = pd.concat([num_scaled_df, cat_encoded_df], axis=1).reindex(columns=COLS_CLF, fill_value=0)
    pred_label = label_encoder.inverse_transform([model_clf.predict(X)[0]])[0]
    descriptions = {
        '1Fam': 'Maison individuelle', '2fmCon': 'Maison 2 logements',
        'Duplex': 'Duplex', 'TwnhsE': 'Maison de ville (fin)', 'Twnhs': 'Maison de ville (intérieure)'
    }
    return {"type_batiment": pred_label, "description": descriptions.get(pred_label, pred_label)}
