import gradio as gr
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

# ── Constantes ──
NEIGHBORHOODS = [
    'Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr',
    'Crawfor', 'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel',
    'NAmes', 'NPkVill', 'NWAmes', 'NoRidge', 'NridgHt', 'OldTown',
    'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker'
]

HOUSE_STYLES = ['1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl']

NUM_FEATURES_REG = [
    'GrLivArea', 'TotalBsmtSF', 'LotArea', 'BedroomAbvGr', 'FullBath',
    'TotRmsAbvGrd', 'OverallQual', 'OverallCond', 'YearBuilt',
    'YearRemodAdd', 'GarageCars', 'GarageArea', 'PoolArea', 'Fireplaces'
]

NUM_FEATURES_CLF = ['GrLivArea', 'TotRmsAbvGrd', 'OverallQual', 'YearBuilt', 'GarageCars']

COLS_REG = model_reg.feature_names_in_.tolist()
COLS_CLF = model_clf.feature_names_in_.tolist()

BLDG_TYPE_DESC = {
    '1Fam':   '🏠 Maison individuelle (1Fam)',
    '2fmCon': '🏘️ Maison 2 logements (2fmCon)',
    'Duplex': '🏢 Duplex',
    'TwnhsE': '🏙️ Maison de ville - fin (TwnhsE)',
    'Twnhs':  '🏙️ Maison de ville - intérieure (Twnhs)',
}

# ── Fonctions de prédiction ──
def predict_price(GrLivArea, TotalBsmtSF, LotArea, BedroomAbvGr, FullBath,
                  TotRmsAbvGrd, OverallQual, OverallCond, YearBuilt,
                  YearRemodAdd, GarageCars, GarageArea, PoolArea, Fireplaces, Neighborhood):
    try:
        num_values = [GrLivArea, TotalBsmtSF, LotArea, BedroomAbvGr, FullBath,
                      TotRmsAbvGrd, OverallQual, OverallCond, YearBuilt,
                      YearRemodAdd, GarageCars, GarageArea, PoolArea, Fireplaces]

        num_df = pd.DataFrame([num_values], columns=NUM_FEATURES_REG)
        num_scaled = scaler_reg.transform(num_df)
        num_scaled_df = pd.DataFrame(num_scaled, columns=NUM_FEATURES_REG)

        cat_df = pd.DataFrame([[Neighborhood]], columns=['Neighborhood'])
        cat_encoded = encoder_reg.transform(cat_df)
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder_reg.get_feature_names_out(['Neighborhood']))

        X = pd.concat([num_scaled_df.reset_index(drop=True),
                       cat_encoded_df.reset_index(drop=True)], axis=1)
        X = X.reindex(columns=COLS_REG, fill_value=0)

        pred = model_reg.predict(X)[0]
        return f"💰 Prix estimé : **${pred:,.0f}**"
    except Exception as e:
        return f"❌ Erreur : {str(e)}"


def predict_type(GrLivArea, TotRmsAbvGrd, OverallQual, YearBuilt,
                 GarageCars, Neighborhood, HouseStyle):
    try:
        num_values = [GrLivArea, TotRmsAbvGrd, OverallQual, YearBuilt, GarageCars]

        num_df = pd.DataFrame([num_values], columns=NUM_FEATURES_CLF)
        num_scaled = scaler_clf.transform(num_df)
        num_scaled_df = pd.DataFrame(num_scaled, columns=NUM_FEATURES_CLF)

        cat_df = pd.DataFrame([[Neighborhood, HouseStyle]], columns=['Neighborhood', 'HouseStyle'])
        cat_encoded = encoder_clf.transform(cat_df)
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder_clf.get_feature_names_out(['Neighborhood', 'HouseStyle']))

        X = pd.concat([num_scaled_df.reset_index(drop=True),
                       cat_encoded_df.reset_index(drop=True)], axis=1)
        X = X.reindex(columns=COLS_CLF, fill_value=0)

        pred_encoded = model_clf.predict(X)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        desc = BLDG_TYPE_DESC.get(pred_label, pred_label)
        return f"🏗️ Type identifié : **{desc}**"
    except Exception as e:
        return f"❌ Erreur : {str(e)}"


# ── Interface Gradio ──
css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

body, .gradio-container {
    font-family: 'Poppins', sans-serif !important;
    background: linear-gradient(135deg, #f5f0ff 0%, #ede7f6 100%) !important;
}

.tab-nav button {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #6a1b9a !important;
}

.tab-nav button.selected {
    border-bottom: 3px solid #7c3aed !important;
    color: #7c3aed !important;
}

button.primary {
    background: linear-gradient(90deg, #7c3aed, #9d4edd) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    color: white !important;
}

.block {
    border-radius: 16px !important;
    border: 1px solid #e9d5ff !important;
    background: white !important;
    box-shadow: 0 2px 12px rgba(124, 58, 237, 0.08) !important;
}

label span {
    color: #6a1b9a !important;
    font-weight: 600 !important;
}
"""

with gr.Blocks(
    css=css,
    title="Immo Predictor",
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="violet",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Poppins")
    )
) as demo:

    gr.Markdown("""
    # 🏠 Immo Predictor
    ### Plateforme intelligente de valorisation et diagnostic immobilier
    ---
    """)

    with gr.Tabs():

        # ── Onglet Régression ──
        with gr.TabItem("💰 Estimer le Prix"):
            gr.Markdown("### Renseignez les caractéristiques du bien pour estimer son prix de vente.")

            with gr.Row():
                with gr.Column():
                    gr_liv   = gr.Number(label="Surface habitable - GrLivArea (pi²)", value=1500)
                    bsmt     = gr.Number(label="Surface sous-sol - TotalBsmtSF (pi²)", value=800)
                    lot      = gr.Number(label="Surface terrain - LotArea (pi²)", value=8000)
                    bedroom  = gr.Slider(0, 10, value=3, step=1, label="Chambres - BedroomAbvGr")
                    fullbath = gr.Slider(0, 5, value=2, step=1, label="Salles de bain - FullBath")
                    totrms   = gr.Slider(1, 20, value=7, step=1, label="Nb pièces total - TotRmsAbvGrd")
                    neighborhood_r = gr.Dropdown(NEIGHBORHOODS, value='NAmes', label="Quartier - Neighborhood")

                with gr.Column():
                    qual     = gr.Slider(1, 10, value=7, step=1, label="Qualité générale - OverallQual /10")
                    cond     = gr.Slider(1, 10, value=5, step=1, label="Condition générale - OverallCond /10")
                    year_b   = gr.Number(label="Année de construction - YearBuilt", value=2003)
                    year_r   = gr.Number(label="Année rénovation - YearRemodAdd", value=2005)
                    garcar   = gr.Slider(0, 5, value=2, step=1, label="Places garage - GarageCars")
                    gararea  = gr.Number(label="Surface garage - GarageArea (pi²)", value=500)
                    pool     = gr.Number(label="Surface piscine - PoolArea (pi²)", value=0)
                    firepl   = gr.Slider(0, 5, value=1, step=1, label="Nb cheminées - Fireplaces")

            btn_reg = gr.Button("💰 Estimer le prix", variant="primary", size="lg")
            out_reg = gr.Markdown()

            btn_reg.click(
                fn=predict_price,
                inputs=[gr_liv, bsmt, lot, bedroom, fullbath, totrms,
                        qual, cond, year_b, year_r, garcar, gararea, pool, firepl,
                        neighborhood_r],
                outputs=out_reg
            )

        # ── Onglet Classification ──
        with gr.TabItem("🏗️ Classifier le Type"):
            gr.Markdown("### Renseignez les caractéristiques du bien pour identifier son type de bâtiment.")

            with gr.Row():
                with gr.Column():
                    gr_liv_c  = gr.Number(label="Surface habitable - GrLivArea (pi²)", value=1500)
                    totrms_c  = gr.Slider(1, 20, value=7, step=1, label="Nb pièces total - TotRmsAbvGrd")
                    qual_c    = gr.Slider(1, 10, value=7, step=1, label="Qualité générale - OverallQual /10")

                with gr.Column():
                    year_c    = gr.Number(label="Année de construction - YearBuilt", value=2003)
                    garcar_c  = gr.Slider(0, 5, value=2, step=1, label="Places garage - GarageCars")
                    neighborhood_c = gr.Dropdown(NEIGHBORHOODS, value='NAmes', label="Quartier - Neighborhood")
                    style_c   = gr.Dropdown(HOUSE_STYLES, value='1Story', label="Style de maison - HouseStyle")

            btn_clf = gr.Button("🏗️ Classifier le type", variant="primary", size="lg")
            out_clf = gr.Markdown()

            gr.Markdown("""
            **Types possibles :** 🏠 1Fam (Maison individuelle) · 🏘️ 2fmCon · 🏢 Duplex · 🏙️ TwnhsE · 🏙️ Twnhs
            """)

            btn_clf.click(
                fn=predict_type,
                inputs=[gr_liv_c, totrms_c, qual_c, year_c, garcar_c, neighborhood_c, style_c],
                outputs=out_clf
            )

    gr.Markdown("---\n*Immo Predictor — Projet Examen Machine Learning 2026*")

demo.launch()
