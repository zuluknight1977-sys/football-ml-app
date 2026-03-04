import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from xgboost import XGBClassifier
import random

st.set_page_config(page_title="Football Betting ML SaaS", layout="wide")

st.title("⚽ Football ML Betting System")

uploaded_file = st.file_uploader("Upload Training Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    required_columns = [
        "League",
        "HomeCornersAvg",
        "AwayCornersAvg",
        "HomeYellowAvg",
        "AwayYellowAvg",
        "CornersOver9_5",
        "CardsOver4_5"
    ]

    if not all(col in df.columns for col in required_columns):
        st.error("Dataset missing required columns.")
        st.stop()

    # ----------------------------
    # 🔥 LEAGUE FILTER
    # ----------------------------
    selected_league = st.selectbox(
        "Select League",
        options=df["League"].unique()
    )

    df = df[df["League"] == selected_league]

    # ----------------------------
    # 🔥 AUTOMATIC CLASS BALANCING
    # ----------------------------
    def balance_data(data, target):
        majority = data[data[target] == data[target].value_counts().idxmax()]
        minority = data[data[target] != data[target].value_counts().idxmax()]

        minority_upsampled = resample(
            minority,
            replace=True,
            n_samples=len(majority),
            random_state=42
        )

        return pd.concat([majority, minority_upsampled])

    df_corners = balance_data(df, "CornersOver9_5")
    df_cards = balance_data(df, "CardsOver4_5")

    # ----------------------------
    # FEATURES
    # ----------------------------
    features = [
        "HomeCornersAvg",
        "AwayCornersAvg",
        "HomeYellowAvg",
        "AwayYellowAvg"
    ]

    X_c = df_corners[features]
    y_c = df_corners["CornersOver9_5"]

    X_y = df_cards[features]
    y_y = df_cards["CardsOver4_5"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_c, y_c, test_size=0.2, random_state=42
    )

    X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(
        X_y, y_y, test_size=0.2, random_state=42
    )

    # ----------------------------
    # 🔥 XGBOOST MODEL
    # ----------------------------
    model_corners = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model_cards = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    model_corners.fit(X_train_c, y_train_c)
    model_cards.fit(X_train_y, y_train_y)

    acc_c = accuracy_score(y_test_c, model_corners.predict(X_test_c))
    acc_y = accuracy_score(y_test_y, model_cards.predict(X_test_y))

    st.success(f"Corners Model Accuracy: {round(acc_c*100,2)}%")
    st.success(f"Cards Model Accuracy: {round(acc_y*100,2)}%")

    # ----------------------------
    # 🔥 50 MATCH TICKET GENERATOR
    # ----------------------------
    st.subheader("🎟 Generate 50-Match Ticket")

    if st.button("Generate Ticket"):

        ticket = []

        for _ in range(50):

            sample = df.sample(1)

            input_data = sample[features]

            prob_c = model_corners.predict_proba(input_data)[0][1]
            prob_y = model_cards.predict_proba(input_data)[0][1]

            ticket.append({
                "HomeCornersAvg": sample["HomeCornersAvg"].values[0],
                "AwayCornersAvg": sample["AwayCornersAvg"].values[0],
                "Corners Prediction": "Over 9.5" if prob_c > 0.5 else "Under 9.5",
                "Corners Confidence %": round(prob_c * 100, 2),
                "Cards Prediction": "Over 4.5" if prob_y > 0.5 else "Under 4.5",
                "Cards Confidence %": round(prob_y * 100, 2),
            })

        ticket_df = pd.DataFrame(ticket)

        ticket_df = ticket_df.sort_values(
            by="Corners Confidence %",
            ascending=False
        )

        st.dataframe(ticket_df)

        csv = ticket_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "Download Ticket CSV",
            csv,
            "50_match_ticket.csv",
            "text/csv"
        )
