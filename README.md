# Titanic Survival Predictor (Streamlit)

Logistic regression model deployed with Streamlit.

## Run locally
    pip install -r requirements.txt
    python train_model.py        # creates titanic_model.pkl (already included)
    streamlit run app.py

## Deploy on Streamlit Community Cloud
1. Put these files in a GitHub repository: app.py, titanic_model.pkl, requirements.txt (and train_model.py, Titanic_train.csv).
2. Go to https://share.streamlit.io, sign in with GitHub and click "Create app".
3. Choose the repository, branch and `app.py` as the main file, then click Deploy.
