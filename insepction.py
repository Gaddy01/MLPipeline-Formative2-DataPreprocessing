import joblib

model = joblib.load("models/product_recommendation_model.pkl")

print(model.n_features_in_)

print(model.feature_names_in_)
