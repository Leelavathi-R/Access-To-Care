from sklearn.linear_model import LinearRegression
import pandas as pd

def train_model(X, y, subgroup):
    X["t"] = X["TIME_PERIOD"] - X["TIME_PERIOD"].min()
    model = LinearRegression()
    if subgroup == "50-64 years" or subgroup == "18-34 years": 
        X["t2"] = X["t"] ** 2
        model.fit(X[["t","t2"]], y)
    elif subgroup == "65 years and older":
        X["post_2022"] = (X["TIME_PERIOD"] >= 2022).astype(int)
        X["t_post"] = X["t"] * X["post_2022"]
        model.fit(X[["t", "t_post"]], y)
    return model

def predict_next_year(model, t_next, subgroup):
    if subgroup == "50-64 years" or subgroup == "18-34 years": 
        X_pred = pd.DataFrame({'t': [t_next], 't2': [t_next**2]})
        return model.predict(X_pred)[0]
    elif subgroup == "65 years and older":
        t_post = t_next * 1
        X_pred = pd.DataFrame({"t": [t_next], "t_post": [t_post]})
        return model.predict(X_pred)[0]