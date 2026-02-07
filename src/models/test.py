# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

df = pd.read_parquet('../../data/processed.parquet')
# %%
df = df.drop(columns=['smiles'])
# %%
y = df['pic50']
X = df.drop(columns='pic50')

print(y)
print(X)
# %%

# %%

rf_regularizado = RandomForestRegressor(
    n_estimators=500,
    max_depth=5,              
    min_samples_leaf=15,      
    max_features=0.2,         
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)

# Cross-validation para avaliação mais robusta
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(rf_regularizado, X, y, cv=cv, scoring='r2')
print(f"CV R²: {scores.mean():.3f} ± {scores.std():.3f}")

# %%

from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

modelos = {
    'Ridge': Ridge(alpha=100),
    'Ridge_forte': Ridge(alpha=1000),
    'Lasso': Lasso(alpha=1.0),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
}

for nome, modelo in modelos.items():
    scores = cross_val_score(modelo, X, y, cv=cv, scoring='r2')
    print(f"{nome}: R² = {scores.mean():.3f} ± {scores.std():.3f}")


# %%

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression

# Remover features com variância muito baixa
var_selector = VarianceThreshold(threshold=0.01)
X_var = var_selector.fit_transform(X)
print(f"Features após VarianceThreshold: {X_var.shape[1]}")

# Selecionar as K melhores
k_best = SelectKBest(f_regression, k=200)  # testar 100, 200, 500
X_selected = k_best.fit_transform(X, y)

# Re-treinar com features selecionadas
scores = cross_val_score(rf_regularizado, X_selected, y, cv=cv, scoring='r2')
print(f"RF com {X_selected.shape[1]} features: R² = {scores.mean():.3f}")


# %%

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [10, 20, 30, 50],
    'max_features': [0.1, 0.2, 0.3, 'sqrt'],
    'n_estimators': [200, 500]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)
print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor R² CV: {grid_search.best_score_:.3f}")