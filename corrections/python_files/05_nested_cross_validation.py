# %% [markdown]
#
# # Hyperparameter tuning with nested cross-validation
#
# In this notebook, we will illustrate how to perform hyperparameter tuning
# using nested cross-validation.
#
# We start from the `tabular_pipeline` introduced in the heterogeneous-data notebook.

# %%
import pandas as pd

df = pd.read_csv("../datasets/penguins.csv")
df

# %%
feature_names = [
    "Island",
    "Culmen Depth (mm)",
    "Flipper Length (mm)",
    "Body Mass (g)",
    "Sex",
]
target_name = "Species"
X = df[feature_names]
y = df[target_name]

# %% [markdown]
#
# One penguin has `Sex` recorded as `.`. That token is not a sex, so we replace it with
# a missing value. A cross-validation fold can otherwise meet it only in the test rows,
# and the encoder warns about an unknown category.

# %%
X["Sex"] = X["Sex"].replace(".", pd.NA)
categorical_columns = X.select_dtypes(include=["object", "str"]).columns
X[categorical_columns] = X[categorical_columns].astype("category")

# %%
from sklearn.linear_model import LogisticRegression
from skrub import tabular_pipeline

logistic_regression = tabular_pipeline(LogisticRegression())
logistic_regression

# %% [markdown]
#
# In the past, we saw how to evaluate this model with cross-validation.

# %%
import pandas as pd
from sklearn.model_selection import cross_validate

cv_results = cross_validate(logistic_regression, X, y, cv=5, return_train_score=True)
cv_results = pd.DataFrame(cv_results)
cv_results[["train_score", "test_score"]]

# %% [markdown]
#
# However, this machine learning pipeline has some hyperparameters that are set to
# default values and not necessarily optimal for our problem.

# %%
logistic_regression.get_params()

# %% [markdown]
#
# We can tune these hyperparameters with
# `RandomizedSearchCV`.

# %%
from scipy.stats import loguniform
from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    "simpleimputer__strategy": ["mean", "median", "most_frequent"],
    "logisticregression__C": loguniform(1e-3, 1e3),
}
tuned_model = RandomizedSearchCV(
    logistic_regression,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    random_state=0,
)
tuned_model

# %%
# set `return_estimator=True` to access the best model found during the search
cv_results = cross_validate(tuned_model, X, y, cv=5, return_estimator=True)
cv_results = pd.DataFrame(cv_results)
cv_results

# %% [markdown]
#
# We can check the best estimator that has been found for each fold.

# %%
for estimator in cv_results["estimator"]:
    print(estimator.best_params_)

# %% [markdown]
#
# The search above uses the default score, accuracy. For a model that predicts
# probabilities, the log loss is a proper scoring rule, so it matches those
# probabilities. We pass `neg_log_loss` to `RandomizedSearchCV`. Each inner split
# then selects the hyperparameters with the smallest log loss.
#
# Scikit-learn scores are higher-is-better, so the log loss is passed with a `neg_`
# prefix. The values stored by `cross_validate` are negative: a value closer to zero
# is a smaller log loss.

# %%
tuned_model = RandomizedSearchCV(
    logistic_regression,
    param_distributions=param_distributions,
    n_iter=10,
    cv=5,
    scoring="neg_log_loss",
    random_state=0,
)
cv_results = cross_validate(
    tuned_model, X, y, cv=5, scoring="neg_log_loss", return_estimator=True
)
cv_results = pd.DataFrame(cv_results)
cv_results

# %%
for estimator in cv_results["estimator"]:
    print(estimator.best_params_)

# %%
