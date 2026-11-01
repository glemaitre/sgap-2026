# %% [markdown]
#
# # Evaluate a predictive model
#
# In this first notebook, we show how to evaluate a predictive model.
#
# Let's consider a simple regression problem by loading a subset of the penguins
# dataset. Let's check what data are available in this dataset.

# %% [markdown]
#
# ## Environment setup
#
# Run this cell in JupyterLite. A local installation already provides these packages.

# %%
# %pip install -q "scikit-learn~=1.9.1" "skrub~=0.11.0" "skore~=0.29.0" pandas matplotlib

# %%
import pandas as pd

df = pd.read_csv("../datasets/penguins_regression.csv")
df

# %% [markdown]
#
# In this dataset, we observe that we have two variables: the flipper length and the
# body mass of the penguins. The objective here is to create a predictive model allowing
# us to predict the body mass of a penguin based on its flipper length.
#
# First, we can have a look at the relationship between the flipper length and the body
# mass of the penguins.

# %%
ax = df.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
_ = ax.set_title("Penguin Flipper Length vs Body Mass")

# %% [markdown]
#
# Looking at this plot, we observe that there is a kind of linear relationship between
# the flipper length and the body mass of the penguins. We will start by fitting a
# linear regression model to this data.
#
# To do so, we will first prepare the data by creating the input data `X` and the target
# data `y`.

# %%
X = df[["Flipper Length (mm)"]]
y = df["Body Mass (g)"]

# %%
print(f"The dimensions of X are: {X.shape}")
print(f"The dimensions of y are: {y.shape}")

# %% [markdown]
#
# Here, the matrix `X` only contains a single feature. However, in the future, we might
# want to add several features that allow us to predict the target `y`. The target `y`
# is a one-dimensional array here, meaning that we only predict a single target. Note
# that in some cases, it is possible that the target to be predicted is a
# multi-dimensional array.
#
# Also, here we try to predict a continuous target. This is why we are in a regression
# setting. In other cases, we might want to predict a categorical target. This is called
# a classification problem.
#
# Let's start by fitting a scikit-learn model that is a simple linear regression model.
# With a single feature, the predicted body mass is a straight line:
#
# $$
# \text{predicted body mass} = \text{coef} \times \text{flipper length} + \text{intercept}
# $$

# %%
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(X, y)

# %% [markdown]
#
# In scikit-learn, the method {py:meth}`~sklearn.linear_model.LinearRegression.fit` is
# used to train a model. In this case, it allows us to find the best parameters of the
# linear regression model to fit the data. These parameters are stored in the attributes
# `coef_` and `intercept_` of the instance.

# %%
linear_regression.coef_, linear_regression.intercept_

# %% [markdown]
#
# Let's now use this model to predict the body mass of the penguins based on their
# flipper length. We create a synthetic dataset of potential flipper length values and
# predict the body mass of the penguins using our model.

# %%
import numpy as np

X_to_infer = pd.DataFrame({"Flipper Length (mm)": np.linspace(175, 230, 100)})
y_pred = linear_regression.predict(X_to_infer)

# %% [markdown]
#
# The method {py:meth}`~sklearn.linear_model.LinearRegression.predict` allows us to get
# the prediction of the model on new data. Now, we plot the obtained values.

# %%
ax = df.plot.scatter(x="Flipper Length (mm)", y="Body Mass (g)")
ax.set_title("Penguin Flipper Length vs Body Mass")
ax.plot(X_to_infer, y_pred, linewidth=3, color="tab:orange", label="Linear Regression")
_ = ax.legend()

# %% [markdown]
#
# Fitting chooses the coefficient and the intercept that minimize a loss. For
# {py:class}`~sklearn.linear_model.LinearRegression`, that loss is the residual sum of
# squares between the observed body mass and the prediction:
#
# $$
# \text{RSS}(\text{coef}, \text{intercept}) =
# \sum_{i=1}^{n} \left(y_i - (\text{coef} \times x_i + \text{intercept})\right)^2
# $$
#
# $x_i$ is the flipper length of penguin $i$ and $y_i$ is its body mass. The mean
# squared error is this sum divided by $n$. The factor $1/n$ does not depend on the
# parameters, so both losses are minimized by the same coefficient and intercept. The
# scikit-learn user guide presents this estimator as [ordinary least
# squares](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares).
# The prediction is an estimate of the conditional mean of the body mass.
#
# ```{note}
# The same loss can be minimized by gradient descent. The gradient is the pair of
# partial derivatives of the residual sum of squares with respect to the coefficient
# and the intercept:
#
# $$
# \frac{\partial \text{RSS}}{\partial \text{coef}} =
# -2 \sum_{i=1}^{n} \left(y_i - (\text{coef} \times x_i + \text{intercept})\right) x_i
# $$
#
# $$
# \frac{\partial \text{RSS}}{\partial \text{intercept}} =
# -2 \sum_{i=1}^{n} \left(y_i - (\text{coef} \times x_i + \text{intercept})\right)
# $$
#
# The term in parentheses is the residual of penguin $i$. Differentiating the squared
# residual brings a factor $2$, and the minus sign comes from the residual decreasing
# when the prediction increases. The coefficient derivative has an extra $x_i$ because
# the prediction depends on the coefficient through $\text{coef} \times x_i$.
#
# Each partial derivative is the rate at which the loss increases when that parameter
# increases. An optimizer starts from some parameters and repeatedly steps in the
# opposite direction, with $\eta$ the learning rate:
#
# $$
# \text{coef} \leftarrow
# \text{coef} - \eta \frac{\partial \text{RSS}}{\partial \text{coef}}
# \qquad
# \text{intercept} \leftarrow
# \text{intercept} - \eta \frac{\partial \text{RSS}}{\partial \text{intercept}}
# $$
#
# With a small enough learning rate, each step reduces the loss. This loss is a convex
# quadratic, so the updates then reach the ordinary least-squares solution.
# {py:class}`~sklearn.linear_model.LinearRegression` computes that solution directly.
# ```
#
# We can compute the mean squared error on the same dataset that we used to train the
# model.

# %%
from sklearn.metrics import mean_squared_error

error = mean_squared_error(y, linear_regression.predict(X))
print(f"The mean squared error is: {error:.2f}")

# %% [markdown]
#
# ### Exercise
#
# {py:class}`~sklearn.linear_model.LinearRegression` estimates the conditional mean. Another linear model,
# {py:class}`~sklearn.linear_model.QuantileRegressor`, estimates a conditional quantile. The documentation is in the
# user guide on
# [quantile regression](https://scikit-learn.org/stable/modules/linear_model.html#quantile-regression)
# and in the
# {py:class}`~sklearn.linear_model.QuantileRegressor` docstring.
#
# The model minimizes the pinball loss. With `quantile=0.5`, the target is the
# conditional median, and the loss matches the median absolute error. Set `alpha=0`:
# the default value adds an L1 penalty and would confound the comparison with
# {py:class}`~sklearn.linear_model.LinearRegression`.
#
# Fit this median model on the same data. Plot its predictions on the same figure as
# the linear regression. Compute the mean squared error and the median absolute error
# of both models.
#
# Which model wins on which metric? The model we choose depends on the quantity we
# want to estimate, and that quantity should match the objective of the problem.

# %%

# %% [markdown]
#
# Up to now, we have evaluated the models on the same dataset used to fit them. This
# does not tell us how well they generalize to new data.
#
# We set aside a testing set with {py:func}`~sklearn.model_selection.train_test_split`. The training set is used to
# create the model, and the testing set is used to evaluate it on unseen data.

# %%
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import median_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %% [markdown]
#
# We fit the linear model and the median model. The median model is a
# {py:class}`~sklearn.linear_model.QuantileRegressor` with `quantile=0.5` and `alpha=0`, as in the exercise. We compare
# them with the median absolute error on the testing set.

# %%
linear_regression.fit(X_train, y_train)
quantile_regression = QuantileRegressor(quantile=0.5, alpha=0)
quantile_regression.fit(X_train, y_train)
print(
    "Median absolute error of the linear model on the testing set: "
    f"{median_absolute_error(y_test, linear_regression.predict(X_test)):.2f}"
)
print(
    "Median absolute error of the quantile model on the testing set: "
    f"{median_absolute_error(y_test, quantile_regression.predict(X_test)):.2f}"
)

# %% [markdown]
#
# Each printed number is a single estimate of the generalization error. It has no
# variance. A lower value on this split is not evidence that the improvement is
# significant: we cannot tell whether the difference would show up again on another
# split.

# %% [markdown]
#
# ### Exercise
#
# Repeat the train-test evaluation of the linear model for several seeds, and plot the
# distribution of the test error.
#
# Use a `for` loop over `range`. At each seed, call {py:func}`~sklearn.model_selection.train_test_split` with
# `random_state=seed`, fit the linear model, and append the median absolute error on
# the test set to a list. Plot that list with {py:meth}`pandas.Series.plot.hist`.

# %%

# %% [markdown]
#
# Doing this repetition by hand is tedious. {py:func}`~sklearn.model_selection.cross_validate`,
# with a {py:class}`~sklearn.model_selection.KFold` splitter, repeats the split for us
# and stores the results in a Python dictionary.
#
# We do not pass a scorer. For a regressor, the default score is the R² coefficient:
# a higher score is better. This score is not the median absolute error from the
# previous cells.

# %%
from sklearn.model_selection import KFold, cross_validate

cv = KFold(n_splits=5, shuffle=True, random_state=0)
cv_results = cross_validate(
    linear_regression,
    X,
    y,
    cv=cv,
    return_train_score=True,
)
cv_results = pd.DataFrame(cv_results)
cv_results

# %%
cv_results[["train_score", "test_score"]].describe()

# %% [markdown]
#
# {py:func}`skore.evaluate` runs the same kind of evaluation and reports the metrics directly.
# The mean absolute error is available under the name `mean_absolute_error`, with no
# `neg_` prefix.

# %%
from skore import evaluate

report = evaluate(linear_regression, X, y, splitter=cv)
report.metrics.summarize().frame()

# %%
report.metrics.add("mean_absolute_error")
report.metrics.get("mean_absolute_error")

# %% [markdown]
#
# We can pass both models to the same function. The summary then gives the mean and
# the standard deviation of each metric across the splits. This is the comparison that
# a single split could not support.

# %%
comparison = evaluate(
    {"linear": linear_regression, "quantile": quantile_regression},
    X,
    y,
    splitter=cv,
)
comparison.metrics.summarize().frame()
