# %% [markdown]
#
# # Working with heterogeneous data
#
# In this notebook, we will present how to handle heterogeneous data. Usually, examples
# only show how to deal with numerical data but in practice, we often have to deal with
# a mix of numerical and categorical data.
#
# So let's look at the entire penguins dataset this time.

# %% [markdown]
#
# ## Environment setup
#
# Run this cell in JupyterLite. A local installation already provides these packages.

# %%
# %pip install -q "scikit-learn~=1.9.1" "skrub~=0.11.0" "skore~=0.29.0" pandas matplotlib

# %%
import pandas as pd

df = pd.read_csv("../datasets/penguins.csv")
df

# %% [markdown]
#
# ## A classification problem
#
# We see that we have some strings and numbers in this dataset. Let's set up a
# classification problem: we want to predict the species of the penguins given some
# numerical and categorical features. `Region` is the same value for every penguin, so
# we leave it out.

# %%
feature_names = [
    "Island",
    "Culmen Length (mm)",
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
# ## A single train-test split
#
# Before evaluating a model through cross-validation, we will first look at a model
# using a single split.

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# %% [markdown]
#
# ### Exercise
#
# - List every problem you can think of with fitting a model on this training data as
#   it is.
# - Fit a `LogisticRegression` model on the training data.

# %%
#
# Your code comes here.

# %% [markdown]
#
# ## A table report
#
# A `TableReport` summarizes each column of the full table `X`: the data type, the
# missing values, and the distribution of the values.

# %%
import skrub

table_report = skrub.TableReport(X)
table_report

# %% [markdown]
#
# Machine learning models do not necessarily handle non-numeric data and missing values.
# We need a vectorization that turns this raw table into numeric data. That
# vectorization also has to make sense for the statistical model we want to use. A
# linear model does not treat a category code the way a tree model does.
#
# ## Missing values in numerical data
#
# The numerical columns can contain missing values. A `LogisticRegression` does not fit
# those columns until the missing entries are filled in.
#
# So life is difficult. So let's start by looking at the numerical part.

# %%
X_numeric = X.select_dtypes(include="number")

# %% [markdown]
#
# Let's fit a `LogisticRegression` on the numerical columns.

# %%
logistic_regression = LogisticRegression()
try:
    logistic_regression.fit(X_numeric, y)
except ValueError as error:
    print(error)

# %% [markdown]
#
# The fit fails. The numerical columns contain missing values, and `LogisticRegression`
# does not handle them.
#
# ### Which statistical strategy
#
# One strategy is to replace each missing entry by the mean of the observed values in
# its column. For a linear model this is a simple, explicit choice: the filled value is
# a summary of the column, and the model then sees a complete numeric matrix.
#
# ### How to apply the strategy
#
# The mean has to be computed on the rows used to fit the model. If we compute it on
# the whole column, the held-out rows take part in that summary. Information from data
# we pretend not to have seen leaks into the training step, and the error on those rows
# no longer stands for future production data.
#
# A `Pipeline`, created with `make_pipeline`, keeps the strategy and the model together.
# `fit` learns the column means on the training rows, imputes those rows, and then fits
# the logistic regression. `predict` applies the stored means to new rows and then
# predicts. During cross-validation the whole pipeline is refitted on each training
# fold, so the held-out fold never enters the mean.
#
# ![During fit, the training data go through the transformer fit_transform, and the transformed data together with the training target go through the predictor fit.](../images/pipeline_fit.png)
#
# ![During predict, the testing data go through the transformer transform, and the transformed data go through the predictor predict.](../images/pipeline_predict.png)

# %%
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

logistic_regression = make_pipeline(SimpleImputer(), LogisticRegression())
logistic_regression.fit(X_numeric, y)

# %% [markdown]
#
# ## Scaling numerical values
#
# The pipeline in the previous section fits, and it raises a convergence warning. Let's
# check how many iterations were performed.

# %%
logistic_regression

# %%
logistic_regression[-1].n_iter_

# %% [markdown]
#
# Indeed, we reached the maximum number of iterations. We can increase the number of
# iterations. Let's check which parameters we can set with the `get_params` method.

# %%
logistic_regression.get_params()

# %% [markdown]
#
# We can set the `max_iter` parameter to a higher value through the variable
# `logisticregression__max_iter`.

# %%
logistic_regression.set_params(logisticregression__max_iter=10_000)
logistic_regression.fit(X_numeric, y)

# %%
logistic_regression[-1].n_iter_

# %% [markdown]
#
# The model now converges, but it required almost 2,500 iterations. The warning message
# mentioned that we could try to scale the data. Let's try to scale the data using a
# `StandardScaler`. We can then check if the convergence is faster.

# %%
from sklearn.preprocessing import StandardScaler

logistic_regression = make_pipeline(StandardScaler(), SimpleImputer(), LogisticRegression())
logistic_regression.fit(X_numeric, y)

# %%
logistic_regression[-1].n_iter_

# %% [markdown]
#
# It only requires 11 iterations now and will provide the exact same result. Scaling
# does not change the model we want to fit. It changes the path gradient descent takes
# to reach it. Without scaling, the steps zigzag because the features have very
# different ranges. With scaled features, the steps point more directly toward the
# minimum.
#
# ![Gradient descent zigzags when the features are not scaled, and goes more directly to the minimum when they are.](../images/gradient_descent_scaling.png)

# %% [markdown]
#
# We can now evaluate the model using cross-validation.
# Because the imputer is inside the pipeline, each fold learns its means from the
# training rows of that fold only.
#
# ### Exercise
#
# Use `skore.evaluate` to check the performance of this model with 10-fold
# cross-validation.

# %%
#
# Your code comes here.

# %% [markdown]
#
# ## Categorical features
#
# Now, let's only consider the categorical data.

# %%
X_categorical = X.select_dtypes(exclude="number")
X_categorical

# %% [markdown]
#
# ### Exercise
#
# - Think of a way to transform the string categories into numerical data.

# %% [markdown]
#
# ### Ordinal encoding
#
# We need to find a strategy to "encode" the categorical data into numerical data. The
# simplest strategy is to use an `OrdinalEncoder`.

# %%
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder().set_output(transform="pandas")
X_encoded = ordinal_encoder.fit_transform(X_categorical)
X_encoded

# %% [markdown]
#
# It replaces a category by an integer. However, with linear models, it means that we
# would assume that the difference between two categories is the same. Also, there is
# an ordering imposed by this transformation.

# %% [markdown]
#
# ### One-hot encoding
#
# If this modelling assumption is not desired, we can use a one-hot encoding.

# %%
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
X_encoded = one_hot_encoder.fit_transform(X_categorical)
X_encoded

# %% [markdown]
#
# In this case, we create independent binary columns for each category. We therefore
# have an individual coefficient for each category. Usually, this is a more appropriate
# encoding for linear models.
#
# ### Exercise
#
# A fold can contain a category in the left-out rows that is absent from its training
# rows. Check the documentation of
# [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
# and set the parameter that handles an unknown category.
#
# - Build a pipeline with a `OneHotEncoder` and a `LogisticRegression` on the
#   categorical columns.
# - Use `skore.evaluate` to check the performance of this model with 10-fold
#   cross-validation.

# %%
#
# Your code comes here.

# %% [markdown]
#
# ## Combining numerical and categorical features
#
# Now, we need to combine both numerical and categorical preprocessing and feed the
# output to a single linear model. The `ColumnTransformer` class is designed for this
# purpose: we provide a list of columns so that they are transformed by a specific
# transformer (or a pipeline of transformers). This `ColumnTransformer` can be used as
# a preprocessing stage of a pipeline containing a linear model as the final stage.
#
# The figure below shows the three steps. Each group of columns is selected, passed
# through its own transformer, and the transformed columns are concatenated.
#
# ![A ColumnTransformer selects groups of columns, applies a different transformer to each group, and concatenates the transformed columns.](../images/column_transformer.png)

# %%
from sklearn.compose import make_column_selector as selector

numerical_columns = selector(dtype_include="number")
numerical_columns(X)

# %%
from sklearn.compose import make_column_transformer

numerical_selector = selector(dtype_include="number")
categorical_selector = selector(dtype_exclude="number")
preprocessor = make_column_transformer(
    (make_pipeline(StandardScaler(), SimpleImputer()), numerical_selector),
    (OneHotEncoder(handle_unknown="ignore"), categorical_selector),
)
logistic_regression = make_pipeline(preprocessor, LogisticRegression())
logistic_regression

# %%
import skore

report = skore.evaluate(logistic_regression, X, y, splitter=10)
report

# %% [markdown]
#
# ## Numbers, categories, text, and dates
#
# The preprocessing above is the one we usually want: scale and impute the numerical
# columns, and one-hot encode the categorical columns. A table can also contain text or
# dates, and those columns need their own encoding.
#
# [TableVectorizer](https://skrub-data.org/stable/reference/generated/skrub.TableVectorizer.html#skrub.TableVectorizer)
# detects these column types and applies a suitable encoder to each of them. It is easier
# to use on the standard types of data, and each encoder can still be configured. Text
# columns are encoded with a `StringEncoder`, and date columns with a `DatetimeEncoder`.
#
# To recover the same preprocessing as above, we scale and impute the numerical columns,
# and we one-hot encode the low-cardinality columns while ignoring unknown categories.
# `sparse_output=False` is required because `TableVectorizer` returns a dataframe.

# %%
from skrub import TableVectorizer

table_vectorizer = TableVectorizer(
    numeric=make_pipeline(StandardScaler(), SimpleImputer()),
    low_cardinality=OneHotEncoder(handle_unknown="ignore", sparse_output=False),
)
logistic_regression = make_pipeline(table_vectorizer, LogisticRegression())
logistic_regression

# %%
report = skore.evaluate(logistic_regression, X, y, splitter=10)
report

# %% [markdown]
#
# ## Choosing the preprocessing from the model
#
# [tabular_pipeline](https://skrub-data.org/stable/reference/generated/skrub.tabular_pipeline.html#skrub.tabular_pipeline)
# goes one step further. We pass it the predictor, and it chooses the `TableVectorizer`
# and the following steps from that model.
#
# A `LogisticRegression` cannot handle missing values, so the pipeline imputes them and
# adds an indicator column for each missing entry. It also scales the numerical columns
# with a `SquashingScaler`.

# %%
from skrub import tabular_pipeline

logistic_regression = tabular_pipeline(LogisticRegression())
logistic_regression

# %%
report = skore.evaluate(logistic_regression, X, y, splitter=10)
report

# %% [markdown]
#
# ## Tree-based models
#
# We gave basic preprocessing steps for a linear model. However, there is another group
# of models that can handle heterogeneous data: tree-based models.
#
# ### Exercise
#
# Reuse `tabular_pipeline` with a `HistGradientBoostingClassifier`.
#
# - Pass a `HistGradientBoostingClassifier` to `tabular_pipeline`.
# - Evaluate this pipeline with `skore.evaluate` on a 10-fold split.
# - What modelling choices were made for this model?

# %%
#
# Your code comes here.

# %% [markdown]
#
# ## Conclusion
#
# Heterogeneous data mixes numbers, categories, and sometimes text or dates. Turning
# that table into model inputs has to suit the predictor we fit. For a linear model,
# numerical columns are imputed and scaled, and categories are one-hot encoded, with a
# rule for categories that appear only in the left-out rows. A `Pipeline` learns this
# preprocessing on the training rows of each split. `TableVectorizer` covers the usual
# column types, and `tabular_pipeline` chooses the encoders from the predictor we pass
# it. A gradient boosting model keeps low-cardinality columns as a categorical dtype
# and handles missing values itself. `evaluate` reports the performance of each choice.
