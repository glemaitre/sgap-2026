# %% [markdown]
#
# # Cross-validation strategies
#
# The previous notebooks introduced how to evaluate a model and how to build a
# preprocessing pipeline that depends on that model.
#
# A cross-validation score is useful when the split emulates what will happen in
# production.

# %% [markdown]
#
# ## The split has to match production
#
# The training rows and the test rows have to stand in for the data we will meet later.
# The rule that separates them has to be the rule of production. The figures in
# [Visualizing cross-validation behavior](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html)
# show which rows each splitter puts in the training set and in the test set.
#
# ### Time series
#
# In production we predict a later period from earlier data. A shuffled split puts
# future rows into the training set. `TimeSeriesSplit` keeps each test block after its
# training block.
#
# ### Groups
#
# In production we meet a new writer, patient, or site. A shuffled split puts other rows
# from the same group on both sides of the split. `GroupKFold` keeps each group entirely
# in the training set or entirely in the test set.
#
# ### Stratification
#
# Forcing the same class mix into every fold reduces the variance of the scores. That
# reduction is a bias: production does not hand over a batch with the same mix.
# Stratification is therefore usually not the requirement.

# %% [markdown]
#
# ## Digits from several writers
#
# The digits dataset contains handwritten digits from several writers. Three pixel
# columns never vary, so we drop them. We then fit a logistic regression on the scaled
# pixels.

# %%
from sklearn.datasets import load_digits

X, y = load_digits(return_X_y=True)
X = X[:, X.std(axis=0) > 0]

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

logistic_regression = make_pipeline(MinMaxScaler(), LogisticRegression())

# %% [markdown]
#
# The samples are ordered by writer. This `groups` array gives the writer of each
# sample.

# %%
from itertools import count

import numpy as np

# Define the lower and upper bounds of sample indices for each writer.
writer_boundaries = [
    0,
    130,
    256,
    386,
    516,
    646,
    776,
    915,
    1029,
    1157,
    1287,
    1415,
    1545,
    1667,
    1797,
]
groups = np.zeros_like(y)
lower_bounds = writer_boundaries[:-1]
upper_bounds = writer_boundaries[1:]

for group_id, lower, upper in zip(count(), lower_bounds, upper_bounds):
    groups[lower:upper] = group_id

# %%
import matplotlib.pyplot as plt

plt.plot(groups)
plt.yticks(np.unique(groups))
plt.xticks(writer_boundaries, rotation=90)
plt.xlabel("Sample index")
plt.ylabel("Writer index")
_ = plt.title("Writer of each sample")

# %% [markdown]
#
# `cross_validate` forwards `groups` to the splitter. With a `GroupKFold`, each writer
# stays entirely in the training rows or entirely in the test rows. A `KFold` and a
# `StratifiedKFold` do not use these groups.

# %%
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, cross_validate

splitters = {
    "GroupKFold": GroupKFold(n_splits=13),
    "KFold": KFold(n_splits=13),
    "StratifiedKFold": StratifiedKFold(n_splits=13),
}
for name, cv in splitters.items():
    cv_groups = groups if name == "GroupKFold" else None
    cv_results = cross_validate(logistic_regression, X, y, groups=cv_groups, cv=cv)
    print(
        f"{name}: {cv_results['test_score'].mean():.3f} +/- "
        f"{cv_results['test_score'].std():.3f}"
    )

# %% [markdown]
#
# The `KFold` and the `StratifiedKFold` scores sit higher, because samples from the same
# writer appear on both sides of those splits. The `GroupKFold` score is the one we
# would expect on a new writer.
