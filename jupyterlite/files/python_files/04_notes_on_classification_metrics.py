# %% [markdown]
#
# # Classification metrics
#
# This notebook looks at three ways to score a classifier: the probabilities it
# produces, the ranking of those probabilities, and the class it predicts once a
# threshold is chosen.
#
# We come back to the penguins dataset and keep two species, with Chinstrap as the
# positive class.

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

categorical_columns = X.select_dtypes(include=["object", "str"]).columns
X[categorical_columns] = X[categorical_columns].astype("category")

# %% [markdown]
#
# The class name is printed on the confusion matrix, so we keep the common name.

# %%
positive = "Chinstrap"
y = y.replace(
    {
        "Adelie Penguin (Pygoscelis adeliae)": "Adelie",
        "Chinstrap penguin (Pygoscelis antarctica)": "Chinstrap",
    }
)
classes_to_keep = ["Adelie", positive]
X = X[y.isin(classes_to_keep)]
y = y[y.isin(classes_to_keep)]

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

hist_gradient_boosting = HistGradientBoostingClassifier(categorical_features="from_dtype")

# %% [markdown]
#
# ## One report
#
# `evaluate` fits the model on one part of the data and scores it on the held-out part.
# Each figure below is that single test set. Average precision is not a built-in metric,
# so we register it on the report and call it with the others.

# %%
from skore import evaluate

report = evaluate(hist_gradient_boosting, X, y, pos_label=positive, splitter=0.5)
report.metrics.add("average_precision")
report

# %% [markdown]
#
# ## Probability estimates
#
# `predict_proba` returns a probability for each class. With a positive class set, the
# report gives the probability of that class. A few test rows:

# %%
probabilities = report.get_predictions(data_source="test", response_method="predict_proba")
pd.Series(probabilities, name=positive).head()

# %% [markdown]
#
# Log loss scores these probabilities. It is a proper scoring rule: it is smallest when
# the predicted probabilities match the true probabilities. The Brier score is the other
# usual proper scoring rule. Accuracy only sees the class chosen after a threshold, so
# it does not score the probabilities.
#
# We can decompose the Brier score after grouping predictions into bins:
#
# $$
# \text{Brier} =
# \underbrace{\sum_b \frac{n_b}{n}(\bar{p}_b - \bar{y}_b)^2}_{\text{reliability}}
# -
# \underbrace{\sum_b \frac{n_b}{n}(\bar{y}_b - \bar{y})^2}_{\text{resolution}}
# +
# \underbrace{\bar{y}(1 - \bar{y})}_{\text{uncertainty}}.
# $$
#
# Here, $\bar{p}_b$ is the mean predicted probability in bin $b$, $\bar{y}_b$ is its
# observed positive fraction, and $n_b$ is its number of samples. Reliability is the
# calibration loss: it is small when the probabilities agree with the observed
# frequencies. Resolution rewards groups with different observed frequencies, so it is
# related to separating and ranking positive and negative cases. Uncertainty is
# $\bar{y}(1 - \bar{y})$, the variance of the outcome. It depends only on how common the
# positive class is, so the model cannot change it. It is largest when the two classes
# are equally common. ROC AUC and average precision score the ranking directly; they do
# not assess calibration. Proper scoring rules assess both the probability values and
# this ability to separate cases.

# %%
report.metrics.log_loss()

# %% [markdown]
#
# The calibration curve draws those bins. The horizontal position of a point is
# $\bar{p}_b$, the mean predicted probability in the bin, and the vertical position is
# $\bar{y}_b$, the observed fraction of Chinstrap penguins. A calibrated model lies on
# the diagonal, where the two agree.

# %%
report.inspection.calibration_curve(n_bins="auto").plot()

# %% [markdown]
#
# ## Ranking
#
# A ranking metric judges the order of the scores, not their magnitude.
#
# The ROC curve follows the true-positive rate and the false-positive rate as the
# threshold moves. The area under that curve is the ROC AUC.

# %%
report.metrics.roc_auc()

# %%
report.metrics.roc().plot()

# %% [markdown]
#
# The precision-recall curve follows precision and recall over the same thresholds.
# Average precision summarizes that curve. At each threshold it takes the precision
# and weights it by the gain in recall since the previous threshold.

# %%
report.metrics.precision_recall().plot()

# %%
report.metrics.average_precision()

# %% [markdown]
#
# ## Thresholded decisions
#
# A confusion matrix is the decision at one threshold. The default cut is a probability
# of 0.5. Precision and recall are counts from that matrix, so they move when the cut
# moves.

# %%
report.metrics.confusion_matrix().plot()

# %% [markdown]
#
# The same matrix at a probability of 0.1. A lower cut predicts Chinstrap more often.

# %%
report.metrics.confusion_matrix().plot(threshold_value=0.1)

# %% [markdown]
#
# `FixedThresholdClassifier` applies a threshold we choose. It is useful when the cost
# of a mistake is known in advance. A low threshold predicts the positive class more
# often, which is what we want when missing a positive case is expensive.

# %%
from sklearn.model_selection import FixedThresholdClassifier

fixed_threshold = FixedThresholdClassifier(hist_gradient_boosting, threshold=0.1)
fixed_threshold.fit(report.X_train, report.y_train)
fixed_threshold.threshold

# %% [markdown]
#
# `TunedThresholdClassifierCV` searches for the threshold that maximizes a metric, here
# the F1 score of the positive class. The search uses cross-validation. The fitted
# model stays the same; only the decision cut changes.

# %%
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import TunedThresholdClassifierCV

scorer = make_scorer(f1_score, pos_label=positive)
tuned_threshold = TunedThresholdClassifierCV(hist_gradient_boosting, scoring=scorer, cv=5)
tuned_threshold.fit(report.X_train, report.y_train)
tuned_threshold.best_threshold_

# %% [markdown]
#
# ## Conclusion
#
# A classifier can be scored in three ways, and each way answers a different question.
# Log loss and the Brier score judge the probabilities themselves. The Brier score splits
# into reliability, resolution, and uncertainty, and the calibration curve shows the
# reliability term: predicted probabilities against the observed fraction of the positive
# class. ROC AUC and average precision judge the ranking of those scores, whatever
# threshold we later choose. A confusion matrix, precision, and recall are the decision
# at one cut. `FixedThresholdClassifier` applies a cut we choose, and
# `TunedThresholdClassifierCV` searches for the cut that maximizes a metric. One
# `evaluate` report holds the probabilities, the curves, and both matrices.
