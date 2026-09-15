# Logistic Regression

This repository teaches multi-class logistic regression using scikit-learn across three concrete datasets. You work through two parallel tracks: a homework notebook where you fill in code to classify building heating loads and handwritten digits, and a full lesson notebook that walks through Iris species classification end-to-end - covering data exploration, model fitting, performance evaluation, decision boundary visualization, and feature scaling. By the end, you will understand not just how to call `LogisticRegression`, but why each modeling and preprocessing decision is made.

---

## Learning Objectives

- Distinguish classification from regression and identify when each applies
- Convert a continuous target variable into discrete classes using binning
- Fit a logistic regression model using scikit-learn's `LogisticRegression`
- Compare One-vs-Rest (OvR) and Softmax (multinomial) multi-class strategies
- Apply Min-Max and Standard Normal feature scaling and measure the effect on accuracy
- Visualize decision boundaries by scoring a 2D mesh grid and plotting predicted class regions
- Interpret a confusion matrix and derive precision and recall from it
- Recognize why accuracy alone is an insufficient performance metric

---

## Data / File Dictionary

| File | Description |
|------|-------------|
| `logistic_regression_homework.ipynb` | Guided homework notebook - energy load classification and handwritten digit classification with fill-in exercises |
| `logistic_regression_sklearn_lesson.ipynb` | Full teaching notebook - Iris species classification with decision boundary visualization and scaling comparison |
| `Energy.csv` | 768 building configurations with 8 physical attributes (compactness, surface area, wall area, etc.) and heating load target |
| `iris_classification.csv` | 150 Iris flower measurements (sepal length and width) labeled by species: setosa, versicolor, virginica |
| `requirements.txt` | Python package dependencies |
| `assets/` | Reference images embedded in the notebooks (confusion matrix diagram, Iris photo, expected output screenshots) |
| `images/` | Banner and thumbnail images used in notebook headers |
| `archive/` | Legacy notebooks and PDFs from earlier versions of the material - not part of the primary workflow |

### Energy dataset columns

| Column | Meaning |
|--------|---------|
| `X1` | Relative compactness |
| `X2` | Surface area (m^2) |
| `X3` | Wall area (m^2) |
| `X4` | Roof area (m^2) |
| `X5` | Overall height (m) |
| `X6` | Orientation (2-5) |
| `X7` | Glazing area (fraction of floor area) |
| `X8` | Glazing area distribution |
| `Y1` | Heating load (kWh/m^2) - target variable |

---

## Workflow Diagram

```
HOMEWORK NOTEBOOK                     LESSON NOTEBOOK
Energy.csv   sklearn digits           iris_classification.csv
     |              |                          |
     v              v                          v
[ Load & describe ]  [ Load digits dataset ]  [ Load & inspect ]
     |              |                          |
     v              v                          v
[ Check nulls  ]  [ Visualize samples ]    [ Check class balance ]
     |              |                       [ Encode labels ]
     v              v                          |
[ Bin Y1 -> Low/Med/High ]             [ Train/test split 80:20 ]
     |              |                          |
     v              v                          v
[ Train/test split 80:20 ]         [ Fit LogisticRegression (Softmax) ]
     |                                         |
   --+------------------              ---------+---------
   |                  |               |                 |
   v                  v               v                 v
OvR model      Multinomial       Training           Test accuracy
                 model           accuracy
   |                  |               |
   v                  v               v
Min-Max scale  Standard scale    Precision, Recall, Confusion matrix
   |                  |               |
   v                  v               v
Compare 4 model accuracies      Decision boundary
                                visualization (2D mesh grid)
                                         |
                                         v
                                Standard scaling comparison
```

---

## Step-by-Step Walkthrough

### Part 1 - Understanding the data (both notebooks)

Before fitting any model, you describe the data: ranges, null counts, and feature distributions. This step is not optional - you need to know whether features are on similar scales before deciding whether to scale them, and you need to catch nulls before they silently corrupt your model.

The homework notebook asks you to compute these statistics yourself using `DataFrame.describe` and `isnull`. The lesson notebook computes them and then plots per-class feature distributions, which reveals which features carry the most discriminative signal between species.

### Part 2 - Converting regression to classification

The energy dataset has a continuous target (`Y1`). To turn this into a classification problem, you bin it into three categories:
- Low: heating load 0-14 kWh/m^2
- Medium: 14-28 kWh/m^2
- High: above 28 kWh/m^2

This is a deliberate design choice - logistic regression is a classifier, not a regressor. By bucketing Y1 with `pd.cut`, you can apply the same model structure to a physically meaningful multi-class problem. The bin boundaries are domain-driven, not data-driven, which is a separate discussion from automatic binning methods like quantiles.

### Part 3 - One-vs-Rest vs. Softmax

When there are more than two classes, logistic regression needs a strategy for extending the binary sigmoid to a multi-class probability estimate. Two options:

- **One-vs-Rest (OvR)**: trains one binary classifier per class (Low vs. not-Low, Medium vs. not-Medium, High vs. not-High). Probabilities from each classifier are not guaranteed to sum to 1.
- **Softmax (multinomial)**: trains a single model that outputs a probability distribution over all classes simultaneously. Probabilities always sum to 1.

You fit both and compare accuracy on training and test data. The goal is not just to see which wins numerically - it is to understand that the choice of multi-class strategy is a modeling decision with structural consequences, not merely a hyperparameter.

### Part 4 - Feature scaling

Some features span very different numerical ranges. Gradient-based solvers (lbfgs, newton-cg) update weights proportionally to feature magnitude, so a feature with range 0-800 will dominate updates over a feature with range 0-1, even if both are equally predictive. You compare two scalers:

- **Min-Max scaler**: compresses every feature to [0, 1] by subtracting the column minimum and dividing by the range
- **Standard scaler**: transforms each feature to have mean 0 and standard deviation 1

Critically, you fit each scaler on the training set only, then apply the learned parameters to transform both train and test. Fitting on training data only prevents information about the test set from leaking into the normalization parameters - a subtle form of data leakage that inflates test accuracy.

### Part 5 - Decision boundary visualization (lesson notebook)

The lesson notebook adds a visualization step not present in the homework: a 2D decision boundary plot. Because the Iris dataset uses only two input features (sepal length and sepal width), you can create a fine mesh grid that covers the entire feature space, predict the class for every grid point, and color the grid by predicted class. Overlaying the training and test points on top reveals exactly where the model's class boundaries fall and which points are misclassified.

This is a diagnostic tool, not part of the fitting process. A cluster of misclassified test points in the wrong color region tells you where in feature space the model's boundary is poorly placed - information a scalar accuracy number cannot convey.

### Part 6 - Precision, recall, and the confusion matrix

Accuracy tells you how many predictions were correct overall, but it hides where the model fails. The confusion matrix breaks predictions down by class - it shows you which classes the model confuses, not just how often it errs.

Precision and recall each answer a different question:
- Precision: of all the times the model predicted class X, how often was it right?
- Recall: of all the actual class X instances, how many did the model catch?

A model that never predicts class X has perfect precision (0 false positives) but zero recall. In the digits notebook, the confusion matrix reveals which digit pairs the model most frequently swaps - a finding that a single accuracy number would conceal entirely.

---

## How to Run

Prerequisites: [Nix](https://nixos.org/download) with flakes enabled.

This repo's Python environment is fully project-local - a `flake.nix` devShell provides Python and `uv`, and `uv` installs every dependency pinned in `pyproject.toml`/`uv.lock` into a `.venv` inside this directory. Nothing is installed system-wide, and nothing here needs to be added to `home.nix` or `configuration.nix`.

```bash
# Clone the repo
git clone https://github.com/ehcastroh-teach/Logistic_Regression.git
cd Logistic_Regression

# Enter the project's dev shell - this also runs `uv sync` automatically
# the first time, creating .venv with every pinned dependency installed
nix develop

# Homework notebook (fill in the blanks)
uv run jupyter notebook logistic_regression_homework.ipynb

# Full teaching notebook (read-through with all code)
uv run jupyter notebook logistic_regression_sklearn_lesson.ipynb
```

If you don't use Nix, any Python 3.12+ environment with `uv` installed works the same way: run `uv sync` in place of `nix develop` and use the `uv run ...` commands above unchanged.

Work through the homework notebook top to bottom. Fill in `### YOUR CODE HERE ###` sections and run each cell to check your result against the expected output images in `assets/`. Refer to the lesson notebook if you want to see a worked example on a different dataset.

---

## Key Concepts Glossary

| Term | Definition |
|------|-----------|
| Logistic regression | A classification model that estimates the probability an input belongs to each class, using the logistic (sigmoid) function to bound outputs to [0, 1] |
| Multi-class classification | A task where the target can take one of three or more discrete values |
| One-vs-Rest (OvR) | A multi-class strategy that trains one binary classifier per class, treating each class as positive and all others as negative |
| Softmax regression | A multi-class strategy that produces a single probability distribution over all classes simultaneously; probabilities always sum to 1 |
| Feature scaling | Transforming features so they share a common numerical range, reducing sensitivity to magnitude differences between columns |
| Min-Max scaler | Scales each feature to [0, 1] by subtracting the column minimum and dividing by the range |
| Standard scaler | Scales each feature to mean 0 and standard deviation 1 |
| Binning | Converting a continuous variable into discrete categories by defining threshold ranges; used here to turn a regression target into a classification target |
| Decision boundary | The surface in feature space where the model is equally likely to predict any class; visualized in the lesson notebook by scoring a mesh grid |
| Confusion matrix | A table showing predicted vs. actual class labels for all test instances; reveals which classes the model confuses |
| Precision | The fraction of predicted positives that are actually positive: TP / (TP + FP) |
| Recall | The fraction of actual positives that were correctly predicted: TP / (TP + FN) |
| F1-score | The harmonic mean of precision and recall; useful when class imbalance makes accuracy misleading |
| Train/test split | Partitioning data into a training set (used to fit the model) and a test set (used to evaluate it on unseen data); prevents overfitting evaluation |

---

## Further Reading

- "An Introduction to Statistical Learning" - Chapter 4 (classification, logistic regression)
- "The Elements of Statistical Learning" - Chapter 4 (linear methods for classification)
- scikit-learn documentation: `sklearn.linear_model.LogisticRegression`
- scikit-learn documentation: `sklearn.metrics` (confusion matrix, precision, recall, F1)
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" - Chapter 3 (classification)
- scikit-learn documentation: Decision boundary visualization examples

---

## Credits and Acknowledgements

Energy efficiency dataset: publicly available benchmark dataset for building energy simulation.

Iris dataset: classic benchmark dataset from the UCI Machine Learning Repository.

Digits dataset: built-in sklearn dataset adapted from the UCI ML handwritten digits collection.

---

## Contact

<div align="center">
  <img src="images/thumbnails/ehcastroh_teach_banner_flower.png" alt="ehcastroh" width="90" style="border-radius: 50%;" />

  <sub>ehcastroh</sub>

  <a href="https://github.com/ehcastroh">GitHub</a> · <a href="https://www.linkedin.com/in/ehcastroh/">LinkedIn</a>
</div>
