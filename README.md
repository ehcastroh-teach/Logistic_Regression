# Logistic Regression

This repository introduces logistic regression for multi-class classification using scikit-learn. You will work through two concrete problems: predicting a building's heating load category from physical measurements, and classifying handwritten digits. A companion teaching notebook covers the Iris dataset end-to-end, from raw data to decision boundary visualization, showing every step a practitioner would take.

---

## Learning Objectives

- Distinguish classification from regression and understand when each applies
- Convert a continuous target variable into discrete classes using binning
- Fit a logistic regression model using scikit-learn's `LogisticRegression`
- Compare One-vs-Rest (OvR) and Softmax (multinomial) multi-class strategies
- Apply Min-Max and Standard Normal feature scaling and measure the effect on accuracy
- Interpret a confusion matrix and derive precision and recall from it
- Recognize why accuracy alone is an insufficient performance metric

---

## Data / File Dictionary

| File | Description |
|------|-------------|
| `hw-m140-logistic-reg-sklearn.ipynb` | Guided homework notebook - energy load classification and handwritten digit classification with fill-in exercises |
| `nb-m140-logistic-reg-sklearn.ipynb` | Full teaching notebook - Iris species classification with decision boundary visualization and scaling comparison |
| `Energy.csv` | 768 building configurations with 8 physical attributes (compactness, surface area, wall area, etc.) and heating load target |
| `iris_classification.csv` | 150 Iris flower measurements (sepal length and width) labeled by species: setosa, versicolor, virginica |
| `requirements.txt` | Python package dependencies |
| `assets/` | Reference images embedded in the notebooks (confusion matrix diagram, Iris photo, expected output screenshots) |
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
Energy.csv                    iris_classification.csv
     |                                   |
     v                                   v
[ Load with pandas ]           [ Load with pandas ]
     |                                   |
     v                                   v
[ Describe & check nulls ]     [ Shuffle & encode labels ]
     |                                   |
     v                                   v
[ Bin Y1 into Low/Med/High ]   [ Train/test split ]
     |                                   |
     v                                   v
[ Train/test split 80:20 ]     [ Fit LogisticRegression ]
     |                                   |
   --+---------------------------     ---+---
   |                           |     |       |
   v                           v     v       v
OvR model              Multinomial  Accuracy Confusion
                         model      metrics  matrix
   |                           |
   v                           v
Min-Max scale           Standard scale      Decision
   |                           |            boundaries
   v                           v            (plotted)
Fit + compare accuracies across all four models
```

---

## Step-by-Step Walkthrough

### Part 1 - Understanding the data (both notebooks)

Before fitting any model, you describe the data: ranges, null counts, and feature distributions. This step is not optional - you need to know whether features are on similar scales before deciding whether to scale them, and you need to catch nulls before they silently corrupt your model.

The homework (`hw`) asks you to compute these statistics yourself using `DataFrame.describe` and `isnull`. The teaching notebook (`nb`) computes them automatically so you can see the expected output.

### Part 2 - Converting regression to classification

The energy dataset has a continuous target (`Y1`). To turn this into a classification problem, you bin it into three categories:
- Low: heating load 0-14 kWh/m^2
- Medium: 14-28 kWh/m^2
- High: above 28 kWh/m^2

This is a deliberate design choice - logistic regression is a classifier, not a regressor. By bucketing Y1, you can apply the same model structure to a physically meaningful multi-class problem.

### Part 3 - One-vs-Rest vs. Softmax

When there are more than two classes, logistic regression needs a strategy for producing probabilities across all classes. Two options:

- **One-vs-Rest (OvR)**: trains one binary classifier per class (Low vs. not-Low, Medium vs. not-Medium, High vs. not-High). Probabilities from each classifier are not guaranteed to sum to 1.
- **Softmax (multinomial)**: trains a single model that outputs a probability distribution over all classes simultaneously. Probabilities always sum to 1.

You fit both and compare accuracy on training and test data to understand the practical difference.

### Part 4 - Feature scaling

Some features span very different numerical ranges. Distance-based and gradient-based algorithms are sensitive to this. You compare two scalers:

- **Min-Max scaler**: compresses every feature to [0, 1]
- **Standard scaler**: transforms each feature to have mean 0 and standard deviation 1

You fit each scaler on the training set only, then transform both train and test. Fitting on train-only prevents information from the test set leaking into the normalization parameters.

### Part 5 - Precision, recall, and the confusion matrix

Accuracy tells you how many predictions were correct overall, but it hides where the model fails. The confusion matrix breaks predictions down by class - it shows you which classes the model confuses, not just how often it errs.

Precision and recall each answer a different question:
- Precision: of all the times the model predicted class X, how often was it right?
- Recall: of all the actual class X instances, how many did the model catch?

A model that never predicts class X has perfect precision (0 false positives) but zero recall. Understanding this trade-off matters in any real application.

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/ehcastroh-teach/Logistic_Regression.git
cd Logistic_Regression

# Install dependencies
pip install -r requirements.txt

# Homework notebook (fill in the blanks)
jupyter notebook hw-m140-logistic-reg-sklearn.ipynb

# Full teaching notebook (read-through with all code)
jupyter notebook nb-m140-logistic-reg-sklearn.ipynb
```

Work through the homework top to bottom. Fill in `### YOUR CODE HERE ###` sections and run each cell to check your result. Refer to the teaching notebook if you want to see a worked example on a different dataset.

---

## Key Concepts Glossary

| Term | Definition |
|------|-----------|
| Logistic regression | A classification model that estimates the probability an input belongs to each class, using the logistic (sigmoid) function to bound outputs to [0, 1] |
| Multi-class classification | A task where the target can take one of three or more discrete values |
| One-vs-Rest (OvR) | A multi-class strategy that trains one binary classifier per class, treating each class as positive and all others as negative |
| Softmax regression | A multi-class strategy that produces a single probability distribution over all classes simultaneously; probabilities always sum to 1 |
| Feature scaling | Transforming features so they share a common numerical range, reducing sensitivity to magnitude differences between columns |
| Min-Max scaler | Scales each feature to [0, 1] by subtracting the min and dividing by the range |
| Standard scaler | Scales each feature to mean 0 and standard deviation 1 |
| Confusion matrix | A table showing predicted vs. actual class labels for all test instances; reveals which classes the model confuses |
| Precision | The fraction of predicted positives that are actually positive: TP / (TP + FP) |
| Recall | The fraction of actual positives that were correctly predicted: TP / (TP + FN) |
| F1-score | The harmonic mean of precision and recall; useful when class imbalance makes accuracy misleading |
| Decision boundary | The surface in feature space where the model is equally likely to predict any class; separates class regions |
| Binning | Converting a continuous variable into discrete categories by defining threshold ranges |
| train_test_split | Partitioning data into a training set (used to fit the model) and a test set (used to evaluate it on unseen data) |

---

## Further Reading

- "An Introduction to Statistical Learning" - Chapter 4 (classification, logistic regression)
- "The Elements of Statistical Learning" - Chapter 4 (linear methods for classification)
- scikit-learn documentation: `sklearn.linear_model.LogisticRegression`
- scikit-learn documentation: `sklearn.metrics` (confusion matrix, precision, recall, F1)
- "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" - Chapter 3 (classification)

---

## Credits and Acknowledgements

Energy efficiency dataset: publicly available benchmark dataset for building energy simulation.

Iris dataset: classic benchmark dataset from the UCI Machine Learning Repository.

---

## Contact

<div align="center">
  <img src="images/thumbnails/ehcastroh_teach_banner_flower.png" alt="ehcastroh" width="90" style="border-radius: 50%;" />

  <sub>ehcastroh</sub>

  <a href="https://github.com/ehcastroh">GitHub</a> · <a href="https://www.linkedin.com/in/ehcastroh/">LinkedIn</a>
</div>
