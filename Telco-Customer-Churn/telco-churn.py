import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer, confusion_matrix, classification_report, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import joblib

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/Telco-Customer-Churn.csv")
print(df.head())
print(df.shape)
print(df.info())


# ============================================================
# 2. DATA QUALITY CHECKS
# ============================================================

print("Blank Charges", df["TotalCharges"].str.strip().eq("").sum())
print(df["Churn"].value_counts())
print("Duplicates", df.duplicated().sum())


# ============================================================
# 3. DATA CLEANING
# ============================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors = "coerce"
)
print(df["TotalCharges"].isnull().sum())

print(df[df["TotalCharges"].isna()][
          ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]

      ])
df["TotalCharges"] = df["TotalCharges"].fillna(0)


# ============================================================
# 4. DESCRIPTIVE STATISTICS
# ============================================================

print(df.describe())
print(df.nunique())
print(df["Contract"].value_counts())
print(df["InternetService"].value_counts())


# ============================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Churn proportions by contract and internet service
print(pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize = "index"
))

print(pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize = "index"
))

# Average numeric values grouped by churn
print(df.groupby("Churn")[
          ["tenure",
           "MonthlyCharges",
           "TotalCharges"]]
      .mean())


# ============================================================
# 6. VISUALIZATION: NUMERIC FEATURES BY CHURN
# ============================================================

# Average tenure by churn
avg_tenure = df.groupby("Churn")["tenure"].mean()
bars = plt.bar(avg_tenure.index, avg_tenure.values)
plt.bar_label(bars)
plt.xlabel("Churn")
plt.ylabel("average tenure")
plt.title("Tenure VS Churn")
plt.show()


# Average monthly charges by churn
mon_chg = df.groupby("Churn")["MonthlyCharges"].mean()
bars = plt.bar(mon_chg.index, mon_chg.values)
plt.bar_label(bars)
plt.xlabel("Churn")
plt.ylabel("Average Monthly Charges")
plt.title("Monthly Charges VS Churn")
plt.show()


# Average total charges by churn
tot_chg = df.groupby("Churn")["TotalCharges"].mean()
bars = plt.bar(tot_chg.index, tot_chg.values)
plt.bar_label(bars)
plt.xlabel("Churn")
plt.ylabel("Average TotalCharges")
plt.title("TotalCharges VS Churn")
plt.show()


# ============================================================
# 7. VISUALIZATION: CATEGORICAL FEATURES AND CHURN
# ============================================================

# Churn proportions by payment method
print(pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize = "index"
))

# Churn rate by contract
churn_rate = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize = "index"
)
churn_rate = churn_rate["Yes"] * 100
bars = plt.bar(churn_rate.index, churn_rate.values)
plt.bar_label(bars)
plt.xlabel("Churn")
plt.ylabel("Churn Rate (%)")
plt.title("Contract VS Churn")
plt.show()


# Churn proportions by tech support and online security
print(pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize = "index"
))

print(pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"],
    normalize = "index"
))


# Churn rate by internet service
churn_net = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize = "index"
)
churn_net = churn_net["Yes"] * 100
bars = plt.bar(churn_net.index, churn_net.values)
plt.bar_label(bars)
plt.xlabel("Churn")
plt.ylabel("InternetService")
plt.title("Churn vs InternetService")
plt.show()


# ============================================================
# 8. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop(columns = ["Churn", "customerID"])
y = df["Churn"]
print(X.shape)
print(y.shape)


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2,
    random_state = 42,
    stratify = y
)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ============================================================
# 10. IDENTIFY NUMERIC AND CATEGORICAL FEATURES
# ============================================================

numeric_features = df.select_dtypes(include = "number").columns.tolist()
print(len(numeric_features))

categorical_features = X_train.drop(columns = numeric_features).columns.tolist()
print(len(categorical_features))


# ============================================================
# 11. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown = "ignore" ), categorical_features)
])

# Fit preprocessing on training data and transform train/test data
X_train_procs = preprocessor.fit_transform(X_train)
X_test_procs = preprocessor.transform(X_test)

print(X_train_procs.shape)
print(X_test_procs.shape)


# ============================================================
# 12. BASELINE LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression()
model.fit(X_train_procs, y_train)
predictions = model.predict(X_test_procs)
print(predictions)

print("Accuracy :", accuracy_score(y_test, predictions))
print("Confusion_Matrix:", confusion_matrix(y_test, predictions))
print("Classification_report:", classification_report(y_test, predictions))


# ============================================================
# 13. BUILD PIPELINE FOR MODEL TUNING
# ============================================================

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10],
    "classifier__class_weight": [None, "balanced"]
}
pipeline = Pipeline([
    ("processor", preprocessor),
    ("classifier", LogisticRegression())
])


# ============================================================
# 14. DEFINE RECALL SCORER FOR CHURN CLASS
# ============================================================

churn_recall = make_scorer(
    recall_score,
    pos_label = "Yes"

    )


# ============================================================
# 15. GRID SEARCH CROSS-VALIDATION
# ============================================================

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv = 5,
    scoring = churn_recall,
)
grid_search.fit(X_train, y_train)
print("Best Parameter :", grid_search.best_params_)
print("Best CV score :", grid_search.best_score_)


# ============================================================
# 16. EVALUATE THE TUNED MODEL
# ============================================================

best_model = grid_search.best_estimator_
grid_predictions = best_model.predict(X_test)
print(grid_predictions)

print("Accuracy :", accuracy_score(y_test, grid_predictions))
print("Confusion_Matrix:", confusion_matrix(y_test, grid_predictions))
print("Classification_report:", classification_report(y_test, grid_predictions))


# ============================================================
# 17. PREDICT CHURN PROBABILITIES
# ============================================================

probabilities = best_model.predict_proba(X_test)
print(probabilities[:10])
print(best_model.classes_)
churn_probabilities = probabilities[:,1]


# ============================================================
# 18. APPLY CUSTOM DECISION THRESHOLD
# ============================================================

threshold = 0.4
threshold_predictions = np.where(
    churn_probabilities >= threshold,
    "Yes",
    "No"
)

print(confusion_matrix(y_test, threshold_predictions))
print(classification_report(y_test, threshold_predictions))


# ============================================================
# 19. SAVE AND RELOAD TRAINED MODEL
# ============================================================

joblib.dump(best_model, "churn_model.pkl")
loaded_model = joblib.load("churn_model.pkl")

print(loaded_model.predict(X_test[:5]))
