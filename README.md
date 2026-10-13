# Telco Customer Churn Prediction

## 📌 Project Overview

Customer churn occurs when customers stop using a company's services. Predicting churn helps businesses identify customers who may leave and take proactive steps to improve customer retention.

This project uses Machine Learning to predict whether a telecommunications customer is likely to churn based on their account details, subscribed services, contract type, payment method, and billing information.

The project includes data analysis, data preprocessing, model training, hyperparameter tuning, evaluation, and a Streamlit web application for making predictions.

## 🎯 Objectives

- Analyze customer data to identify patterns associated with churn.
- Perform data cleaning and exploratory data analysis (EDA).
- Prepare numerical and categorical features for Machine Learning.
- Train a Logistic Regression classification model.
- Tune model hyperparameters using GridSearchCV.
- Evaluate the model with a focus on churn recall.
- Adjust the prediction threshold according to the business objective.
- Build a simple interactive web application using Streamlit.

## 📂 Project Structure

```text
Telco-Customer-Churn/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── telco-churn.py
├── app.py
├── churn_model.pkl
├── requirements.txt
└── README.md
```

## 📊 Dataset

This project uses the IBM Telco Customer Churn dataset.

- **Rows:** 7,043 customers
- **Columns:** 21
- **Target variable:** `Churn`
- **Target classes:** `Yes` and `No`

The dataset contains information about customer demographics, services, contracts, tenure, and billing.

### Important features

| Feature | Description |
|---|---|
| `tenure` | Number of months the customer has stayed |
| `MonthlyCharges` | Monthly amount charged |
| `TotalCharges` | Total amount charged |
| `Contract` | Customer's contract type |
| `InternetService` | Type of internet service |
| `PaymentMethod` | Customer's payment method |
| `TechSupport` | Whether the customer has technical support |
| `OnlineSecurity` | Whether the customer has online security |
| `Churn` | Whether the customer left the service |

## 🧹 Data Cleaning

The following preprocessing steps were performed:

- Checked for missing values and duplicate rows.
- Converted `TotalCharges` to a numeric data type.
- Handled 11 missing `TotalCharges` values by replacing them with `0`.
- Checked for duplicate customer records.
- Removed `customerID` from the model features because it is an identifier rather than a useful predictive feature.
- Separated the input features from the target variable.

## 🔍 Exploratory Data Analysis (EDA)

The analysis examined customer churn across several important features.

### Key findings

- Month-to-month contract customers had a higher churn rate than customers with longer contracts.
- Customers with shorter tenure tended to have higher churn.
- Customers who churned had higher average monthly charges than customers who stayed.
- Fiber optic internet customers had a higher churn rate than DSL customers in this dataset.
- Customers without online security or technical support had higher churn rates.
- Electronic check users had a higher observed churn rate than users of the other payment methods examined.

These findings show associations in the dataset and do not establish that any individual feature causes churn.

## ⚙️ Machine Learning Workflow

The project follows this workflow:

1. Load the dataset.
2. Clean and explore the data.
3. Separate features (`X`) and target (`y`).
4. Split the data into training and testing sets.
5. Scale numerical features using `StandardScaler`.
6. Encode categorical features using `OneHotEncoder`.
7. Train a Logistic Regression baseline model.
8. Build a preprocessing and classification pipeline.
9. Tune hyperparameters using `GridSearchCV` with five-fold cross-validation.
10. Evaluate the tuned model.
11. Experiment with different classification thresholds.
12. Save the trained model using Joblib.
13. Use the saved model in a Streamlit application.

### Model and hyperparameter tuning

The classification algorithm used is **Logistic Regression**.

The hyperparameters explored include:

- `C`: Controls the strength of regularization.
- `class_weight`: Tests the default class weighting and balanced class weighting.

The model selection process uses **recall for the `Yes` class** as its cross-validation scoring metric, emphasizing the identification of customers who may churn.

The best parameters found in the recorded experiment were:

```python
{
    "classifier__C": 1,
    "classifier__class_weight": "balanced"
}
```

## 📈 Model Evaluation

The tuned model was evaluated on the held-out test set.

At the default threshold of `0.5`, the recorded results were:

| Metric | Score |
|---|---:|
| Accuracy | 73.81% |
| Churn precision | 0.50 |
| Churn recall | 0.78 |
| Churn F1-score | 0.61 |

The confusion matrix was:

```text
[[747, 288],
 [ 81, 293]]
```

The model identified 293 customers who churned, while 81 actual churn cases were missed at this threshold.

*These scores describe this particular train/test split and may vary if the data or experimental setup changes.*

## 🎚️ Decision Threshold Tuning

Instead of relying only on the default probability threshold of `0.5`, the project experiments with lower thresholds.

A threshold of `0.4` was selected provisionally for the application.

At this threshold, the recorded test results were:

| Metric | Score |
|---|---:|
| Accuracy | 70% |
| Churn precision | 0.47 |
| Churn recall | 0.87 |

The threshold adjustment increases churn recall but also produces more false-positive predictions.

The appropriate threshold depends on the business costs of missed churners, unnecessary retention offers, and the available retention budget.

## 💾 Model Saving

The trained pipeline is saved using Joblib:

```python
joblib.dump(best_model, "churn_model.pkl")
```

The saved pipeline includes the preprocessing steps and the trained classifier, allowing new customer records to be processed consistently.

## 🌐 Streamlit Web Application

The project includes an interactive web application built with Streamlit.

Users can enter customer information and receive a churn prediction.

### Features

- Customer information input form
- Churn probability display
- Prediction based on the selected threshold
- Simple visual feedback for the prediction

The application loads the saved model from `churn_model.pkl` and uses the same preprocessing pipeline as the trained model.

## 🚀 Installation and Usage

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Open the project folder

```bash
cd Telco-Customer-Churn
```

### 3. Create and activate a virtual environment (optional)

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the training script

```bash
python telco-churn.py
```

This script performs the data analysis and model training steps and saves the trained model.

Ensure that the dataset exists at `data/Telco-Customer-Churn.csv` relative to the project folder.

### 6. Launch the Streamlit application

```bash
streamlit run app.py
```

The terminal will display a local URL where you can open the application in your browser.

If you already have a trained `churn_model.pkl` file, you can launch the app without retraining, provided the model file and dependencies are available.

## 🛠️ Technologies Used

- **Python** — programming language
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib** — data visualization
- **Scikit-learn** — preprocessing, model training, cross-validation, and evaluation
- **Joblib** — model serialization
- **Streamlit** — interactive web application

## ⚠️ Limitations

- The model is trained on a particular dataset and may not generalize to every telecommunications business.
- Model predictions are estimates, not guarantees of customer behavior.
- Threshold selection involves a trade-off between recall and precision.
- The project does not yet calculate the financial impact of retention campaigns.
- Model performance should be monitored and reevaluated when new customer data becomes available.

## 🔮 Future Improvements

- Compare Logistic Regression with Random Forest, XGBoost, and other classifiers.
- Add model interpretation using feature importance or SHAP.
- Evaluate the business impact of different decision thresholds.
- Improve the web application with more informative visualizations.
- Deploy the Streamlit application online.
- Monitor performance on new customer data.

## 👨‍💻 Author

**Owaiz**

BE Computer Science and Engineering student interested in Machine Learning, Artificial Intelligence, and software development.

## 📄 Disclaimer

This project is for educational and portfolio purposes. Predictions should not be treated as guaranteed customer outcomes.
