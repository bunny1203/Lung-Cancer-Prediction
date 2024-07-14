# Lung-Cancer-Prediction
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# For ignoring warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('../input/lung-cancer/survey lung cancer.csv')

# Display the dataframe and its shape
print(df)
print(df.shape)

# Checking for and removing duplicates
if df.duplicated().sum() > 0:
    df = df.drop_duplicates()

# Checking for null values
print(df.isnull().sum())

# Display dataframe information and statistics
print(df.info())
print(df.describe())

# Label encoding categorical variables
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for column in ['GENDER', 'LUNG_CANCER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']:
    df[column] = le.fit_transform(df[column])

# Check dataframe after encoding
print(df)
print(df.info())

# Check the distribution of the target variable
sns.countplot(x='LUNG_CANCER', data=df)
plt.title('Target Distribution')
plt.show()
print(df['LUNG_CANCER'].value_counts())

# Function for plotting feature distribution
def plot(col, df=df):
    df.groupby(col)['LUNG_CANCER'].value_counts(normalize=True).unstack().plot(kind='bar', figsize=(8, 5))
    plt.title(f'Distribution of {col}')
    plt.show()

for col in ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']:
    plot(col)

# Dropping less relevant columns
df_new = df.drop(columns=['GENDER', 'AGE', 'SMOKING', 'SHORTNESS OF BREATH'])
print(df_new)

# Finding and plotting correlation
correlation_matrix = df_new.corr()
print(correlation_matrix)
plt.figure(figsize=(18, 18))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, square=True)
plt.show()

# High correlation subset
high_correlation = correlation_matrix[correlation_matrix >= 0.40]
plt.figure(figsize=(12, 8))
sns.heatmap(high_correlation, cmap="Blues")
plt.show()

# Creating a new feature
df_new['ANXYELFIN'] = df_new['ANXIETY'] * df_new['YELLOW_FINGERS']
print(df_new)

# Splitting independent and dependent variables
X = df_new.drop('LUNG_CANCER', axis=1)
y = df_new['LUNG_CANCER']

# Handling imbalance using ADASYN
from imblearn.over_sampling import ADASYN
adasyn = ADASYN(random_state=42)
X, y = adasyn.fit_resample(X, y)
print(len(X))

# Splitting data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Define a function to train and evaluate models
from sklearn.metrics import classification_report, accuracy_score

def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return y_pred

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(random_state=0)
print("Logistic Regression")
train_evaluate_model(lr_model, X_train, y_train, X_test, y_test)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=0)
print("Decision Tree")
train_evaluate_model(dt_model, X_train, y_train, X_test, y_test)

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
print("K-Nearest Neighbors")
train_evaluate_model(knn_model, X_train, y_train, X_test, y_test)

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()
print("Gaussian Naive Bayes")
train_evaluate_model(gnb_model, X_train, y_train, X_test, y_test)

# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
mnb_model = MultinomialNB()
print("Multinomial Naive Bayes")
train_evaluate_model(mnb_model, X_train, y_train, X_test, y_test)

# Support Vector Classifier
from sklearn.svm import SVC
svc_model = SVC()
print("Support Vector Classifier")
train_evaluate_model(svc_model, X_train, y_train, X_test, y_test)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier()
print("Random Forest")
train_evaluate_model(rf_model, X_train, y_train, X_test, y_test)

# XGBoost
from xgboost import XGBClassifier
xgb_model = XGBClassifier()
print("XGBoost")
train_evaluate_model(xgb_model, X_train, y_train, X_test, y_test)

# Multi-layer Perceptron
from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier()
print("Multi-layer Perceptron")
train_evaluate_model(mlp_model, X_train, y_train, X_test, y_test)

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier()
print("Gradient Boosting")
train_evaluate_model(gb_model, X_train, y_train, X_test, y_test)

# K-Fold Cross Validation
from sklearn.model_selection import KFold, cross_val_score

k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)
models = [lr_model, dt_model, knn_model, gnb_model, mnb_model, svc_model, rf_model, xgb_model, mlp_model, gb_model]
model_names = ["Logistic Regression", "Decision Tree", "KNN", "Gaussian Naive Bayes", "Multinomial Naive Bayes", "Support Vector Classifier", "Random Forest", "XGBoost", "Multi-layer Perceptron", "Gradient Boosting"]

for model, name in zip(models, model_names):
    scores = cross_val_score(model, X, y, cv=kf)
    print(f"{name} models' average accuracy: {np.mean(scores)}")

# Stratified K-Fold Cross Validation
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=k)
for model, name in zip(models, model_names):
    scores = cross_val_score(model, X, y, cv=kf)
    print(f"{name} models' average accuracy: {np.mean(scores)}")
