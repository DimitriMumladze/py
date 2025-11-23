# ============================================================================
# CAR EVALUATION CLASSIFICATION
# ============================================================================
# This script demonstrates three classification algorithms:
# 1. Decision Tree Classifier with Confusion Matrix
# 2. Support Vector Machine (SVM) with Confusion Matrix
# 3. Logistic Regression with ROC Curve
# Dataset: car_evaluation.csv
# ============================================================================

# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.preprocessing import LabelEncoder  # To convert categorical text data to numerical
from sklearn.tree import DecisionTreeClassifier  # Decision tree algorithm for classification
from sklearn.svm import SVC  # Support Vector Classifier algorithm
from sklearn.linear_model import LogisticRegression  # Logistic regression algorithm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score  # Evaluation metrics
import matplotlib.pyplot as plt  # For creating visualizations
from sklearn.model_selection import train_test_split  # To split data into training and testing sets

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

# Read the car evaluation CSV file (no headers in original file)
car = pd.read_csv("car_evaluation.csv", header=None)

# Assign meaningful column names to the dataset
car.columns = ["Buying_Price", "Maintance_cost", "N_of_doors", "N_person", "lug_boot",
               "safety", "Class"]

# Map the target variable (Class) to binary values:
# 'vgood' (very good) = 1, all others ('unacc', 'acc', 'good') = 0
car['Class'] = car['Class'].map({'unacc': 0, 'acc': 0, 'good': 0, 'vgood': 1})

# Convert all categorical (object type) columns to numerical using LabelEncoder
for col in car.columns:
    if car[col].dtype == 'object':  # Check if column contains text data
        car[col] = LabelEncoder().fit_transform(car[col])  # Convert text to numbers

# Display the first 5 rows to verify preprocessing
print("Preprocessed Data:")
print(car.head())
print("\n")

# Separate features (X) and target variable (y)
y = car['Class']  # Target variable (what we want to predict)
X = car.drop('Class', axis=1)  # Features (all columns except Class)

# Split data into training (80%) and testing (20%) sets
# random_state=1 ensures reproducibility (same split every time)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# ============================================================================
# MODEL 1: DECISION TREE CLASSIFIER
# ============================================================================

print("=" * 70)
print("DECISION TREE CLASSIFIER")
print("=" * 70)

# Create a Decision Tree model with default parameters
model_dt = DecisionTreeClassifier()

# Train the model using training data
model_dt.fit(X_train, y_train)

# Calculate and display accuracy on test data
accuracy_dt = model_dt.score(X_test, y_test)
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")

# Make predictions on test data
y_prediction_dt = model_dt.predict(X_test)

# Create confusion matrix to see prediction performance
conf_matrix_dt = confusion_matrix(y_test, y_prediction_dt)

# Create a visual display of the confusion matrix
result_dt = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_dt, display_labels=model_dt.classes_)
result_dt.plot()  # Plot the confusion matrix
plt.title("Decision Tree - Confusion Matrix")  # Add title
plt.show()  # Display the plot

print("\n")

# ============================================================================
# MODEL 2: SUPPORT VECTOR MACHINE (SVM)
# ============================================================================

print("=" * 70)
print("SUPPORT VECTOR MACHINE (SVM)")
print("=" * 70)

# Create an SVM model with probability estimation enabled
# probability=True allows us to get probability scores for predictions
model_svm = SVC(probability=True)

# Train the SVM model
model_svm.fit(X_train, y_train)

# Calculate and display accuracy
accuracy_svm = model_svm.score(X_test, y_test)
print(f"SVM Accuracy: {accuracy_svm:.4f}")

# Make predictions on test data
y_prediction_svm = model_svm.predict(X_test)

# Create confusion matrix
conf_matrix_svm = confusion_matrix(y_test, y_prediction_svm)

# Display the confusion matrix visually
result_svm = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_svm, display_labels=model_svm.classes_)
result_svm.plot()
plt.title("SVM - Confusion Matrix")
plt.show()

print("\n")

# ============================================================================
# MODEL 3: LOGISTIC REGRESSION WITH ROC CURVE
# ============================================================================

print("=" * 70)
print("LOGISTIC REGRESSION WITH ROC CURVE")
print("=" * 70)

# Create a Logistic Regression model
model_lr = LogisticRegression()

# Train the model
model_lr.fit(X_train, y_train)

# Get probability predictions for the positive class (class 1)
# [:,1] selects only probabilities for class 1
positive_probabilities = model_lr.predict_proba(X_test)[:, 1]

# Calculate ROC curve values
# fpr = False Positive Rate, tpr = True Positive Rate, threshold = decision thresholds
fpr, tpr, threshold = roc_curve(y_test, positive_probabilities)

# Calculate Area Under ROC Curve (AUC score)
# AUC ranges from 0 to 1, where 1 is perfect classification
area = roc_auc_score(y_test, positive_probabilities)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {area:.4f})')  # Plot with AUC in label
plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')  # Diagonal line for reference
plt.xlabel('False Positive Rate')  # X-axis label
plt.ylabel('True Positive Rate')  # Y-axis label
plt.title('Logistic Regression - ROC Curve')  # Title
plt.legend()  # Show legend
plt.grid(True)  # Add grid for better readability
plt.show()  # Display plot

print(f"Logistic Regression AUC Score: {area:.4f}")
print("\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 70)
print("SUMMARY OF RESULTS")
print("=" * 70)
print(f"Decision Tree Accuracy: {accuracy_dt:.4f}")
print(f"SVM Accuracy: {accuracy_svm:.4f}")
print(f"Logistic Regression AUC: {area:.4f}")
print("=" * 70)
