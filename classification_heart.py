# ============================================================================
# HEART DISEASE PREDICTION CLASSIFICATION
# ============================================================================
# This script demonstrates advanced classification with:
# 1. Support Vector Machine (SVM) with Pipeline
# 2. MinMaxScaler for feature normalization
# 3. GridSearchCV for hyperparameter optimization
# Dataset: heart.csv (Heart Disease UCI dataset)
# ============================================================================

# Import necessary libraries
import pandas as pd  # For data manipulation
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.pipeline import Pipeline  # To chain preprocessing and model
from sklearn.preprocessing import MinMaxScaler  # To normalize features to [0,1] range
from sklearn.model_selection import train_test_split, GridSearchCV  # For data splitting and optimization

# ============================================================================
# DATA LOADING AND EXPLORATION
# ============================================================================

# Load the heart disease dataset
heart = pd.read_csv("heart.csv")

print("=" * 70)
print("HEART DISEASE DATASET")
print("=" * 70)
print("First 5 rows of the dataset:")
print(heart.head())
print(f"\nDataset Shape: {heart.shape}")
print(f"Number of Features: {heart.shape[1] - 1}")  # Minus 1 for target column
print(f"Number of Samples: {heart.shape[0]}")
print(f"\nColumn Names: {list(heart.columns)}")
print(f"\nTarget Variable (target) Values: {heart['target'].unique()}")
print("  0 = No Heart Disease, 1 = Heart Disease Present")
print("\n")

# Check for missing values
print("Missing Values:")
print(heart.isnull().sum())
print("\n")

# Display basic statistics
print("Dataset Statistics:")
print(heart.describe())
print("\n")

# ============================================================================
# DATA PREPARATION
# ============================================================================

print("=" * 70)
print("DATA PREPARATION")
print("=" * 70)

# Separate features (X) and target variable (Y)
X = heart.drop("target", axis=1)  # All columns except 'target'
Y = heart["target"]               # Only the 'target' column

print(f"Features (X) shape: {X.shape}")
print(f"Target (Y) shape: {Y.shape}")
print(f"\nTarget distribution:")
print(Y.value_counts())
print(f"  Class 0 (No Disease): {Y.value_counts()[0]} samples")
print(f"  Class 1 (Has Disease): {Y.value_counts()[1]} samples")
print("\n")

# Split data into training (80%) and testing (20%) sets
# Note: No random_state specified, so split will be different each run
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print("\n")

# ============================================================================
# MODEL 1: SVM WITH PIPELINE (BASELINE)
# ============================================================================

print("=" * 70)
print("SUPPORT VECTOR MACHINE (SVM) WITH PIPELINE - BASELINE")
print("=" * 70)

# Create a Pipeline combining preprocessing and classification
# Why Pipeline?
# - MinMaxScaler normalizes features to [0,1] range
# - SVM performs better with normalized features
# - Pipeline prevents data leakage by fitting scaler only on training data
hybrid_baseline = Pipeline(steps=[
    ("scaler", MinMaxScaler()),  # Step 1: Scale features to [0,1]
    ("algorithm", SVC())         # Step 2: Apply SVM classifier
])

print("Pipeline Steps:")
print("  1. MinMaxScaler: Normalizes all features to [0, 1] range")
print("  2. SVC: Support Vector Classifier with default parameters")
print("\nTraining the model...")

# Fit the entire pipeline on training data
hybrid_baseline.fit(X_train, y_train)

# Calculate accuracy on test data
accuracy_baseline = hybrid_baseline.score(X_test, y_test)

print(f"\nSVM (Baseline) Accuracy: {accuracy_baseline:.4f}")
print(f"SVM (Baseline) Accuracy: {accuracy_baseline*100:.2f}%")
print("\n")

# ============================================================================
# MODEL 2: SVM WITH GRID SEARCH OPTIMIZATION
# ============================================================================

print("=" * 70)
print("SVM WITH GRID SEARCH HYPERPARAMETER OPTIMIZATION")
print("=" * 70)

# Create a base SVM model (parameters will be optimized)
model_svm = SVC()

# Define parameter grid for hyperparameter tuning
# C is the regularization parameter:
# - Small C: more regularization (simpler model, may underfit)
# - Large C: less regularization (complex model, may overfit)
parameters = {
    "C": [0.01, 0.5, 0.7, 0.08, 0.12, 0.8]  # Different regularization strengths to try
}

print("Hyperparameter Grid:")
print(f"  C (Regularization Parameter): {parameters['C']}")
print(f"  Total parameter combinations to test: {len(parameters['C'])}")
print("\nWhat is C parameter?")
print("  - Controls the trade-off between smooth decision boundary and")
print("    classifying training points correctly")
print("  - Smaller C: Wider margin, more training errors allowed")
print("  - Larger C: Narrower margin, fewer training errors allowed")
print("\nStarting Grid Search with 5-Fold Cross-Validation...")

# Create GridSearchCV object
# estimator: the model to optimize (SVM)
# param_grid: parameter values to try
# scoring: metric to optimize (accuracy)
# cv=5: use 5-fold cross-validation for robust evaluation
hybrid_optimized = GridSearchCV(
    estimator=model_svm,
    param_grid=parameters,
    scoring="accuracy",
    cv=5,  # 5-fold cross-validation
    verbose=1  # Show progress
)

# Fit GridSearchCV on training data
# This will:
# 1. Try each C value
# 2. For each C, perform 5-fold cross-validation
# 3. Select the C with best average accuracy
hybrid_optimized.fit(X_train, y_train)

# Get accuracy with best parameters
accuracy_optimized = hybrid_optimized.score(X_test, y_test)

# Get the best parameters found
best_params = hybrid_optimized.best_params_

print("\nGrid Search Complete!")
print(f"\nSVM (Optimized) Accuracy: {accuracy_optimized:.4f}")
print(f"SVM (Optimized) Accuracy: {accuracy_optimized*100:.2f}%")
print(f"\nBest Parameter Found:")
print(f"  C = {best_params['C']}")
print("\n")

# ============================================================================
# DETAILED GRID SEARCH RESULTS
# ============================================================================

print("=" * 70)
print("DETAILED GRID SEARCH RESULTS")
print("=" * 70)

# Convert results to DataFrame
results_df = pd.DataFrame(hybrid_optimized.cv_results_)

# Display all tested parameters with their scores
print("\nAll Tested C Values and Their Performance:")
print("-" * 70)
for i, c_value in enumerate(parameters['C']):
    mean_score = results_df.loc[i, 'mean_test_score']
    std_score = results_df.loc[i, 'std_test_score']
    print(f"C = {c_value:5.2f}  |  Mean CV Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
print("-" * 70)
print(f"Best: C = {best_params['C']:5.2f}  |  Test Accuracy: {accuracy_optimized:.4f}")
print("\n")

# ============================================================================
# MODEL COMPARISON AND IMPROVEMENT ANALYSIS
# ============================================================================

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)
print(f"SVM (Baseline) Accuracy:   {accuracy_baseline:.4f} ({accuracy_baseline*100:.2f}%)")
print(f"SVM (Optimized) Accuracy:  {accuracy_optimized:.4f} ({accuracy_optimized*100:.2f}%)")

# Calculate improvement
improvement = (accuracy_optimized - accuracy_baseline) * 100
if improvement > 0:
    print(f"\nImprovement: +{improvement:.2f}% (Better)")
elif improvement < 0:
    print(f"\nChange: {improvement:.2f}% (Slightly worse)")
else:
    print(f"\nNo change in performance")

print("\n")

# ============================================================================
# CROSS-VALIDATION INSIGHTS
# ============================================================================

print("=" * 70)
print("CROSS-VALIDATION INSIGHTS")
print("=" * 70)
print(f"Best Cross-Validation Score: {hybrid_optimized.best_score_:.4f}")
print(f"Test Set Score: {accuracy_optimized:.4f}")

# Check for overfitting
cv_test_diff = hybrid_optimized.best_score_ - accuracy_optimized
if abs(cv_test_diff) < 0.05:
    print("\nModel Generalization: GOOD")
    print("  CV and Test scores are similar (difference < 5%)")
elif cv_test_diff > 0.05:
    print("\nModel Generalization: POSSIBLE OVERFITTING")
    print(f"  CV score is {cv_test_diff*100:.2f}% higher than test score")
else:
    print("\nModel Generalization: EXCELLENT")
    print(f"  Test score is actually {abs(cv_test_diff)*100:.2f}% higher than CV score")

print("\n")

# ============================================================================
# KEY INSIGHTS AND RECOMMENDATIONS
# ============================================================================

print("=" * 70)
print("KEY INSIGHTS AND RECOMMENDATIONS")
print("=" * 70)
print("\n1. Why Use Pipeline?")
print("   ✓ Prevents data leakage (scaler fits only on training data)")
print("   ✓ Simplifies code (preprocessing + model in one object)")
print("   ✓ Easier deployment (single object to save/load)")
print("   ✓ Ensures same preprocessing for training and testing")

print("\n2. Why Use MinMaxScaler with SVM?")
print("   ✓ SVM is sensitive to feature scales")
print("   ✓ Features with larger ranges can dominate the model")
print("   ✓ Scaling improves model convergence and performance")

print("\n3. Why Use GridSearchCV?")
print("   ✓ Automatically finds best hyperparameters")
print("   ✓ Uses cross-validation for robust evaluation")
print("   ✓ Prevents overfitting to single train/test split")
print("   ✓ Saves time compared to manual tuning")

print("\n4. Interpreting C Parameter:")
print(f"   Optimal C = {best_params['C']}")
if best_params['C'] < 0.1:
    print("   → Small C: Model favors simpler decision boundary")
    print("   → More regularization, may underfit complex patterns")
elif best_params['C'] > 0.5:
    print("   → Large C: Model favors fitting training data closely")
    print("   → Less regularization, may overfit to training data")
else:
    print("   → Medium C: Balanced regularization")
    print("   → Good trade-off between fitting and generalization")

print("\n5. Final Model Performance:")
print(f"   Accuracy: {accuracy_optimized*100:.2f}%")
if accuracy_optimized >= 0.85:
    print("   → EXCELLENT performance for heart disease prediction")
elif accuracy_optimized >= 0.75:
    print("   → GOOD performance, suitable for preliminary screening")
elif accuracy_optimized >= 0.65:
    print("   → MODERATE performance, may need feature engineering")
else:
    print("   → NEEDS IMPROVEMENT, consider different approach")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
