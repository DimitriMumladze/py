# ============================================================================
# DIABETES PREDICTION CLASSIFICATION
# ============================================================================
# This script demonstrates classification with pipelines and hyperparameter tuning:
# 1. Naive Bayes with MinMaxScaler in Pipeline
# 2. Random Forest with GridSearchCV for hyperparameter optimization
# Dataset: diabetes.csv (Pima Indians Diabetes Database)
# ============================================================================

# Import necessary libraries
import pandas as pd  # For data manipulation
from sklearn.naive_bayes import GaussianNB  # Naive Bayes classifier
from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier
from sklearn.preprocessing import MinMaxScaler  # To scale features to [0,1] range
from sklearn.pipeline import Pipeline  # To chain preprocessing and model together
from sklearn.model_selection import train_test_split, GridSearchCV  # For data splitting and hyperparameter tuning

# ============================================================================
# DATA LOADING AND EXPLORATION
# ============================================================================

# Load the diabetes dataset
data = pd.read_csv("diabetes.csv")

print("=" * 70)
print("DIABETES DATASET")
print("=" * 70)
print("First 5 rows of the dataset:")
print(data.head())
print(f"\nDataset Shape: {data.shape}")
print(f"\nColumn Names: {list(data.columns)}")
print(f"\nTarget Variable (Outcome) Values: {data['Outcome'].unique()}")
print("  0 = No Diabetes, 1 = Diabetes")
print("\n")

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Select features (independent variables) for prediction
# These are medical measurements that might indicate diabetes risk
X = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].values

# Select target variable (dependent variable) - what we want to predict
y = data['Outcome'].values

# Split data into training (70%) and testing (30%) sets
# random_state=1 ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print("=" * 70)
print("DATA SPLIT")
print("=" * 70)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print("\n")

# ============================================================================
# MODEL 1: NAIVE BAYES WITH PIPELINE
# ============================================================================

print("=" * 70)
print("NAIVE BAYES CLASSIFIER WITH PIPELINE")
print("=" * 70)

# Create a Pipeline that chains preprocessing and model
# Step 1: MinMaxScaler - scales all features to range [0, 1]
# Step 2: GaussianNB - applies Naive Bayes classification
# Pipeline ensures scaler is fit only on training data, preventing data leakage
hybrid_nb = Pipeline(steps=[
    ("Scaler", MinMaxScaler()),  # Preprocessing step
    ("algo1", GaussianNB())      # Model step
])

# Fit the pipeline on training data
# This fits both the scaler AND the model
hybrid_nb.fit(X_train, y_train)

# Calculate accuracy on test data
# Pipeline automatically applies scaling before prediction
accuracy_nb = hybrid_nb.score(X_test, y_test)

print(f"Naive Bayes Accuracy: {accuracy_nb:.4f}")
print(f"Naive Bayes Accuracy: {accuracy_nb*100:.2f}%")
print("\n")

# ============================================================================
# MODEL 2: RANDOM FOREST WITHOUT TUNING (BASELINE)
# ============================================================================

print("=" * 70)
print("RANDOM FOREST CLASSIFIER (BASELINE)")
print("=" * 70)

# Create a Random Forest model with manual parameters
# n_estimators=100 means 100 decision trees in the forest
# max_depth=2 limits tree depth to prevent overfitting
model_rf_baseline = RandomForestClassifier(n_estimators=100, max_depth=2)

# Train the model
model_rf_baseline.fit(X_train, y_train)

# Calculate accuracy
accuracy_rf_baseline = model_rf_baseline.score(X_test, y_test)

print(f"Random Forest (Baseline) Accuracy: {accuracy_rf_baseline:.4f}")
print(f"Random Forest (Baseline) Accuracy: {accuracy_rf_baseline*100:.2f}%")
print("\n")

# ============================================================================
# MODEL 3: RANDOM FOREST WITH GRID SEARCH (OPTIMIZED)
# ============================================================================

print("=" * 70)
print("RANDOM FOREST WITH GRID SEARCH OPTIMIZATION")
print("=" * 70)

# Create a Random Forest model (parameters will be tuned)
model_rf = RandomForestClassifier()

# Define parameter grid to search
# GridSearchCV will try ALL combinations of these parameters
parameters = dict()
parameters['n_estimators'] = [20, 30, 10, 50]  # Number of trees to try
parameters['max_depth'] = [2, 3, 4, 5, 6]      # Maximum depth of trees to try

print("Parameter Grid:")
print(f"  n_estimators (number of trees): {parameters['n_estimators']}")
print(f"  max_depth (tree depth): {parameters['max_depth']}")
print(f"  Total combinations to test: {len(parameters['n_estimators']) * len(parameters['max_depth'])}")
print("\nSearching for best parameters...")

# Create GridSearchCV object
# estimator: the model to optimize
# param_grid: parameters to try
# scoring: metric to optimize (accuracy)
# n_jobs=-1: use all CPU cores for parallel processing
# cv=5: use 5-fold cross-validation
hybrid_rf = GridSearchCV(
    estimator=model_rf,
    param_grid=parameters,
    scoring='accuracy',
    n_jobs=-1,
    cv=5  # 5-fold cross-validation
)

# Fit GridSearchCV - this trains models with all parameter combinations
hybrid_rf.fit(X_train, y_train)

# Get accuracy with best parameters found
accuracy_rf_optimized = hybrid_rf.score(X_test, y_test)

# Get the best parameters found by GridSearch
best_params = hybrid_rf.best_params_

print("\nGrid Search Complete!")
print(f"Random Forest (Optimized) Accuracy: {accuracy_rf_optimized:.4f}")
print(f"Random Forest (Optimized) Accuracy: {accuracy_rf_optimized*100:.2f}%")
print(f"\nBest Parameters Found:")
print(f"  n_estimators (number of trees): {best_params['n_estimators']}")
print(f"  max_depth (tree depth): {best_params['max_depth']}")
print("\n")

# ============================================================================
# DETAILED GRID SEARCH RESULTS
# ============================================================================

print("=" * 70)
print("DETAILED GRID SEARCH RESULTS")
print("=" * 70)

# Convert GridSearch results to DataFrame for easy viewing
results_df = pd.DataFrame(hybrid_rf.cv_results_)

# Select and display relevant columns
print("\nTop 5 Parameter Combinations by Mean Test Score:")
print(results_df[['param_n_estimators', 'param_max_depth', 'mean_test_score', 'rank_test_score']]
      .sort_values('rank_test_score')
      .head(5)
      .to_string(index=False))
print("\n")

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("=" * 70)
print("MODEL COMPARISON SUMMARY")
print("=" * 70)
print(f"Naive Bayes Accuracy:              {accuracy_nb:.4f} ({accuracy_nb*100:.2f}%)")
print(f"Random Forest (Baseline) Accuracy: {accuracy_rf_baseline:.4f} ({accuracy_rf_baseline*100:.2f}%)")
print(f"Random Forest (Optimized) Accuracy: {accuracy_rf_optimized:.4f} ({accuracy_rf_optimized*100:.2f}%)")
print(f"\nImprovement from Baseline to Optimized: {(accuracy_rf_optimized - accuracy_rf_baseline)*100:.2f}%")
print("=" * 70)

# ============================================================================
# KEY INSIGHTS
# ============================================================================

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print("1. Pipeline Benefits:")
print("   - Prevents data leakage by fitting scaler only on training data")
print("   - Simplifies code by chaining preprocessing and modeling")
print("   - Makes deployment easier")
print("\n2. GridSearchCV Benefits:")
print("   - Automatically finds best hyperparameters")
print("   - Uses cross-validation to prevent overfitting")
print("   - Saves time compared to manual tuning")
print("\n3. Model Selection:")
if accuracy_nb > accuracy_rf_optimized:
    print("   - Naive Bayes performed best for this dataset")
    print("   - Naive Bayes is simpler and faster")
elif accuracy_rf_optimized > accuracy_nb:
    print("   - Optimized Random Forest performed best")
    print("   - Random Forest can capture complex patterns")
else:
    print("   - Both models performed equally well")
print("=" * 70)
