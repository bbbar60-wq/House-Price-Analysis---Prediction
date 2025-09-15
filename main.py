import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ========================
# Constants & Config
# ========================
NUMERIC_COLUMNS = ["Area", "Price", "Price(USD)", "Room", "Parking", "Warehouse", "Elevator"]
CATEGORY_COLUMNS = ["Address"]
TARGET_VARIABLE = "Price"
FEATURES_FOR_MODEL = ["Area", "Room", "Score"]


# ========================
# Utility Functions
# ========================

def pascal_case(name: str) -> str:
    """Convert column name from snake_case or space separated to PascalCase."""
    return "".join(word.capitalize() for word in name.replace("_", " ").split())


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Load CSV, clean data, convert types, handle missing values."""
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        print("Please make sure 'housePrice.csv' is available in the script's directory.")
        return None

    df = pd.read_csv(file_path)

    # Rename columns to PascalCase for consistency
    new_column_names = {col: pascal_case(col) for col in df.columns}
    df = df.rename(columns=new_column_names)

    # Convert numeric columns, coercing errors to NaN
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing numeric values with median imputation
    for col in NUMERIC_COLUMNS:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Handle missing categorical values with the mode (most frequent value)
    for col in CATEGORY_COLUMNS:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    return df


def basic_summary(df: pd.DataFrame):
    """Print basic information and statistics about the dataframe."""
    print("--- Basic Data Summary ---")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values Check:\n", df.isna().sum())
    print("\nDescriptive Statistics:\n", df.describe().transpose())
    print("-" * 30)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add new features like Score and PricePerArea."""
    print("\n--- Performing Feature Engineering ---")
    # Combine amenities into a Score
    for col in ["Parking", "Warehouse", "Elevator"]:
        if col not in df.columns:
            df[col] = 0
    df["Score"] = df["Parking"] + df["Warehouse"] + df["Elevator"]

    # Price per area
    df["PricePerArea"] = df["Price"] / df["Area"]
    print("New features 'Score' and 'PricePerArea' created.")
    print("-" * 30)
    return df


def handle_outliers(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Remove extreme outliers beyond 1.5 * IQR."""
    print("\n--- Handling Outliers ---")
    initial_rows = len(df)
    for col in numeric_cols:
        if col in df.columns and df[col].dtype in [np.float64, np.int64]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    print(f"Removed {initial_rows - len(df)} rows identified as outliers.")
    print("-" * 30)
    return df


# ========================
# Visualization Functions
# ========================

def plot_scatter(df: pd.DataFrame, x: str, y: str, marker="o", color="blue", title=None):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, marker=marker, color=color)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title or f"Relationship between {y} and {x}")
    plt.grid(True)
    plt.show()


def plot_histogram_kde(df: pd.DataFrame, column: str, bins=30, color="skyblue"):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], bins=bins, color=color, kde=True)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column}")
    plt.show()


def plot_pairplot(df: pd.DataFrame, columns: list):
    print("\n--- Generating Pairplot ---")
    sns.pairplot(df[columns])
    plt.suptitle("Pairwise Relationships Between Key Features", y=1.02)
    plt.show()
    print("-" * 30)


def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolors='k')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs. Predicted House Prices")
    # Add a line for perfect predictions
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--', lw=2, color='red')
    plt.grid(True)
    plt.show()


# ========================
# Aggregation / Grouping
# ========================

def aggregation_examples(df: pd.DataFrame):
    print("\n--- Aggregation Examples ---")
    if "Address" in df.columns and "Area" in df.columns:
        print("\nMean area by Address (Top 5):\n", df.groupby("Address")["Area"].mean().nlargest(5))

    if "Room" in df.columns and "Price" in df.columns:
        print("\nMean price by Number of Rooms:\n", df.groupby("Room")["Price"].mean())
        print("\nRoom vs Parking Crosstab:\n", pd.crosstab(df["Room"], df["Parking"]))
    print("-" * 30)


# ========================
# Machine Learning Model
# ========================

def train_and_evaluate_model(df: pd.DataFrame):
    """Prepare data, train a model, and evaluate its performance."""
    print("\n--- Building Predictive Model ---")

    # One-hot encode the 'Address' column
    df_encoded = pd.get_dummies(df, columns=['Address'], drop_first=True)

    # Define features (X) and target (y)
    y = df_encoded[TARGET_VARIABLE]

    # Add encoded address columns to the feature list
    address_cols = [col for col in df_encoded if col.startswith('Address_')]
    current_features = FEATURES_FOR_MODEL + address_cols

    X = df_encoded[current_features]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")

    # Train a RandomForestRegressor model
    print("Training RandomForestRegressor model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate Performance
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"R-squared (R²) Score: {r2:.4f}")
    print("An R² score closer to 1.0 indicates a better model fit.")
    print("-" * 30)

    # Visualize Predictions
    plot_predictions(y_test, y_pred)


# ========================
# Main Function
# ========================

def main():
    """Main execution block."""
    df = load_and_clean_data("housePrice.csv")
    if df is None:
        return

    basic_summary(df)
    df = feature_engineering(df)
    df = handle_outliers(df, NUMERIC_COLUMNS)
    aggregation_examples(df)

    print("\n--- Generating Visualizations ---")
    plot_scatter(df, "Area", "Price", marker=".", color="blue")
    plot_scatter(df, "Score", "Price", color="red")
    plot_histogram_kde(df, "Price")
    plot_pairplot(df, ["Area", "Room", "Price", "Score"])

    train_and_evaluate_model(df)


if __name__ == "__main__":
    main()
