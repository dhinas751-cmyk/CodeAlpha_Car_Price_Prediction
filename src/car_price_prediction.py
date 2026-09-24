"""
CodeAlpha Data Science Internship - Task 3: Car Price Prediction
================================================================

This script predicts the Selling_Price of a used vehicle from "car data.csv".

Run it from the project root folder with:

    python src/car_price_prediction.py

What it does (in order):
    1. Loads and inspects the dataset
    2. Cleans the data (duplicates, text tidy-up, sanity checks)
    3. Creates the Car_Age feature
    4. Performs EDA and saves 8 charts
    5. Splits the data into training (80%) and testing (20%) sets
    6. Builds preprocessing + model Pipelines (no data leakage)
    7. Compares 5 regression models with 5-fold cross-validation (training data only)
    8. Selects the final model using the cross-validation results ONLY
    9. Evaluates all models on the untouched test set (MAE, RMSE, R2)
   10. Creates comparison / actual-vs-predicted / residual / importance charts
   11. Makes sample predictions
   12. Writes observations and limitations, and saves outputs/results.txt

Every number in outputs/results.txt comes from actually running this script.
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "car data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
IMAGES_DIR = OUTPUT_DIR / "images"
RESULTS_PATH = OUTPUT_DIR / "results.txt"
COMPARISON_CSV_PATH = OUTPUT_DIR / "model_comparison.csv"

RANDOM_STATE = 42  # fixed so results are reproducible
TEST_SIZE = 0.2  # 80% train / 20% test
CV_FOLDS = 5

TARGET = "Selling_Price"
NUMERIC_FEATURES = ["Present_Price", "Driven_kms", "Car_Age", "Owner"]
CATEGORICAL_FEATURES = ["Fuel_Type", "Selling_type", "Transmission"]
PRICE_LABEL = "Selling Price (dataset units)"

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------
# Small helper to print AND remember text for results.txt
# --------------------------------------------------------------------------
class Report:
    """Collects the text that is printed so it can also be saved to results.txt."""

    def __init__(self):
        self.lines = []

    def add(self, text=""):
        print(text)
        self.lines.append(str(text))

    def section(self, title):
        self.add()
        self.add("=" * 70)
        self.add(title)
        self.add("=" * 70)

    def save(self, path):
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def save_figure(fig, filename):
    """Save a matplotlib figure into outputs/images/ and close it."""
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# 1. Load and inspect
# --------------------------------------------------------------------------
def load_data(path):
    """Read the CSV file, with a clear error message if something is wrong."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"ERROR: dataset not found at {path}. Keep 'car data.csv' inside the data/ folder.")
    except pd.errors.ParserError as error:
        sys.exit(f"ERROR: could not parse the CSV file: {error}")


def inspect_data(df, report):
    """Document the raw dataset exactly as it was loaded."""
    report.section("1. DATA INSPECTION (raw dataset)")
    report.add(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    report.add(f"Column names: {list(df.columns)}")
    report.add("\nData types:")
    report.add(df.dtypes.astype(str).to_string())
    report.add("\nFirst 5 rows:")
    report.add(df.head().to_string())
    report.add("\nStatistical summary (numeric columns):")
    report.add(df.describe().round(3).to_string())
    report.add("\nMissing values per column:")
    report.add(df.isnull().sum().to_string())
    report.add(f"\nTotal missing values: {int(df.isnull().sum().sum())}")
    report.add(f"Duplicate rows (all 9 columns identical): {int(df.duplicated().sum())}")

    report.add("\nUnique values of important categorical columns:")
    for column in ["Fuel_Type", "Selling_type", "Transmission", "Owner"]:
        report.add(f"  {column}: {df[column].value_counts().to_dict()}")
    report.add(f"  Car_Name: {df['Car_Name'].nunique()} unique names")

    report.add(f"\nYear range: {df['Year'].min()} to {df['Year'].max()}")

    target = df[TARGET]
    report.add(f"\nTarget ({TARGET}) distribution:")
    report.add(target.describe().round(3).to_string())
    report.add(f"  Skewness: {target.skew():.3f}")
    report.add(f"  Vehicles with {TARGET} > 20: {int((target > 20).sum())}")


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------
def clean_data(df, report):
    """Clean the dataset and document every decision."""
    report.section("2. DATA CLEANING")
    df = df.copy()
    rows_before = len(df)

    # 2a. Column names and text values: remove stray spaces
    df.columns = df.columns.str.strip()
    text_columns = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    names_before = df["Car_Name"].nunique()
    for column in text_columns:
        df[column] = df[column].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    report.add("Decision 1: Stripped extra spaces in column names and text values "
               "(e.g. 'Bajaj  ct 100' had a double space).")
    report.add(f"  Unique Car_Name values: {names_before} before -> {df['Car_Name'].nunique()} after tidy-up.")

    # 2b. Target must be numeric
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    report.add(f"Decision 2: Converted {TARGET} to numeric. Non-numeric values found: "
               f"{int(df[TARGET].isnull().sum())}.")

    # 2c. Missing values
    missing_total = int(df.isnull().sum().sum())
    if missing_total > 0:
        sys.exit(f"ERROR: {missing_total} missing values found. Add an imputation step before continuing.")
    report.add("Decision 3: No missing values exist, so no imputation was needed.")

    # 2d. Duplicates
    duplicate_mask = df.duplicated(keep="first")
    duplicated_rows = df[df.duplicated(keep=False)].sort_index()
    report.add(f"\nDuplicate rows found: {int(duplicate_mask.sum())}. The rows involved (original index shown):")
    report.add(duplicated_rows.to_string())
    df = df[~duplicate_mask].reset_index(drop=True)
    report.add(f"Decision 4: REMOVED the duplicates (kept the first copy of each). Rows: {rows_before} -> {len(df)}.")
    report.add("  Why: the copies match in ALL 9 columns, including exact mileage and price, so they are most likely")
    report.add("  accidental double entries. Keeping them would let the same car land in both the training and")
    report.add("  test sets, which would make test scores look better than they really are (a form of leakage).")

    # 2e. Suspicious / impossible value checks
    report.add("\nSanity checks for suspicious or impossible values:")
    report.add(f"  Selling_Price <= 0: {int((df[TARGET] <= 0).sum())}")
    report.add(f"  Present_Price <= 0: {int((df['Present_Price'] <= 0).sum())}")
    report.add(f"  Driven_kms <= 0: {int((df['Driven_kms'] <= 0).sum())}")
    report.add(f"  Owner values found: {sorted(df['Owner'].unique().tolist())} (no negative values)")
    report.add(f"  Selling_Price greater than Present_Price: {int((df[TARGET] > df['Present_Price']).sum())}")
    high_km = df[df["Driven_kms"] >= 300000]
    report.add(f"  Very high mileage (>= 300,000 km): {len(high_km)} row(s)")
    if len(high_km) > 0:
        report.add(high_km.to_string())
    report.add("Decision 5: No impossible values were found. The one very high-mileage row is unusual but not")
    report.add("  impossible, and there is no evidence it is an error, so it was KEPT (it is documented as a limitation).")

    return df


def count_two_wheelers(df):
    """
    Heuristic used ONLY for documentation (not as a model feature): in this file the cars have
    lowercase names (e.g. 'city', 'fortuner') while motorcycles/scooters start with a capital
    letter (e.g. 'Royal Enfield Classic 350', 'Activa 3g').
    """
    return df["Car_Name"].str[0].str.isupper()


# --------------------------------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------------------------------
def add_features(df, report):
    """Create Car_Age using a dataset-consistent reference year (not today's year)."""
    report.section("3. FEATURE ENGINEERING")
    df = df.copy()
    reference_year = int(df["Year"].max())
    df["Car_Age"] = reference_year - df["Year"]
    report.add(f"Created Car_Age = {reference_year} - Year")
    report.add(f"  Reference year = the newest Year in the dataset ({reference_year}). The listings are historical,")
    report.add("  so using 2026 would add the same constant to every car and make ages look unrealistically large;")
    report.add("  a constant shift would not help the models anyway.")
    report.add(f"  Car_Age range: {df['Car_Age'].min()} to {df['Car_Age'].max()} years")
    report.add("Year is NOT used as a model input because it carries exactly the same information as Car_Age.")
    report.add("Car_Name is NOT used as a model input: it has about 98 different values for fewer than 300 rows,")
    report.add("many appear only once or twice, and one-hot encoding would create ~98 mostly-empty columns")
    report.add("(easy overfitting). Present_Price and Fuel_Type already describe the vehicle's class and price level.")
    report.add("No other features were created, to keep the project simple.")
    return df, reference_year


# --------------------------------------------------------------------------
# 4. Exploratory data analysis
# --------------------------------------------------------------------------
def run_eda(df, report):
    """Create the 8 EDA charts and print a factual interpretation for each."""
    report.section("4. EXPLORATORY DATA ANALYSIS (charts saved in outputs/images/)")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    price = df[TARGET]

    # ---- Chart 1: target distribution ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(price, bins=30, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].axvline(price.mean(), color="red", linestyle="--", label=f"Mean = {price.mean():.2f}")
    axes[0].axvline(price.median(), color="green", linestyle="--", label=f"Median = {price.median():.2f}")
    axes[0].set_title("Distribution of Selling Price")
    axes[0].set_xlabel(PRICE_LABEL)
    axes[0].legend()
    sns.boxplot(x=price, ax=axes[1], color="#8FB8DE")
    axes[1].set_title("Selling Price Box Plot")
    axes[1].set_xlabel(PRICE_LABEL)
    save_figure(fig, "01_selling_price_distribution.png")
    report.add("\n[01] Selling price distribution")
    report.add(f"  Mean = {price.mean():.2f}, median = {price.median():.2f}, min = {price.min():.2f}, "
               f"max = {price.max():.2f}, skewness = {price.skew():.2f}.")
    report.add(f"  The mean is higher than the median and {int((price > 20).sum())} vehicles sell above 20, so the "
               "distribution is right-skewed: most listings are low-priced with a few expensive ones.")

    # ---- Chart 2: Present price vs selling price (2 panels: one very expensive car squeezes the rest) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    zoom_limit = 40
    for ax, data, title in [(axes[0], df, "All rows"),
                            (axes[1], df[df["Present_Price"] <= zoom_limit],
                             f"Zoomed: Present Price up to {zoom_limit}")]:
        sns.scatterplot(data=data, x="Present_Price", y=TARGET, hue="Fuel_Type", alpha=0.7, ax=ax)
        limit = max(data["Present_Price"].max(), data[TARGET].max())
        ax.plot([0, limit], [0, limit], color="gray", linestyle="--", label="Selling = Present price")
        ax.set_title(title)
        ax.set_xlabel("Present Price (dataset units)")
        ax.set_ylabel(PRICE_LABEL)
        ax.legend()
    hidden = int((df["Present_Price"] > zoom_limit).sum())
    fig.suptitle(f"Present Price vs Selling Price ({hidden} very expensive car hidden in the zoomed panel)")
    save_figure(fig, "02_present_price_vs_selling_price.png")
    pearson = df["Present_Price"].corr(price)
    spearman = df["Present_Price"].corr(price, method="spearman")
    report.add("\n[02] Present price vs selling price")
    report.add(f"  Pearson r = {pearson:.3f}, Spearman rho = {spearman:.3f}: a strong positive association.")
    report.add(f"  Selling price is above present price in {int((price > df['Present_Price']).sum())} rows "
               "(no listing sells for more than its present price).")
    report.add(f"  {hidden} car with Present_Price above {zoom_limit} (max {df['Present_Price'].max():.1f}) stretches the "
               "left chart, so a zoomed panel is also shown.")

    # ---- Chart 3: Car age vs selling price ----
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="Car_Age", y=TARGET, alpha=0.5, ax=ax, color="#4C72B0", label="Individual cars")
    median_by_age = df.groupby("Car_Age")[TARGET].median()
    ax.plot(median_by_age.index, median_by_age.values, color="red", marker="o", label="Median price at each age")
    ax.set_title("Car Age vs Selling Price")
    ax.set_xlabel("Car Age (years, relative to newest Year in data)")
    ax.set_ylabel(PRICE_LABEL)
    ax.legend()
    save_figure(fig, "03_car_age_vs_selling_price.png")
    age_corr = df["Car_Age"].corr(price)
    newer = price[df["Car_Age"] <= 3].median()
    older = price[df["Car_Age"] >= 8].median()
    report.add("\n[03] Car age vs selling price")
    report.add(f"  Pearson r = {age_corr:.3f} (weak negative). Median price for cars aged 0-3 years = "
               f"{newer:.2f} vs {older:.2f} for cars aged 8+ years.")
    report.add(f"  Only {int((df['Car_Age'] == 0).sum())} car has Car_Age = 0, so the first point of the median line is a single car.")
    report.add("  Older cars tend to be cheaper, but the trend is noisy because the original price of the cars differs a lot.")

    # ---- Chart 4: Driven kms vs selling price (2 panels because of one extreme value) ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(data=df, x="Driven_kms", y=TARGET, alpha=0.6, ax=axes[0], color="#4C72B0")
    axes[0].set_title("All rows")
    zoom = df[df["Driven_kms"] < 300000]
    sns.scatterplot(data=zoom, x="Driven_kms", y=TARGET, alpha=0.6, ax=axes[1], color="#55A868")
    axes[1].set_title(f"Zoomed: rows under 300,000 km ({len(df) - len(zoom)} extreme row hidden)")
    for ax in axes:
        ax.set_xlabel("Driven kms")
        ax.set_ylabel(PRICE_LABEL)
    fig.suptitle("Driven KMs vs Selling Price")
    save_figure(fig, "04_driven_kms_vs_selling_price.png")
    km_all = df["Driven_kms"].corr(price)
    km_zoom = zoom["Driven_kms"].corr(zoom[TARGET])
    report.add("\n[04] Driven kms vs selling price")
    report.add(f"  Pearson r = {km_all:.3f} with all rows and {km_zoom:.3f} without the {len(df) - len(zoom)} "
               "row above 300,000 km.")
    report.add("  The relationship is weak. One extreme mileage value distorts the full chart, so a zoomed panel is also shown.")

    # ---- Charts 5-7: price by category (box plot + individual points) ----
    def price_by_category(column, filename, title):
        counts = df[column].value_counts()
        order = counts.index.tolist()
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.boxplot(data=df, x=column, y=TARGET, order=order, hue=column, palette="Set2",
                    legend=False, showfliers=False, ax=ax)
        sns.stripplot(data=df, x=column, y=TARGET, order=order, color="black", alpha=0.35, size=3, ax=ax)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{name}\n(n={counts[name]})" for name in order])
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel(PRICE_LABEL)
        save_figure(fig, filename)
        return df.groupby(column)[TARGET].agg(["count", "median", "mean"]).round(2)

    def group_comment(stats):
        """Describe which group has the highest/lowest median, and warn about very small groups."""
        top, low = stats["median"].idxmax(), stats["median"].idxmin()
        text = (f"  Highest median price: {top} ({stats.loc[top, 'median']:.2f}); "
                f"lowest: {low} ({stats.loc[low, 'median']:.2f}).")
        small = [f"{g} (n={int(stats.loc[g, 'count'])})" for g in stats.index if stats.loc[g, "count"] < 10]
        if small:
            text += " Very small group(s), not reliable: " + ", ".join(small) + "."
        return text

    fuel_stats = price_by_category("Fuel_Type", "05_price_by_fuel_type.png", "Selling Price by Fuel Type")
    report.add("\n[05] Selling price by fuel type")
    report.add(fuel_stats.to_string())
    report.add(group_comment(fuel_stats))

    trans_stats = price_by_category("Transmission", "06_price_by_transmission.png", "Selling Price by Transmission")
    report.add("\n[06] Selling price by transmission")
    report.add(trans_stats.to_string())
    report.add(group_comment(trans_stats))

    type_stats = price_by_category("Selling_type", "07_price_by_selling_type.png", "Selling Price by Selling Type")
    report.add("\n[07] Selling price by selling type")
    report.add(type_stats.to_string())
    two_wheeler = count_two_wheelers(df)
    share = pd.crosstab(df["Selling_type"], two_wheeler.map({True: "two-wheeler", False: "car"}))
    report.add("  Vehicle mix by selling type (car vs motorcycle/scooter, judged from the name):")
    report.add(share.to_string())
    report.add(group_comment(type_stats))
    report.add("  The dataset mixes cars with motorcycles/scooters, so any gap between selling types may partly reflect "
               "vehicle type and original price rather than the selling channel. No causal conclusion is drawn.")

    # ---- Chart 8: correlation heatmap ----
    corr_columns = [TARGET, "Present_Price", "Driven_kms", "Car_Age", "Owner"]
    corr = df[corr_columns].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Correlation Heatmap (numerical variables)")
    save_figure(fig, "08_correlation_heatmap.png")
    report.add("\n[08] Correlation heatmap (Pearson)")
    report.add(corr[TARGET].drop(TARGET).round(3).to_string())
    report.add("  Present_Price has the strongest linear correlation with Selling_Price. Correlation shows association only, "
               "not cause and effect, and Pearson r is sensitive to extreme values.")
    report.add("  (Year is not shown separately: it is exactly the opposite of Car_Age.)")


# --------------------------------------------------------------------------
# 5-6. Preprocessing and models
# --------------------------------------------------------------------------
def build_preprocessor():
    """
    Scale numeric columns and one-hot encode categorical columns.
    Inside a Pipeline this is fitted ONLY on the data the pipeline is trained on,
    so no information from the test set (or from a validation fold) leaks in.
    """
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def build_models():
    """Return the 5 regression models, each wrapped in a Pipeline with its own preprocessor."""
    estimators = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "KNN": KNeighborsRegressor(n_neighbors=5),
    }
    return {name: Pipeline([("preprocessor", build_preprocessor()), ("model", est)])
            for name, est in estimators.items()}


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# --------------------------------------------------------------------------
# 7. Cross-validation (training data only)
# --------------------------------------------------------------------------
def run_cross_validation(models, X_train, y_train, report):
    report.section("7. CROSS-VALIDATION (5-fold, training data only)")
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error", "neg_mae": "neg_mean_absolute_error"}
    rows = []
    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X_train, y_train, cv=kfold, scoring=scoring)
        rows.append({
            "Model": name,
            "CV_R2_mean": scores["test_r2"].mean(),
            "CV_R2_std": scores["test_r2"].std(),
            "CV_RMSE_mean": -scores["test_neg_rmse"].mean(),
            "CV_RMSE_std": scores["test_neg_rmse"].std(),
            "CV_MAE_mean": -scores["test_neg_mae"].mean(),
        })
    cv_table = pd.DataFrame(rows).set_index("Model")
    report.add("Each pipeline is re-fitted inside every fold, so the preprocessing is learned only from that fold's "
               "training part. The test set is NOT used here.")
    report.add(cv_table.round(4).to_string())
    return cv_table


# --------------------------------------------------------------------------
# 8. Test-set evaluation
# --------------------------------------------------------------------------
def evaluate_on_test(models, X_train, y_train, X_test, y_test, report):
    report.section("8. TEST-SET EVALUATION (all models fitted on the training set only)")
    rows = []
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        test_pred = pipeline.predict(X_test)
        rows.append({
            "Model": name,
            "Test_MAE": mean_absolute_error(y_test, test_pred),
            "Test_RMSE": rmse(y_test, test_pred),
            "Test_R2": r2_score(y_test, test_pred),
            "Train_R2": r2_score(y_train, pipeline.predict(X_train)),
        })
    test_table = pd.DataFrame(rows).set_index("Model")
    report.add(test_table.round(4).to_string())
    report.add("\nMetric guide: MAE = average size of the error (lower is better). RMSE = like MAE but punishes big "
               "mistakes more (lower is better). R2 = share of price variation the model explains (higher is better, max 1).")
    report.add("Train_R2 is shown only to spot overfitting: a much higher train R2 than test R2 means the model memorised "
               "the training data.")
    return test_table


# --------------------------------------------------------------------------
# Charts 9-12
# --------------------------------------------------------------------------
def plot_model_comparison(cv_table, test_table):
    names = cv_table.index.tolist()
    x = np.arange(len(names))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    panels = [
        (axes[0], "CV_R2_mean", "CV_R2_std", "Test_R2", "R2 Score (higher is better)", "R2"),
        (axes[1], "CV_RMSE_mean", "CV_RMSE_std", "Test_RMSE", "RMSE (lower is better)", "RMSE (dataset price units)"),
    ]
    for ax, cv_col, cv_std_col, test_col, title, ylabel in panels:
        cv_bars = ax.bar(x - width / 2, cv_table[cv_col], width, yerr=cv_table[cv_std_col], capsize=4,
                         label="Cross-validation mean (+/- std)", color="#4C72B0")
        test_bars = ax.bar(x + width / 2, test_table[test_col], width, label="Test set", color="#DD8452")
        ax.bar_label(cv_bars, fmt="%.2f", padding=3, fontsize=8)
        ax.bar_label(test_bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    # Bars always start at zero (R2 only goes below zero if a model is worse than guessing the mean)
    lowest = min(0.0, cv_table["CV_R2_mean"].min(), test_table["Test_R2"].min())
    axes[0].set_ylim(lowest - 0.05 if lowest < 0 else 0, 1.3)
    axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.2)  # extra headroom so the legend does not hide bars
    for ax in axes:
        ax.legend(loc="upper center", ncol=2, fontsize=8)
    fig.suptitle("Model Comparison")
    save_figure(fig, "09_model_comparison.png")


def plot_actual_vs_predicted(y_test, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.scatter(y_test, y_pred, alpha=0.7, color="#4C72B0", edgecolor="white")
    low = min(y_test.min(), y_pred.min())
    high = max(y_test.max(), y_pred.max())
    ax.plot([low, high], [low, high], color="red", linestyle="--", label="Perfect prediction (y = x)")
    ax.set_xlabel("Actual " + PRICE_LABEL)
    ax.set_ylabel("Predicted " + PRICE_LABEL)
    ax.set_title(f"Actual vs Predicted (Test Set) - {model_name}\nR2 = {r2_score(y_test, y_pred):.3f}")
    ax.legend()
    save_figure(fig, "10_actual_vs_predicted.png")


def plot_residuals(y_test, y_pred, model_name):
    residuals = y_test - y_pred  # actual minus predicted
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.7, color="#55A868", edgecolor="white")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted " + PRICE_LABEL)
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs Predicted")
    sns.histplot(residuals, bins=15, kde=True, ax=axes[1], color="#8172B3")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual (actual - predicted)")
    axes[1].set_title("Distribution of Residuals")
    fig.suptitle(f"Residual Analysis (Test Set) - {model_name}")
    save_figure(fig, "11_residual_analysis.png")
    return residuals


def get_feature_names(pipeline):
    """Map the transformed columns back to readable names (e.g. 'Fuel_Type_Diesel')."""
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    return [name.split("__", 1)[1] for name in names]


def plot_feature_importance(pipeline, X_test, y_test, model_name, report):
    """Model-based importance (if supported) plus permutation importance on the test set."""
    model = pipeline.named_steps["model"]
    perm = permutation_importance(pipeline, X_test, y_test, n_repeats=30,
                                  random_state=RANDOM_STATE, scoring="r2")
    perm_df = pd.DataFrame({"Feature": X_test.columns, "Importance": perm.importances_mean,
                            "Std": perm.importances_std}).sort_values("Importance")

    has_native = hasattr(model, "feature_importances_")
    n_panels = 2 if has_native else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.5))
    axes = np.atleast_1d(axes)

    native_df = None
    if has_native:
        native_df = pd.DataFrame({"Feature": get_feature_names(pipeline),
                                  "Importance": model.feature_importances_}).sort_values("Importance")
        axes[0].barh(native_df["Feature"], native_df["Importance"], color="#4C72B0")
        axes[0].set_title(f"Built-in importance ({model_name})\nafter one-hot encoding")
        axes[0].set_xlabel("Importance (sums to 1)")
    axes[-1].barh(perm_df["Feature"], perm_df["Importance"], xerr=perm_df["Std"], color="#DD8452")
    axes[-1].set_title("Permutation importance on test set\n(drop in R2 when a column is shuffled)")
    axes[-1].set_xlabel("Mean drop in R2")
    fig.suptitle("Feature Importance (shows what the model relies on, NOT what causes price)")
    save_figure(fig, "12_feature_importance.png")

    report.add("\nFeature importance (final model)")
    if native_df is not None:
        report.add("  Built-in importance (encoded feature names):")
        report.add(native_df.sort_values("Importance", ascending=False).round(4).to_string(index=False))
    report.add("  Permutation importance on the test set (original columns, 30 repeats):")
    report.add(perm_df.sort_values("Importance", ascending=False).round(4).to_string(index=False))
    return native_df, perm_df


# --------------------------------------------------------------------------
# 11. Sample predictions
# --------------------------------------------------------------------------
def make_sample_predictions(pipeline, df, reference_year, report):
    """Predict prices for realistic example cars built from the dataset's own ranges."""
    report.section("11. SAMPLE PREDICTIONS (model output, not a guaranteed market price)")
    samples = {
        "Sample 1 - typical car (dataset medians / most common categories)": {
            "Present_Price": float(df["Present_Price"].median()),
            "Driven_kms": int(df["Driven_kms"].median()),
            "Car_Age": int(df["Car_Age"].median()),
            "Owner": 0,
            "Fuel_Type": df["Fuel_Type"].mode()[0],
            "Selling_type": df["Selling_type"].mode()[0],
            "Transmission": df["Transmission"].mode()[0],
        },
        "Sample 2 - newer premium diesel automatic": {
            "Present_Price": 20.0,
            "Driven_kms": 40000,
            "Car_Age": 2,
            "Owner": 0,
            "Fuel_Type": "Diesel",
            "Selling_type": "Dealer",
            "Transmission": "Automatic",
        },
    }
    for name, values in samples.items():
        # Make sure the example is inside the range seen in the real data
        for column in NUMERIC_FEATURES:
            assert df[column].min() <= values[column] <= df[column].max(), f"{column} outside dataset range"
        for column in CATEGORICAL_FEATURES:
            assert values[column] in set(df[column]), f"{column} value not in dataset"
        prediction = float(pipeline.predict(pd.DataFrame([values]))[0])
        report.add(f"\n{name}")
        report.add("  Input:")
        for column, value in values.items():
            extra = f" (Year {reference_year - value})" if column == "Car_Age" else ""
            report.add(f"    {column}: {value}{extra}")
        report.add(f"  Predicted Selling_Price (model prediction): {prediction:.2f} dataset price units")
    report.add("\nNote: these are model predictions learned from a small historical dataset, not guaranteed market prices.")


# --------------------------------------------------------------------------
# 9. Final model selection and error analysis
# --------------------------------------------------------------------------
def choose_final_model(cv_table):
    """Pick the final model from the CROSS-VALIDATION table only (never from the test set)."""
    ranked = cv_table.sort_values(["CV_R2_mean", "CV_RMSE_mean"], ascending=[False, True])
    return ranked.index[0]


def report_model_selection(final_name, cv_table, test_table, n_test, report):
    report.section("9. FINAL MODEL SELECTION")
    report.add(f"Selected final model: {final_name}")
    report.add("Selection rule (fixed in the code before any test result was seen): highest mean cross-validation R2 "
               "on the TRAINING data, with lower CV RMSE as the tie-breaker.")
    report.add(f"  Its CV R2 = {cv_table.loc[final_name, 'CV_R2_mean']:.4f} (std {cv_table.loc[final_name, 'CV_R2_std']:.4f}), "
               f"CV RMSE = {cv_table.loc[final_name, 'CV_RMSE_mean']:.4f}.")
    cv_spread = cv_table["CV_R2_mean"].max() - cv_table["CV_R2_mean"].min()
    best_std = cv_table.loc[final_name, "CV_R2_std"]
    report.add(f"  Honest caveat: mean CV R2 of all five models lies within {cv_spread:.4f} of each other, while the "
               f"fold-to-fold std of the selected model is {best_std:.4f}. "
               + ("Cross-validation therefore cannot clearly separate the models; the choice is close to a tie."
                  if cv_spread < best_std else "The gap between models is larger than the fold-to-fold variation."))
    best_test_name = test_table["Test_R2"].idxmax()
    if best_test_name == final_name:
        report.add(f"  The test set agrees: {final_name} also has the highest test R2.")
    else:
        report.add(f"  Note: the test ranking is different - {best_test_name} has the highest test R2. The test set was NOT used "
                   f"to choose, because that would leak test information into model selection. With only "
                   f"{n_test} test rows, the test ranking can be strongly affected by chance.")


def report_error_analysis(df, X_test, y_test, final_pred, residuals, report):
    """Describe the biggest test errors so the residual plot can be interpreted with facts."""
    report.add(f"  Mean residual = {residuals.mean():.3f}, std = {residuals.std():.3f}, "
               f"largest absolute residual = {residuals.abs().max():.3f}.")
    predictions = pd.Series(final_pred, index=y_test.index)
    largest = residuals.abs().sort_values(ascending=False)
    report.add("  Three largest errors (test rows):")
    for idx in largest.head(3).index:
        report.add(f"    {df.loc[idx, 'Car_Name']} ({df.loc[idx, 'Year']}, Owner = {df.loc[idx, 'Owner']}): "
                   f"actual = {y_test.loc[idx]:.2f}, predicted = {predictions.loc[idx]:.2f}, "
                   f"Present_Price = {X_test.loc[idx, 'Present_Price']:.2f}")
    top2_share = (largest.head(2) ** 2).sum() / (residuals ** 2).sum()
    report.add(f"  The two largest errors make up {top2_share * 100:.0f}% of the final model's total squared test error "
               "(which is why RMSE is much bigger than MAE).")
    cheap = y_test < 10
    low_err = residuals[cheap].abs().mean()
    high_err = residuals[~cheap].abs().mean() if (~cheap).any() else float("nan")
    report.add(f"  Mean absolute residual for test vehicles priced below 10: {low_err:.3f} (n = {int(cheap.sum())}); "
               f"priced 10 or above: {high_err:.3f} (n = {int((~cheap).sum())}).")
    if high_err > low_err:
        report.add("  Interpretation: errors are larger for the higher-priced vehicles, but only "
                   f"{int((~cheap).sum())} test vehicles are in that group, so this should not be over-interpreted.")
    else:
        report.add("  Interpretation: errors are not larger for the higher-priced vehicles. Because the test set is small, "
                   "no strong conclusion should be drawn.")


def report_observations_and_limitations(df, raw_rows, raw_duplicates, final_name, cv_table, test_table, perm_df, n_test, report):
    """Write the closing observations and limitations - all numbers come from the run."""
    report.section("12. IMPORTANT OBSERVATIONS")
    report.add(f"- Dataset: {raw_rows} raw rows, {raw_duplicates} duplicates removed, {len(df)} rows used for modelling.")
    report.add(f"- Final model (by cross-validation): {final_name}.")
    top_perm = perm_df.sort_values("Importance", ascending=False).iloc[0]
    report.add(f"- Most influential input for the final model (permutation importance): {top_perm['Feature']}.")
    memorised = test_table.index[test_table["Train_R2"] >= 0.98].tolist()
    report.add(f"- Models that fit the training data almost perfectly (train R2 >= 0.98): {', '.join(memorised) or 'none'}. "
               "A big gap between train R2 and test R2 is a sign of overfitting.")
    lower_on_test = (test_table["Test_R2"] < cv_table["CV_R2_mean"]).sum()
    report.add(f"- {lower_on_test} of {len(test_table)} models score lower on the test set than in cross-validation. "
               f"CV R2 values lie between {cv_table['CV_R2_mean'].min():.2f} and {cv_table['CV_R2_mean'].max():.2f}, "
               f"but test R2 ranges from {test_table['Test_R2'].min():.2f} to {test_table['Test_R2'].max():.2f}. "
               f"With only {n_test} test rows, a few unusual vehicles can change the score a lot, so the test ranking "
               "of models is not very reliable.")
    is_bike = count_two_wheelers(df)
    individual = df["Selling_type"] == "Individual"
    dealer = df["Selling_type"] == "Dealer"
    report.add(f"- Selling_type is closely tied to vehicle type here: {int((is_bike & individual).sum())} of "
               f"{int(individual.sum())} 'Individual' listings are motorcycles/scooters, while "
               f"{int((is_bike & dealer).sum())} of {int(dealer.sum())} 'Dealer' listings are. "
               "Its effect on price should not be read as a pure selling-channel effect.")
    report.add("- Permutation importance can be negative or near zero: shuffling that column did not hurt the test score "
               f"(the feature was not useful on these {n_test} rows). It does not mean the feature lowers prices.")

    report.section("13. LIMITATIONS")
    report.add(f"- Small dataset: only {len(df)} rows after cleaning, so the test set has just {n_test} rows. "
               "One random split gives a noisy estimate.")
    report.add(f"- Data is historical (Year {df['Year'].min()}-{df['Year'].max()}) and has no market-date column; "
               "the model does not know today's prices.")
    report.add("- Price units are not stated in the CSV (they look like lakhs of Indian rupees, but this was not verified).")
    report.add(f"- The 'car' file also contains {int(is_bike.sum())} motorcycles/scooters, so it is not a pure car dataset.")
    report.add(f"- Rare categories: only {int((df['Fuel_Type'] == 'CNG').sum())} CNG vehicles, only "
               f"{int((df['Owner'] > 0).sum())} rows with Owner above 0 (just {int((df['Owner'] == 3).sum())} with Owner = 3), and "
               f"{int((df['Transmission'] == 'Automatic').sum())} automatics; the models cannot learn these small groups reliably.")
    report.add(f"- Extreme values were kept: maximum Driven_kms = {df['Driven_kms'].max():,} and "
               f"maximum Present_Price = {df['Present_Price'].max():.1f}.")
    report.add("- Important real-world factors are missing: city, variant/model trim, condition, accident history, service records.")
    report.add("- Car_Name was not used, so the model cannot separate two models with similar price and age.")
    report.add("- Models use default settings (no tuning), and the final model was fitted on the training split only.")
    report.add("- Feature importance and correlations show association, not causation.")


# --------------------------------------------------------------------------
# Main workflow
# --------------------------------------------------------------------------
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # avoids console encoding errors on Windows
    except AttributeError:
        pass

    report = Report()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    report.add("CodeAlpha Data Science Internship - Task 3: Car Price Prediction")
    report.add("All numbers below were produced by running src/car_price_prediction.py")

    # 1. Load + inspect
    raw_df = load_data(DATA_PATH)
    required = {"Car_Name", "Year", TARGET, "Present_Price", "Driven_kms",
                "Fuel_Type", "Selling_type", "Transmission", "Owner"}
    raw_df.columns = raw_df.columns.str.strip()
    if not required.issubset(raw_df.columns):
        sys.exit(f"ERROR: missing expected columns: {required - set(raw_df.columns)}")
    inspect_data(raw_df, report)
    raw_rows = len(raw_df)
    raw_duplicates = int(raw_df.duplicated().sum())
    two_wheeler_flag = count_two_wheelers(raw_df)
    report.add(f"\nVehicle types (judged from names, documentation only): {int((~two_wheeler_flag).sum())} cars and "
               f"{int(two_wheeler_flag.sum())} motorcycles/scooters. The file is called 'car data' but includes both.")

    # 2-3. Clean + features
    df = clean_data(raw_df, report)
    df, reference_year = add_features(df, report)

    # 4. EDA
    run_eda(df, report)

    # 5-6. Split (BEFORE any learning from the data)
    report.section("5. TRAIN / TEST SPLIT AND PREPROCESSING")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    report.add(f"Split: {len(X_train)} training rows ({100 - TEST_SIZE * 100:.0f}%) and {len(X_test)} test rows "
               f"({TEST_SIZE * 100:.0f}%), random_state = {RANDOM_STATE}.")
    report.add("Training data = the rows the models learn from. Testing data = rows kept hidden until the very end, "
               "used to check how the model does on cars it has never seen.")
    report.add("Why split: scoring a model on the same rows it learned from would reward memorising, not learning.")
    report.add(f"Model inputs: numeric = {NUMERIC_FEATURES}; categorical = {CATEGORICAL_FEATURES}; target = {TARGET}")
    report.add("Preprocessing (StandardScaler + OneHotEncoder) lives inside each Pipeline, so it is fitted on training data only.")
    report.add(f"Fuel_Type in training set: {X_train['Fuel_Type'].value_counts().to_dict()}")
    report.add(f"Fuel_Type in test set: {X_test['Fuel_Type'].value_counts().to_dict()}")
    report.add("(OneHotEncoder uses handle_unknown='ignore', so a category missing from training does not crash prediction.)")

    # 7. Models
    models = build_models()
    report.section("6. MODELS TESTED")
    report.add("Linear Regression, Decision Tree, Random Forest (200 trees), Gradient Boosting, KNN (k = 5).")
    report.add("All use library default settings (apart from a fixed random_state); no hyperparameter tuning was done.")

    # 7. Cross-validation on the TRAINING set decides the final model
    cv_table = run_cross_validation(models, X_train, y_train, report)
    final_name = choose_final_model(cv_table)

    # 8. Test evaluation (all models, shown for transparency - not used to choose)
    test_table = evaluate_on_test(models, X_train, y_train, X_test, y_test, report)
    cv_table.join(test_table).round(4).to_csv(COMPARISON_CSV_PATH)

    # 9. Report the selection
    report_model_selection(final_name, cv_table, test_table, len(X_test), report)
    final_pipeline = models[final_name]  # already fitted on the training set
    final_pred = final_pipeline.predict(X_test)
    report.add(f"  Final model test results: MAE = {mean_absolute_error(y_test, final_pred):.4f}, "
               f"RMSE = {rmse(y_test, final_pred):.4f}, R2 = {r2_score(y_test, final_pred):.4f}")

    # 10. Charts 9-12
    report.section("10. MODEL CHARTS AND INTERPRETATION")
    plot_model_comparison(cv_table, test_table)
    report.add("[09] Model comparison chart saved (CV and test R2 / RMSE; bars start at zero).")
    plot_actual_vs_predicted(y_test, final_pred, final_name)
    report.add("[10] Actual vs predicted chart saved. Points close to the red dashed line are accurate predictions.")
    residuals = plot_residuals(y_test, final_pred, final_name)
    report.add("[11] Residual analysis saved (residual = actual - predicted).")
    report_error_analysis(df, X_test, y_test, final_pred, residuals, report)
    _, perm_df = plot_feature_importance(final_pipeline, X_test, y_test, final_name, report)
    report.add("[12] Feature importance chart saved. Importance shows what the model relies on; it does not prove cause and effect.")

    # 11. Sample predictions
    make_sample_predictions(final_pipeline, df, reference_year, report)

    # 12-13. Observations + limitations
    report_observations_and_limitations(df, raw_rows, raw_duplicates, final_name, cv_table, test_table,
                                        perm_df, len(X_test), report)

    report.save(RESULTS_PATH)
    print(f"\nDone. Results saved to {RESULTS_PATH}")
    print(f"Charts saved to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
