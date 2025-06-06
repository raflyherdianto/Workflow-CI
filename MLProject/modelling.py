import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt

# Atur nama eksperimen MLflow
EXPERIMENT_NAME = "Supplement_Revenue_Prediction_CI"
mlflow.set_experiment(EXPERIMENT_NAME)

def load_data(file_path):
    """Memuat data yang sudah diproses dari file CSV."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data berhasil dimuat dari {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File data tidak ditemukan di {file_path}")
        return None

def train_and_tune_model(data_path):
    """
    Melatih model dengan tuning, melacak menggunakan MLflow, dan menyimpan run_id.
    """
    print("\n--- Memulai Proses Pelatihan & Tuning Model ---")
    df = load_data(data_path)
    if df is None:
        return

    target_column = 'Revenue'
    if target_column not in df.columns:
        print(f"Error: Kolom target '{target_column}' tidak ditemukan.")
        return

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data dibagi menjadi: {X_train.shape[0]} train dan {X_test.shape[0]} test.")

    # Definisikan model dan grid parameter untuk tuning
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Setup GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, 
                               scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

    print("\nMemulai hyperparameter tuning...")
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    best_cv_score = grid_search.best_score_
    
    # Konversi skor CV (neg_mean_squared_error) menjadi RMSE positif
    best_cv_rmse = np.sqrt(-best_cv_score)

    print(f"\nParameter terbaik ditemukan: {best_params}")
    print(f"RMSE terbaik dari Cross-Validation: {best_cv_rmse:.4f}")
    
    # Evaluasi model terbaik di test set
    predictions = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- Hasil Evaluasi Model Terbaik di Test Set ---")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2 Score: {r2:.4f}")

    # MLflow Manual Logging
    with mlflow.start_run(run_name="Automated_Tuned_Run") as run:
        print("\n--- Memulai Logging ke MLflow ---")
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # 1. Log Parameter Terbaik
        print("Logging best parameters...")
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "RandomForestRegressor_Tuned")
        
        # Log parameter dari GridSearchCV
        mlflow.log_param("cv_folds", grid_search.cv)
        mlflow.log_param("scoring_metric", grid_search.scoring)

        # 2. Log Metrik
        print("Logging metrics...")
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2_score", r2)
        mlflow.log_metric("best_cv_rmse", best_cv_rmse)

        # 3. Log Model Terbaik
        print("Logging best model...")
        mlflow.sklearn.log_model(best_model, "tuned_model")

        # 4. Log Artifak Plot
        print("Logging feature importance plot...")
        feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
        fig, ax = plt.subplots(figsize=(10, 8))
        feature_importances.nlargest(20).plot(kind='barh', ax=ax)
        ax.set_title("Top 20 Feature Importances (Tuned Model)")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        
        plot_path = "tuned_feature_importances.png"
        fig.savefig(plot_path)
        plt.close(fig)
        mlflow.log_artifact(plot_path, "feature_plots")
        os.remove(plot_path)

        # Menyimpan run_id ke file teks untuk digunakan oleh workflow CI/CD
        with open("mlflow_run_id.txt", "w") as f:
            f.write(run_id)
        
        print("--- Logging ke MLflow Selesai ---")
        print(f"Run ID {run_id} disimpan ke mlflow_run_id.txt")


if __name__ == "__main__":
    data_file = "supplement_sales_preprocessing.csv"
    train_and_tune_model(data_path=data_file)
    print("\nProses selesai.")