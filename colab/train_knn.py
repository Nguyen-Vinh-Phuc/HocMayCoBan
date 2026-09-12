from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def train() -> None:
    """Huấn luyện KNN trên dữ liệu Iris và lưu model cùng scaler."""
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "colab" / "knn_model.pkl"
    scaler_path = project_root / "colab" / "scaler.pkl"

    print("1. Đang tải tập dữ liệu Iris...")
    iris = load_iris()
    X, y = iris.data, iris.target

    print("2. Chia tập dữ liệu Train/Test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("3. Chuẩn hóa dữ liệu (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("4. Huấn luyện mô hình K-Nearest Neighbors (K=5)...")
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn.fit(X_train_scaled, y_train)

    print("5. Đánh giá mô hình...")
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"--> Độ chính xác (Accuracy): {acc * 100:.2f}%")
    print("\nBáo cáo chi tiết:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

    print("6. Lưu mô hình và Scaler...")
    # Lưu model và scaler cạnh script để API có thể tái sử dụng cùng quy trình tiền xử lý.
    joblib.dump(knn, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"--> Đã lưu model tại: {model_path}")
    print(f"--> Đã lưu scaler tại: {scaler_path}")


if __name__ == "__main__":
    train()