import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train():
    print("1. Đang tải tập dữ liệu Iris...")
    iris = load_iris()
    X, y = iris.data, iris.target

    print("2. Chia tập dữ liệu Train/Test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3. Chuẩn hóa dữ liệu (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("4. Huấn luyện mô hình K-Nearest Neighbors (K=5)...")
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn.fit(X_train_scaled, y_train)

    print("5. Đánh giá mô hình...")
    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"--> Độ chính xác (Accuracy): {acc * 100:.2f}%")
    print("\nBáo cáo chi tiết:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

    print("6. Lưu mô hình và Scaler...")
    joblib.dump(knn, 'colab/knn_model.pkl')
    joblib.dump(scaler, 'colab/scaler.pkl')
    print("--> Đã lưu thành công vào thư mục colab/")

if __name__ == "__main__":
    train()