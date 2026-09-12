from pathlib import Path

import matplotlib
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def prepare_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Tải, chia và chuẩn hóa dữ liệu Iris."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, list(iris.target_names)


def plot_accuracy_by_k(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    """Vẽ Accuracy theo K từ 1 đến 15 và lưu thành file PNG."""
    k_values = range(1, 16)
    accuracies: list[float] = []

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        model.fit(X_train, y_train)
        accuracies.append(accuracy_score(y_test, model.predict(X_test)))

    plt.figure(figsize=(10, 6))
    plt.plot(list(k_values), accuracies, marker="o", linewidth=2, color="#1f77b4")
    plt.xticks(list(k_values))
    plt.ylim(0.0, 1.05)
    plt.xlabel("Giá trị K")
    plt.ylabel("Accuracy")
    plt.title("Accuracy của mô hình KNN theo từng giá trị K")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    """Vẽ confusion matrix cho K=5 dưới dạng heatmap và lưu thành file PNG."""
    model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    matrix = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Nhãn dự đoán")
    plt.ylabel("Nhãn thực tế")
    plt.title("Ma trận nhầm lẫn của mô hình KNN (K=5)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Huấn luyện KNN và lưu hai biểu đồ đánh giá vào thư mục colab."""
    output_directory = Path(__file__).resolve().parent
    elbow_path = output_directory / "elbow_k.png"
    confusion_matrix_path = output_directory / "confusion_matrix.png"

    X_train, X_test, y_train, y_test, class_names = prepare_data()
    plot_accuracy_by_k(X_train, X_test, y_train, y_test, elbow_path)
    plot_confusion_matrix(
        X_train,
        X_test,
        y_train,
        y_test,
        class_names,
        confusion_matrix_path,
    )

    print(f"Đã lưu biểu đồ Accuracy theo K tại: {elbow_path}")
    print(f"Đã lưu confusion matrix tại: {confusion_matrix_path}")


if __name__ == "__main__":
    main()
