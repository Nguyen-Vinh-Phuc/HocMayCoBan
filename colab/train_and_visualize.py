import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Tạo thư mục colab nếu chưa có
os.makedirs('colab', exist_ok=True)

# 2. Load dữ liệu Iris
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Tìm K tối ưu và vẽ biểu đồ ELBOW (Ảnh 1 cho Slide)
k_range = range(1, 16)
accuracies = []

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    accuracies.append(accuracy_score(y_test, knn_temp.predict(X_test_scaled)))

plt.figure(figsize=(7, 4))
plt.plot(k_range, accuracies, marker='o', color='b', linestyle='--')
plt.title('Độ chính xác (Accuracy) theo giá trị K')
plt.xlabel('Giá trị K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig('colab/elbow_k.png') # LƯU ẢNH 1
plt.close()

# 5. Huấn luyện model tối ưu với K=5
best_k = 5
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)

# 6. Vẽ Confusion Matrix (Ảnh 2 cho Slide)
y_pred = knn.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.tight_layout()
plt.savefig('colab/confusion_matrix.png') # LƯU ẢNH 2
plt.close()

# 7. Lưu Model và Scaler
joblib.dump(knn, 'colab/knn_model.pkl')
joblib.dump(scaler, 'colab/scaler.pkl')

print("✅ ĐÃ HOÀN THÀNH TẤT CẢ:")
print(f"- Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("- Đã lưu model: colab/knn_model.pkl")
print("- Đã xuất 2 ảnh biểu đồ: colab/elbow_k.png và colab/confusion_matrix.png")