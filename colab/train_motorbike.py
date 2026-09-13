from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "colab"

# 1. Giả lập dữ liệu thị trường Xe Máy Cũ Việt Nam
np.random.seed(42)
n_samples = 600

# Các dòng xe phổ biến: Wave(110cc), AirBlade/Lead(125cc), Exciter/Winner(150cc), SH(160cc)
engine_cc = np.random.choice([110, 125, 150, 160], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
year_bought = np.random.randint(2015, 2024, size=n_samples) # Năm đăng ký (2015 - 2023)
mileage_km = (2024 - year_bought) * np.random.randint(4000, 12000, size=n_samples) # Số km đi được

# Công thức tính giá xe cũ thực tế tại Việt Nam (Triệu VNĐ)
base_price = {110: 20, 125: 40, 150: 50, 160: 90} # Giá xe mới trung bình
price = np.array([base_price[cc] for cc in engine_cc])
# Khấu hao theo năm (mỗi năm giảm ~7%) và theo km (mỗi 10,000 km giảm 1.5 triệu)
price = price * (0.93 ** (2024 - year_bought)) - (mileage_km / 10000 * 1.5) + np.random.normal(0, 1.5, n_samples)
price = np.clip(price, 5.0, 110.0) # Giá tối thiểu 5 triệu, tối đa 110 triệu

df = pd.DataFrame({
    'year_bought': year_bought,
    'mileage_km': mileage_km,
    'engine_cc': engine_cc,
    'price_vnd': price
})

# 2. Chia tập Train/Test
X = df[['year_bought', 'mileage_km', 'engine_cc']]
y = df['price_vnd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Chuẩn hóa dữ liệu (Feature Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Tìm K tối ưu & lưu Đồ thị Elbow
k_range = range(1, 16)
r2_scores = []

for k in k_range:
    knn_temp = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn_temp.fit(X_train_scaled, y_train)
    r2_scores.append(r2_score(y_test, knn_temp.predict(X_test_scaled)))

OUTPUT_DIRECTORY.mkdir(exist_ok=True)
plt.figure(figsize=(7, 4))
plt.plot(k_range, r2_scores, marker='o', color='g', linestyle='--')
plt.title('Đánh giá chỉ số R2-Score theo giá trị K (Dự đoán Giá Xe Máy)')
plt.xlabel('Giá trị K')
plt.ylabel('R2 Score')
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIRECTORY / "elbow_k.png")
plt.close()

# 5. Huấn luyện model tối ưu với K=5
best_k = 5
model = KNeighborsRegressor(n_neighbors=best_k, weights='distance')
model.fit(X_train_scaled, y_train)

# 6. Đánh giá và lưu ảnh Sai số Dự đoán
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, alpha=0.7, color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title(f'So sánh Giá Thực tế vs Dự đoán (MAE: {mae:.2f} Triệu VNĐ)')
plt.xlabel('Giá Thực Tế (Triệu VNĐ)')
plt.ylabel('Giá Dự Đoán (Triệu VNĐ)')
plt.tight_layout()
plt.savefig(OUTPUT_DIRECTORY / "confusion_matrix.png")
plt.close()

# 7. Lưu Model và Scaler
joblib.dump(model, OUTPUT_DIRECTORY / "knn_model.pkl")
joblib.dump(scaler, OUTPUT_DIRECTORY / "scaler.pkl")

print("✅ ĐÃ HOÀN THÀNH HUẤN LUYỆN DỰ ĐOÁN GIÁ XE MÁY:")
print(f"- Sai số trung bình (MAE): {mae:.2f} triệu VNĐ")
print(f"- Đạt độ tương quan R2 Score: {r2*100:.2f}%")
print("- Đã lưu model: colab/knn_model.pkl")
print("- Đã lưu scaler: colab/scaler.pkl")
print("- Đã lưu ảnh colab/elbow_k.png và colab/confusion_matrix.png")