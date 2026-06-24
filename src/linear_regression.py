import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =========================
# 1. データ読み込み
# =========================
df = pd.read_csv('date/wine.csv')

# =========================
# 2. 説明変数と目的変数
# =========================
# 相関が強そうな変数を選択（理由を説明できるように）
X = df[['alcohol', 'volatile acidity', 'sulphates']]
y = df['quality']

# =========================
# 3. モデル作成・学習
# =========================
model = LinearRegression()
model.fit(X, y)

# =========================
# 4. 予測
# =========================
y_pred = model.predict(X)

# =========================
# 5. 評価（←ここ入れると一気に良くなる）
# =========================
r2 = model.score(X, y)
print("R^2:", r2)

# =========================
# 6. 可視化
# =========================
plt.figure()

plt.scatter(y, y_pred)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Linear Regression Result")

# 理想線（y=x）を追加
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')

# =========================
# 7. 保存
# =========================
plt.savefig('result.png')
plt.show()
