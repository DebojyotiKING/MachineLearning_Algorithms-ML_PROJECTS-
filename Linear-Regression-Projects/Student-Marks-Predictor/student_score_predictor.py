import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

Data = pd.read_csv(r"/content/students_scores.csv")
X = Data[["Hours"]]
Y = Data["Scores"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

Scaler = StandardScaler()
X_train_scaled = Scaler.fit_transform(X_train)
X_test_scaled = Scaler.transform(X_test)

Regression = LinearRegression()
Regression.fit(X_train_scaled, Y_train)

Y_pred = Regression.predict(X_test_scaled)
print(f"The predicted Scores are: {Y_pred}")

MSE = mean_squared_error(Y_test, Y_pred)
R2 = r2_score(Y_test, Y_pred)
print(f"The Mean Squared Error is: {MSE}")
print(f"The R2 Score is: {R2}")

plt.scatter(X_train, Y_train, color='blue', label="Train Data")
plt.plot(X_train, Regression.predict(X_train_scaled), color='red', label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Linear Regression - Study Hours vs Score")
plt.legend()
plt.grid(True)
plt.show()
