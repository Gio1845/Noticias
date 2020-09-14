from sklearn import datasets, linear_model, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("OnlineNewsPopularity", sep= ";")

X = df[["G1", "G2", "self_reference_min_shares", "self_reference_max_shares", "self_reference_avg_sharess"]]
y = df [["G3"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.10, random_state=93)

model = datasets.NewsPopularity(n_news = 2, weigths = "Distance")

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_test[["G1"]], y_test, label="Test")
plt.scatter(X_test[["G1"]], y_pred, label="pred")
plt.legend()
plt.title("News Popularity Online")
plt.show