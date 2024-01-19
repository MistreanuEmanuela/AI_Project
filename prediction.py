import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def retain_random_n_per_category(data, column_name, n=200):
    result_df = pd.DataFrame()
    values_set = set(data[column_name])

    for value in values_set:
        category_data = data[data[column_name] == value]
        if len(category_data) <= n:
            retained_category_data = category_data
        else:
            retained_category_data = category_data.sample(n=n, random_state=42)
        result_df = pd.concat([result_df, retained_category_data])

    return result_df


def data_for_classification(filename, target_column):
    len_category = []
    data = pd.read_csv(filename)
    values = set(data[target_column])
    for column_value in values:
        category_df = data[data[target_column] == float(column_value)]
        len_category.append(len(category_df[target_column]))
    data = retain_random_n_per_category(data, target_column, n=round(np.mean(len_category)))
    return data


def load_data(filename, target_column):
    df = data_for_classification(filename, target_column)
    output = df[target_column]
    attributes = df.drop(columns=[target_column])
    return attributes, output


X, Y = load_data("preprocessing_data.csv",'HHCAHPS Survey Summary Star Rating')

print("------------------ID3----------------------")


def preprocess_data(X, Y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(Y)
    return X, y_encoded


def train_id3(X, Y):
    X, y_encoded = preprocess_data(X, Y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(X_train, y_train)
    y_pred = dt_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy id3: {accuracy}")
    print(f"Error id3: {1 - accuracy}")
    return dt_classifier, accuracy


model_id3, accuracy_id3 = train_id3(X, Y)


def test_id3(model, instance):
    new_instance = np.array(instance).reshape(1, -1)
    new_instance_prediction = model.predict(new_instance)
    print(f"Prediction for the new instance: {new_instance_prediction}")


test_id3(model_id3, [4,3,3,4])


print()
print("RN -------------y is a continue value from 1 to 5 ---------------------")


def train_neuronal_network_continuous(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
    model = Sequential()
    model.add(Dense(units=50, activation='relu', input_dim=4))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=200, batch_size=100, validation_data=(X_test, y_test))
    loss, mae = model.evaluate(X_test, y_test)
    print(f'Test Mean Absolute Error: {mae}')
    predictions = model.predict(X_test)
    accuracy = 1 - mae
    return model, accuracy, history


model_rn, accuracy_RN_1, history = train_neuronal_network_continuous(X, Y)


def plot_rn_mae(history):
    plt.plot(history.history['val_mae'], label='Test MAE')
    plt.title('Mean test error')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()


def predict_neuronal_network(model, new_instance):
    predictions = model.predict([new_instance])
    return predictions[0][0]


print(predict_neuronal_network(model_rn, [4,3,3,4]))

# print()
# print("RN -------------9 CLASSES WITH FIXED VALUES---------------------")
#
# y = pd.get_dummies(y)
# num_classes = 5
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y_encoded)
# model = Sequential()
# model.add(Dense(units=50, activation='relu', input_dim=4))
# model.add(Dense(units=num_classes, activation='softmax'))
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(X_train, y_train, epochs=200, batch_size=100, validation_data=(X_test, y_test))
# loss, accuracy = model.evaluate(X_test, y_test)
# accuracy_RN_2 = accuracy
# print(f'Test Accuracy: {accuracy}')
# plt.plot(history.history['val_accuracy'], label='Test Accuracy')
# plt.title('Accuracy on Test Set')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
#
# y_pred_proba = model.predict(X_test)
#
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
#
# for i in range(num_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test.iloc[:, i], y_pred_proba[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])
#
# plt.figure(figsize=(8, 6))
# colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']
#
# for i in range(num_classes):
#     plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
#
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic - Neural Network with Categorical Output')
# plt.legend(loc='lower right')
# plt.show()
#
# print()
# print("--------------------------------------BAYES NAIVE ------------------------------------------")
#
#
# label_encoder = LabelEncoder()
# y = df['HHCAHPS Survey Summary Star Rating']
# y = label_encoder.fit_transform(y)
#
# X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:5], y, test_size=0.25, random_state=42, stratify=y_encoded)
#
# scaler = StandardScaler()
#
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# vectorizer = TfidfVectorizer()
#
# X_train_tfidf = vectorizer.fit_transform(X_train.astype(str))
# X_test_tfidf = vectorizer.transform(X_test.astype(str))
#
# gnb = GaussianNB()
# gnb.fit(X_train_scaled, y_train)
#
# y_pred = gnb.predict(X_test_scaled)
# accuracy = accuracy_score(y_test, y_pred)
# accuracy_Bayes_Gaussian = accuracy
# print(f'Test Accuracy: {accuracy}')
#
# print()
# print("--------------------------------------LINEAR REGRESSION------------------------------------------")
#
#
# X = df.iloc[:, 1:5]
# y = df['HHCAHPS Survey Summary Star Rating']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y_encoded)
#
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# regressor = LinearRegression()
# regressor.fit(X_train_scaled, y_train)
#
# y_pred = regressor.predict(X_test_scaled)
# mae = mean_absolute_error(y_test, y_pred)
# print(f'Test Mean Absolute Error: {mae}')
# accuracy_linear_regression = 1 - mae
# print(f'Accuracy: {1 - mae}')
#
# print("--------------------------------------COMPARISON------------------------------------------")
#
# print("ID3:")
# print(f'Test accuracy: {accuracy_id3}')
# print()
#
# print("Neural Network (Regression):")
# print(f'Test accuracy: {accuracy_RN_1}')
# print()
#
# print("Neural Network (Classification):")
# print(f'Test Accuracy: {accuracy_RN_2}')
# print()
#
# print("Naive Bayes (Gaussian):")
# print(f'Test Accuracy: {accuracy_Bayes_Gaussian}')
# print()
#
#
# print("Linear Regression:")
# print(f'Test accuracy: {accuracy_linear_regression}')
# print()
#
# models = ['ID3', 'Neural Network (Regression)', 'Neural Network (Classification)', 'Naive Bayes (Gaussian)',
#           'Linear Regression']
# accuracies = [accuracy_id3, accuracy_RN_1, accuracy_RN_2, accuracy_Bayes_Gaussian,
#               accuracy_linear_regression]
#
# plt.bar(models, accuracies, color=['pink', 'blue', 'orange', 'green', 'red', 'purple'])
# plt.ylim(0, 1.2)
# plt.title('Test Accuracies Comparison')
# plt.ylabel('Accuracy')
# plt.xticks(rotation=0, ha='center')
# plt.show()
#
# print("---------------------RANDOM FOREST--------------------------------")
#
# X = df.iloc[:, 1:5]
# y = df['HHCAHPS Survey Summary Star Rating']
#
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)
#
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)
#
# y_pred_rf = rf_classifier.predict(X_test)
#
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print(f'Test Accuracy (Random Forest): {accuracy_rf}')
#
# feature_importances = rf_classifier.feature_importances_
# print("Feature Importances:")
# for feature, importance in zip(X.columns, feature_importances):
#     print(f"{feature}: {importance}")
#
#
#
# print()
# print("--------------------------------------ADABOOST (Regression)------------------------------------------")
#
# from sklearn.ensemble import AdaBoostRegressor
#
# X = df.iloc[:, 1:5]
# y = df['HHCAHPS Survey Summary Star Rating']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# adaboost_regressor = AdaBoostRegressor(n_estimators=50, random_state=42)
# adaboost_regressor.fit(X_train_scaled, y_train)
#
# y_pred_adaboost_regression = adaboost_regressor.predict(X_test_scaled)
#
# mae_adaboost_regression = mean_absolute_error(y_test, y_pred_adaboost_regression)
# print(f'Test Mean Absolute Error (AdaBoost Regression): {mae_adaboost_regression}')
# accuracy_adaboost_regression = 1 - mae_adaboost_regression
# print(f'Accuracy (AdaBoost Regression): {accuracy_adaboost_regression}')
#
# print("---------------------ACCURACY SUMMARY--------------------------------")
#
# models = ['ID3', 'Neural Network (Regression)', 'Neural Network (Classification)', 'Naive Bayes (Gaussian)',
#           'Linear Regression', 'AdaBoost Regression', 'Random Forest']
# accuracies = [accuracy_id3, accuracy_RN_1, accuracy_RN_2, accuracy_Bayes_Gaussian,
#               accuracy_linear_regression, accuracy_adaboost_regression, accuracy_rf]
#
# plt.bar(models, accuracies, color=['pink', 'blue', 'orange', 'green', 'red', 'purple', 'cyan', 'magenta'])
# plt.ylim(0, 1.2)
# plt.title('Test Accuracies Comparison')
# plt.ylabel('Accuracy')
# plt.xticks(rotation=45, ha='right')
# plt.show()