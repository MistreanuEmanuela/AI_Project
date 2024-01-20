import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
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
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor
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
    model = dt_classifier
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy


def train_neuronal_network_continuous(X, Y):
    X, y_encoded = preprocess_data(X, Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=y_encoded)
    model = Sequential()
    model.add(Dense(units=50, activation='relu', input_dim=4))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=200, batch_size=100, validation_data=(X_test, y_test))
    loss, mae = model.evaluate(X_test, y_test)
    accuracy = 1 - mae
    return model, history, accuracy


def train_neuronal_network_multi_class(X, Y):
    X, y_encoded = preprocess_data(X, Y)
    Y = pd.get_dummies(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=y_encoded)
    num_classes = 5
    model = Sequential()
    model.add(Dense(units=50, activation='relu', input_dim=4))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=200, batch_size=100, validation_data=(X_test, y_test))
    loss, accuracy = model.evaluate(X_test, y_test)
    return model, history, accuracy, X_train, X_test, y_test


def train_bayes(X, Y):
    X, y_encoded = preprocess_data(X, Y)
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42,
                                                     stratify=y_encoded)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return gnb, accuracy, label_encoder


def train_linear_regression(X, Y):
    X, y_encoded = preprocess_data(X, Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=y_encoded)
    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    y_pred = linear_regression.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = 1 - mae
    return linear_regression, accuracy


def random_forest(X, Y):
    X, y_encoded = preprocess_data(X, Y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return rf_classifier, accuracy


def ada_boost(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
    adaboost_regressor = AdaBoostRegressor(n_estimators=50, random_state=42)
    adaboost_regressor.fit(X_train, y_train)
    y_pred = adaboost_regressor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = 1 - mae
    return adaboost_regressor, accuracy


def predict_instance_id3(model, instance):
    new_instance = np.array(instance).reshape(1, -1)
    new_instance_prediction = model.predict(new_instance)
    return new_instance_prediction[0]


def predict_instance_RN_continuous(model, instance):
    predictions = model.predict([instance])
    return predictions[0][0]


def predict_instance_RN_multiclass(label_encoder, model, instance, X_train):
    scaler = StandardScaler()
    scaler.fit_transform(X_train)
    new_instance_scaled = scaler.transform(np.array(instance).reshape(1, -1))
    predictions = model.predict(new_instance_scaled.reshape(1, -1))
    predicted_class = np.argmax(predictions)
    predicted_class_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_class_label


def predict_instance_bayes(label_encoder, model, instance):
    new_instance_encoded = label_encoder.transform([instance[0]])[0]
    new_instance_features = np.array([new_instance_encoded, instance[1], instance[2], instance[3]]).reshape(1, -1)
    prediction = model.predict(new_instance_features)
    decoded_prediction = label_encoder.inverse_transform(prediction)[0]
    return decoded_prediction


def predict_instance_linear_regression(model, instance):
    instance = np.array(instance).reshape(1, -1)
    prediction = model.predict(instance)
    return prediction[0]


def predict_instance_random_forest(model, instance):
    instance = np.array(instance).reshape(1, -1)
    prediction = model.predict(instance)
    return prediction[0]


def predict_instance_ada_boost(model, instance):
    instance = np.array(instance).reshape(1, -1)
    prediction = model.predict(instance)
    return prediction[0]


def plot_rn_mae(history):
    plt.plot(history.history['val_mae'], label='Test MAE')
    plt.title('Mean test error (MAE) for Neural Network with Continuous Output')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()


def plot_rn_accuracy_continuous(history):
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title('Accuracy on test set for Neural Network with Multi-class Output')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_rn_ROC(model, X_test, num_classes, y_test):
    y_pred_proba = model.predict(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test.iloc[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive']
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Neural Network with Multi-class Output')
    plt.legend(loc='lower right')
    plt.show()


def plot_test_accuracies_comparison(accuracy_id3, accuracy_RN_1, accuracy_RN_2, accuracy_bayes, accuracy_linear_regression, accuracy_adaboost_regression, accuracy_rf):
    models = ['ID3', 'RN(Regression)', 'RN (Classification)', 'Naive Bayes',
              'Linear Regression', 'AdaBoost', 'Random Forest']
    accuracies = [accuracy_id3, accuracy_RN_1, accuracy_RN_2, accuracy_bayes,
                  accuracy_linear_regression, accuracy_adaboost_regression, accuracy_rf]
    plt.bar(models, accuracies, color=['pink', 'blue', 'orange', 'green', 'red', 'purple', 'cyan', 'magenta'])
    plt.ylim(0, 1.2)
    plt.title('Test Accuracies Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.show()


def main():
    X, Y = load_data("preprocessing_data.csv", 'HHCAHPS Survey Summary Star Rating')
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(Y)

    dt_model, accuracy_id3 = train_id3(X, Y)
    RN_model1, history_rn1, accuracy_RN_1= train_neuronal_network_continuous(X,Y)
    RN_model2, history_rn2, accuracy_RN_2, X_train_RN2, X_test_RN2, y_test_RN2 = train_neuronal_network_multi_class(X,Y)
    bayes_model, accuracy_bayes, label_encoder_bayes = train_bayes(X,Y)
    linear_regression_model, accuracy_linear_regression = train_linear_regression(X,Y)
    rf_model, accuracy_rf = random_forest(X,Y)
    ada_boost_model, accuracy_ada_boost = ada_boost(X,Y)

    print("-----------------Accuracy on testing------------------")
    print("Accuracy id3: ", accuracy_id3)
    print("Accuracy RN with continous output: ", accuracy_RN_1)
    print("Accuracy RN multiclass: ", accuracy_RN_2)
    print("Accuracy Bayes: ", accuracy_bayes)
    print("Accuracy Linear Regression: ", accuracy_linear_regression)
    print("Accuracy Random Forest: ", accuracy_rf)
    print("Accuracy Ada Boost: ", accuracy_ada_boost)

    print("-----------------Predicting instance------------------")
    print("predicting instance with id3: ", predict_instance_id3(dt_model, [4, 3, 4, 4]))
    print("predicting instance with RN - continous: ", predict_instance_RN_continuous(RN_model1, [4, 3, 4, 4]))
    print("predicting instance with RN - multiclass: ", predict_instance_RN_multiclass(label_encoder, RN_model2, [4, 3, 4, 4], X_train_RN2))
    print("predicting instance with Bayes: ", predict_instance_bayes(label_encoder, bayes_model, [4, 3, 4, 4]))
    print("predicting instance with Linear Regression: ", predict_instance_linear_regression(linear_regression_model, [4, 3, 4, 4]))
    print("predicting instance with Random Forest: ", predict_instance_random_forest(rf_model, [4, 3, 4, 4]))
    print("predicting instance with Ada Boost: ", predict_instance_ada_boost(ada_boost_model, [4, 3, 4, 4]))

    plot_rn_mae(history_rn1)
    plot_rn_accuracy_continuous(history_rn2)
    plot_rn_ROC(RN_model2, X_test_RN2, 5, y_test_RN2)
    plot_test_accuracies_comparison(accuracy_id3, accuracy_RN_1, accuracy_RN_2, accuracy_bayes, accuracy_linear_regression, accuracy_ada_boost, accuracy_rf)

if __name__ == "__main__":
    main()