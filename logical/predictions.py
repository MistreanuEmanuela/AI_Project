from keras.src.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import joblib
import os


def custom_round(number, precision):
    return round(number / precision) * precision


def data_view_per_class(data, target):
    target_counts = data[target].value_counts()
    target_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Number of Rows for Each Value in "target" Column')
    plt.xlabel('Target Values')
    plt.ylabel('Count')
    plt.savefig("./diagrams/nr_inst_classes.png")
    plt.close()


def retain_random_n_per_category(data, column_name, n=50):
    result_df = pd.DataFrame()
    values_set = set(data[column_name])
    for value in values_set:
        category_data = data[data[column_name] == value]
        if len(category_data) <= n:
            retained_category_data = category_data
        else:
            retained_category_data = category_data.head(n)
        result_df = pd.concat([result_df, retained_category_data])
    return result_df


def data_for_classification(filename, target_column):
    len_category = []
    data = pd.read_csv(filename)
    data_view_per_class(data, target_column)
    values = set(data[target_column])
    for column_value in values:
        category_df = data[data[target_column] == float(column_value)]
        len_category.append(len(category_df[target_column]))
    data = retain_random_n_per_category(data, target_column, n=200)
    return data


def load_data(filename, target_column):
    df = data_for_classification(filename, target_column)
    output = df[target_column]
    attributes = df.drop(columns=[target_column])
    return attributes, output


def preprocess_data(x, y):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return x, y_encoded


def preprocess_data_rn(x, y):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return x, y_encoded, scaler


def train_id3(x, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(x_train, y_train)
    model = dt_classifier
    y_prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_prediction)
    return model, accuracy


def train_neuronal_network_continuous(x, y):
    x_inc, y_encoded, sc = preprocess_data_rn(x, y)
    input_dim = 5
    x_train, x_test, y_train, y_test = train_test_split(x_inc, y, test_size=0.25, random_state=42, stratify=y)
    model = Sequential()
    model.add(Dense(units=50, activation='relu', input_dim=input_dim))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    history = model.fit(x_train, y_train, epochs=200, batch_size=100, validation_data=(x_test, y_test))
    loss, mae = model.evaluate(x_test, y_test)
    accuracy = 1 - mae
    return model, history, accuracy, sc


def create_model(units=50, activation='relu', input_dim=4, alpha=0.01, num_classes=5):
    print(input_dim, num_classes)
    model = Sequential()
    model.add(Dense(units=units, activation=activation, input_dim=input_dim, kernel_regularizer=l2(alpha)))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_neuronal_network_multi_class(x, y, alpha_values=[0.01, 0.1, 1.0]):
    x_inp, y_enc, sc = preprocess_data_rn(x, y)
    print(x_inp)
    y = pd.get_dummies(y)
    num_classes = len(set(y))
    x_train, x_test, y_train, y_test = train_test_split(x_inp, y, test_size=0.25, random_state=42, stratify=y)
    input_dimension = len(x_inp[0])
    best_accuracy = 0
    best_params = {}

    for units in [50]:
        for activation in ['relu']:
            for alpha in alpha_values:
                model = create_model(units=units, activation=activation, input_dim=input_dimension, alpha=alpha,
                                     num_classes=num_classes)
                history = model.fit(x_train, y_train, epochs=50, batch_size=100, validation_data=(x_test, y_test),
                                    verbose=0)
                _, accuracy = model.evaluate(x_test, y_test, verbose=0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'units': units, 'activation': activation, 'input_dim': input_dimension,
                                   'alpha': alpha, 'num_classes': num_classes}

    final_model = create_model(**best_params)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    history = final_model.fit(x_train, y_train, epochs=200, batch_size=100, validation_data=(x_test, y_test),
                              callbacks=[checkpoint])

    final_model.load_weights('best_model.h5')
    accuracy = final_model.evaluate(x_test, y_test, verbose=0)
    return final_model, history, accuracy[1], x_train, x_test, y_test, sc


def train_bayes(x, y):
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_prediction = gnb.predict(x_test)
    accuracy = accuracy_score(y_test, y_prediction)
    return gnb, accuracy, label_encoder


def train_linear_regression(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)
    y_pred = linear_regression.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = 1 - mae
    return linear_regression, accuracy


def random_forest(x, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)
    y_prediction = rf_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_prediction)
    return rf_classifier, accuracy


def ada_boost(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
    adaboost_regressor = AdaBoostRegressor(n_estimators=50, random_state=42)
    adaboost_regressor.fit(x_train, y_train)
    y_prediction = adaboost_regressor.predict(x_test)
    mae = mean_absolute_error(y_test, y_prediction)
    accuracy = 1 - mae
    return adaboost_regressor, accuracy


def predict_instance_id3(model, instance, val):
    new_instance_prediction = model.predict(instance)
    return val[new_instance_prediction[0]]


def predict_instance_RN_continuous(model, instance, scaler):
    instance = scaler.fit_transform(instance)
    predictions = model.predict([instance])
    print(predictions)
    return predictions[0][0]


def predict_instance_RN_multiclass(label_encoder, model, instance, scaler):
    new_instance = scaler.transform(instance)
    predictions = model.predict(new_instance)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]

    return predicted_class_label


def predict_instance_bayes(label_encoder, model, instance):
    prediction = model.predict(instance)
    decoded_prediction = label_encoder.inverse_transform(prediction)[0]
    return decoded_prediction


def predict_instance_linear_regression(model, instance):
    prediction = model.predict(instance)
    return prediction[0]


def predict_instance_random_forest(model, instance, val):
    prediction = model.predict(instance)
    return val[prediction[0]]


def predict_instance_ada_boost(model, instance):
    prediction = model.predict(instance)
    return prediction[0]


def plot_rn_mae(history):
    plt.plot(history.history['val_mae'], label='Test MAE')
    plt.title('Mean test error (MAE) for Neural Network with Continuous Output')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.savefig("./diagrams/mean_error.png")
    plt.close()


def plot_rn_accuracy_continuous(history):
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title('Accuracy on test set for Neural Network with Multi-class Output')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("./diagrams/accuracy_test.png")
    plt.close()


def plot_rn_ROC(model, x_test, num_classes, y_test):
    y_prediction_proba = model.predict(x_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test.iloc[:, i], y_prediction_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
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
    plt.savefig("./diagrams/roc_neural.png")
    plt.close()


def plot_test_accuracies_comparison(accuracy_id3, accuracy_RN_1, accuracy_RN_2, accuracy_bayes,
                                    accuracy_linear_regression, accuracy_adaboost_regression, accuracy_rf):
    models = ['ID3', 'RN (Reg)', 'RN (Class)', 'NB (Gaus)',
              'NB (Multi)', 'Linear Reg', 'Rand Forest']
    accuracies = [accuracy_id3, accuracy_RN_1, accuracy_RN_2, accuracy_bayes,
                  accuracy_linear_regression, accuracy_adaboost_regression, accuracy_rf]
    plt.bar(models, accuracies, color=['pink', 'blue', 'orange', 'green', 'red', 'purple', 'cyan', 'magenta'])
    plt.ylim(0, 1.2)
    plt.title('Test Accuracies Comparison')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=0, ha='center', fontsize=8)
    plt.savefig("./diagrams/accuracy_comparison.png")
    plt.close()


def round_five(numar):
    return round(numar / 0.5) * 0.5


def predictions():
    X, Y = load_data("./data/Preprocessing_HH.csv", 'Quality of patient care star rating')
    print(X, Y)
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(Y)

    dt_model, accuracy_id3 = train_id3(X, Y)
    RN_model1, history_rn1, accuracy_RN_1, sc_rn1 = train_neuronal_network_continuous(X, Y)
    RN_model2, history_rn2, accuracy_RN_2, X_train_RN2, X_test_RN2, y_test_RN2, sc_rn2 = \
        train_neuronal_network_multi_class(X, Y)
    bayes_model, accuracy_bayes, label_encoder_bayes = train_bayes(X, Y)
    linear_regression_model, accuracy_linear_regression = train_linear_regression(X, Y)
    rf_model, accuracy_rf = random_forest(X, Y)
    ada_boost_model, accuracy_ada_boost = ada_boost(X, Y)

    print("-----------------Accuracy on testing------------------")
    print("Accuracy id3: ", accuracy_id3)
    print("Accuracy RN with continuous output: ", accuracy_RN_1)
    print("Accuracy RN multiclass: ", accuracy_RN_2)
    print("Accuracy Bayes: ", accuracy_bayes)
    print("Accuracy Linear Regression: ", accuracy_linear_regression)
    print("Accuracy Random Forest: ", accuracy_rf)
    print("Accuracy Ada Boost: ", accuracy_ada_boost)

    plot_rn_mae(history_rn1)
    plot_rn_accuracy_continuous(history_rn2)
    plot_rn_ROC(RN_model2, X_test_RN2, 5, y_test_RN2)
    plot_test_accuracies_comparison(accuracy_id3, accuracy_RN_1, accuracy_RN_2, accuracy_bayes,
                                    accuracy_linear_regression, accuracy_ada_boost, accuracy_rf)

    # ---------------------SAVING------------------------
    folder_path = "./trained_models"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    save_directory = './trained_models'

    joblib.dump(label_encoder, os.path.join(save_directory, 'label_encoder.joblib'))

    joblib.dump(dt_model, os.path.join(save_directory, 'dt_model.joblib'))

    joblib.dump(RN_model1, os.path.join(save_directory, 'RN_model1.joblib'))
    joblib.dump(sc_rn1, os.path.join(save_directory, 'sc_rn1.joblib'))

    joblib.dump(RN_model2, os.path.join(save_directory, 'RN_model2.joblib'))
    joblib.dump(sc_rn2, os.path.join(save_directory, 'sc_rn2.joblib'))

    joblib.dump(label_encoder_bayes, os.path.join(save_directory, 'label_encoder_bayes.joblib'))
    joblib.dump(bayes_model, os.path.join(save_directory, 'bayes_model.joblib'))

    joblib.dump(linear_regression_model, os.path.join(save_directory, 'linear_regression_model.joblib'))

    joblib.dump(rf_model, os.path.join(save_directory, 'rf_model.joblib'))

    joblib.dump(ada_boost_model, os.path.join(save_directory, 'ada_boost_model.joblib'))

