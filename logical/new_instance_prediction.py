import pandas as pd
import numpy as np
import joblib
import os


def predict_instance_id3(model, instance, val):
    prediction = model.predict(instance)
    if prediction[0] > 5:
        return 5
    else:
        return val[prediction[0]]


def predict_instance_RN_continuous(model, instance, scaler):
    instance = scaler.transform(instance)
    predictions = model.predict([instance])
    return min(5, max(1.5, predictions[0][0]))


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

    if prediction[0] < 1.5:
        return 1.5
    else:
        return prediction[0]


def predict_instance_random_forest(model, instance, val):
    prediction = model.predict(instance)

    if prediction[0] > 5:
        return 5
    else:
        return val[prediction[0]]


def predict_instance_ada_boost(model, instance):
    prediction = model.predict(instance)
    return prediction[0]


def round_five(numar):
    return round(numar / 0.5) * 0.5


def predict_dataset_1(atr1, atr2, atr3, atr4, atr5):
    # -----------------LOAD--------------------
    save_directory = './trained_models'

    label_encoder = joblib.load(os.path.join(save_directory, 'label_encoder.joblib'))

    dt_model = joblib.load(os.path.join(save_directory, 'dt_model.joblib'))

    RN_model1 = joblib.load(os.path.join(save_directory, 'RN_model1.joblib'))
    sc_rn1 = joblib.load(os.path.join(save_directory, 'sc_rn1.joblib'))

    RN_model2 = joblib.load(os.path.join(save_directory, 'RN_model2.joblib'))
    sc_rn2 = joblib.load(os.path.join(save_directory, 'sc_rn2.joblib'))

    label_encoder_bayes = joblib.load(os.path.join(save_directory, 'label_encoder_bayes.joblib'))
    bayes_model = joblib.load(os.path.join(save_directory, 'bayes_model.joblib'))

    linear_regression_model = joblib.load(os.path.join(save_directory, 'linear_regression_model.joblib'))

    rf_model = joblib.load(os.path.join(save_directory, 'rf_model.joblib'))

    ada_boost_model = joblib.load(os.path.join(save_directory, 'ada_boost_model.joblib'))

    new_instance = pd.DataFrame({
        "How often patients got better at walking or moving around": [atr1],
        "How often patients got better at getting in and out of bed": [atr2],
        "How often patients got better at bathing": [atr3],
        "How often patients' breathing improved": [atr4],
        "How often patients got better at taking their drugs correctly by mouth": [atr5]
    })

    val = [1.5, 2, 2.5, 3, 4, 4.5, 5]
    print("-----------------Predicting instance------------------")
    id3_prediction = round_five(predict_instance_id3(dt_model, new_instance, val))
    RN_continuous_prediction = round_five(predict_instance_RN_continuous(RN_model1, new_instance, sc_rn1))
    RN_multiclass_prediction = round_five(
        predict_instance_RN_multiclass(label_encoder, RN_model2, new_instance, sc_rn2))
    bayes_prediction = round_five(predict_instance_bayes(label_encoder_bayes, bayes_model, new_instance))
    linear_regression_prediction = round_five(predict_instance_linear_regression(linear_regression_model, new_instance))
    random_forest_prediction = round_five(predict_instance_random_forest(rf_model, new_instance, val))
    ada_boost_prediction = round_five(predict_instance_ada_boost(ada_boost_model, new_instance))

    print("predicting instance with id3: ", id3_prediction)
    print("predicting instance with RN - continuous: ", RN_continuous_prediction)
    print("predicting instance with RN - multiclass: ", RN_multiclass_prediction)
    print("predicting instance with Bayes: ", bayes_prediction)
    print("predicting instance with Linear Regression: ", linear_regression_prediction)
    print("predicting instance with Random Forest: ", random_forest_prediction)
    print("predicting instance with Ada Boost: ", ada_boost_prediction)

    return id3_prediction, RN_continuous_prediction, RN_multiclass_prediction, bayes_prediction, linear_regression_prediction, random_forest_prediction, ada_boost_prediction
