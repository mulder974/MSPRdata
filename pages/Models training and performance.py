import streamlit

import streamlit as st
import db
import pickle
import pandas as pd
import model
import numpy as np
from datetime import datetime
import sklearn.metrics
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import time



def load_model_from_disk(model_path):
    loaded_model = pickle.load(open(f'{model_path}', 'rb'))
    return loaded_model



df_data = pd.DataFrame(data = db.query_data())
x_train, x_test, y_train, y_test, le = model.preprocess(df_data)


dic_model =  {"log_reg_clf": {"model": model.create_model_logreg(),
                                      "Grid_Search_Param": {'solver': ('newton-cg', 'liblinear'),
                                                            'max_iter': [e for e in range(5000, 7000, 250)]}
                                      },

              "random_forest": {"model": model.create_model_rand_for(),
                                "Grid_Search_Param": {'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200],
                                                      'criterion': ['gini', 'entropy', 'log_loss'],
                                                      'max_depth': [e for e in range(0, 32, 2)]}
                                },

              "neur_net": {"model": model.create_model_neur_net(),
                                 "Grid_Search_Param": {'hidden_layer_sizes': (300, 400, 500),
                                                       'alpha' : [0.01, 0.001, 0.05, 0.1],
                                                       'max_iter': [e+1 for e in range(500, 2000, 250)],
                                                       'learning_rate': ['constant']}
                                 }
              }

pages_list = ["Retrain Model", "See models performances", "Test models"]


def selectbox_without_default(label, options, key):
    format_func = lambda x: 'Select one option' if x == '' else x
    return st.selectbox(label, options, format_func=format_func, key = key)



def measure_accuracy(y_test,y_predicted):
    accuracy = sklearn.metrics.accuracy_score(y_test, y_predicted)
    return accuracy


def plot_confusion_matrix(model, y_test, y_predicted):
    cm = sklearn.metrics.confusion_matrix(y_test, y_predicted, labels=model.classes_)
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                                  display_labels=model.classes_)
    fig = disp.plot()

    return fig


def plot_confusion_matrix_last_perf(model, last_y_test, last_y_predicted):
    cm = sklearn.metrics.confusion_matrix(last_y_test, last_y_predicted, labels=model.classes_)
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                                  display_labels=model.classes_)
    fig = disp.plot()

    return fig



page_selected = selectbox_without_default("what do you want to do ? ", pages_list, key = "3" )



if page_selected == "Retrain Model":
    st.write("If you click the button below, the 3 models will be retrained")
    if st.button('Retrain all models'):
        with st.spinner(" Printing trainings details:..."):

            model_trainings = model.retrain_all_model(x_train, x_test, y_train, y_test, le)



elif page_selected == "See models performances":
    model_selected = selectbox_without_default("Choose a model", dic_model.keys(), key = "1")
    model_loaded = load_model_from_disk(f"{model_selected}.SAV")
    model_ = model_loaded["model"]
    x_test = model_loaded["x_test"]
    y_test = model_loaded["y_test"]
    model_encoder = model_loaded["encoder"]

    y_pred, accuracy = model.test_model(model_,x_test, y_test)
    # y_pred = pd.DataFrame(y_pred, columns=["partie"])
    # y_pred = model_encoder.inverse_transform(y_pred)
    # y_test = pd.DataFrame(y_test, columns=["partie"])
    y_test2 = model_encoder.inverse_transform(y_test)
    st.write(y_test)
    st.write(y_test2)
    st.pyplot(plot_confusion_matrix(model_, y_test, y_pred).figure_)
    """Accuracy on test_set was : """ + str(accuracy)


elif page_selected == "Test models":

    model_selected = selectbox_without_default("Choose a model", dic_model.keys(), key = "4")
    hab = st.slider("Nombre d'habitants de la commune", 1, 2200000, 1000)
    police = st.slider("Policiers pour 100 habs", 0, 100, 1)
    revenu_fiscal = st.slider("Revenu fiscal des foyers", 0, 7800000,10000)
    evolution = st.slider("Evolution en %", 0, 50, 1)
    age = st.slider("age moyen ", 20.00, 40.00, 0.2)

    model_selected = selectbox_without_default("Choose a model", dic_model.keys(), key = "2")
    model_loaded = load_model_from_disk(f"{model_selected}.SAV")
    model_ = model_loaded["model"]
    x_test = model_loaded["x_test"]
    model_encoder = model_loaded["encoder"]

    x = (hab,police,revenu_fiscal,evolution,age)
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_standardize = scaler.transform(np.array(x).reshape(1,-1))


    if st.button('Click to predict'):
        with st.spinner(" Predicting..."):
            time.sleep(1)
            y_pred = model_.predict(x_standardize)
            y_pred = model_encoder.inverse_transform(y_pred)
            st.write(f"The model predict that the winner for this city would be : {y_pred}")




st.markdown("***")






# else :
#
#     database.query_model(model_selected)
#
#     st.write("Model overview after training: ")
#     st.pyplot(plot_learning_curve(model_selected))
#     st.write(f"Accuracy is : {accuracy}")
#     st.pyplot(confusion_matrix)
#     st.markdown("***")







#ploting learning curve








