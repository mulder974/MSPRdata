import streamlit as st
import db
import pickle
import pandas as pd
import model
import numpy as np
from datetime import datetime
import sklearn.metrics
from sklearn.model_selection import GridSearchCV







def load_model_from_disk(model_path):
    loaded_model = pickle.load(open(f'{model_path}', 'rb'))
    return loaded_model



df_data = pd.DataFrame(data = db.query_data())
print(df_data)


dic_model =  {"log_reg_clf": {"model": model.create_model_logreg(),
                                      "Grid_Search_Param": {'solver': ('newton-cg', 'liblinear'),
                                                            'max_iter': [e for e in range(5000, 7000, 250)]}
                                      },

              "random_forest": {"model": model.create_model_rand_for(),
                                "Grid_Search_Param": {'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200],
                                                      'criterion': ['gini', 'entropy', 'log_loss'],
                                                      'max_depth': [e for e in range(0, 32, 2)]}
                                },

              "neural_net": {"model": model.create_model_neur_net(),
                                 "Grid_Search_Param": {'hidden_layer_sizes': (300, 400, 500),
                                                       'alpha' : [0.01, 0.001, 0.05, 0.1],
                                                       'max_iter': [e+1 for e in range(500, 2000, 250)],
                                                       'learning_rate': ['constant']}
                                 }
              }

pages_list = ["Retrain Model", "See models performances", "Test models"]


def selectbox_without_default(label, options):

    format_func = lambda x: 'Select one option' if x == '' else x
    return st.selectbox(label, options, format_func=format_func)



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



page_selected = selectbox_without_default("what do you want to do ? ", pages_list)



if page_selected == "Retrain Model":
    st.write("If you click the button below, the 3 models will be retrains")
    if st.button('Retrain all models'):

        with st.spinner(" Printing trainings details:..."):
            model.retrain_all_model()




        """Accuracy on last real data is : """ + str(accuracy)
elif page_selected == "See models performances":
    model_selected = selectbox_without_default("Choose a model", dic_model.keys())
    model = load_model_from_disk(f"{model_selected}.SAV")
    # st.pyplot(plot_confusion_matrix(model, y_test, y_predicted).figure_)
    # """Accuracy on test_set was : """ + str(accuracy)


elif page_selected == "Test models":

    with st.spinner(" Printing metrics..."):
        accuracies, dates = database.query_accuracies(model_id)
        print(accuracies)
        fig, ax = plt.subplots()
        ax.plot([e for e in range(0,len(accuracies))], accuracies)
        ax.set(title = "accuracy over trainings")

        # giving a title to my graph
        st.pyplot(fig)

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








