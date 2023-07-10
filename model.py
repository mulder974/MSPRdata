from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import db
from sklearn.preprocessing import LabelEncoder
import streamlit as st



def standardization(x_train,x_test):
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, scaler



def create_model():

    model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
                    hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

    return model

def train_model(model,x_train,y_train):
    model.fit(x_train,y_train)


def predict(input, model):
    y_pred = model.predict(input)
    return y_pred

def mesure_accuracy(y_test,y_pred):
    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

def save_model(model,filename):

    pickle.dump(model, open(filename, 'wb'))



def load_model_from_disk(model_path):
    loaded_model = pickle.load(open(f'{model_path}', 'rb'))
    return loaded_model





#-----------------------------------------------------#

def create_model_logreg():
    clf = LogisticRegression()
    return clf

def create_model_svm():
    clf = SVC()
    return clf

def create_model_rand_for():
    clf = RandomForestClassifier()
    return clf

def create_model_neur_net():
    clf = MLPClassifier()
    return clf




def retrain_all_model():

    #Preprocessing
    le = LabelEncoder()
    df = db.query_data()
    df = df.drop("Unnamed: 0", axis = 1)
    df = df.drop("index", axis = 1)
    df = df.drop("Libellé de la commune", axis = 1)
    df = df.drop("longitude", axis = 1)
    df = df.drop("latitude", axis=1)
    df = df.drop("Geo Point", axis=1)
    df = df.drop("NOM_COM", axis=1)
    df = df.drop("president_gagnant", axis=1)
    df["Age moyen"] = df["Age moyen"].replace(",", ".", regex=True)
    df["Age moyen"] = df["Age moyen"].astype("float")




    X = df.drop("partie" ,axis=1)
    Y = df["partie"]
    Y_encoded = le.fit_transform(Y)

    x_train, x_test, y_train, y_test = train_test_split( X, Y_encoded, test_size=0.33, random_state=42)

    x_train_log_reg, x_test_log_reg = x_train.copy(), x_test.copy()
    x_train_rand_for, x_test_rand_for = x_train.copy(), x_test.copy()
    x_train_neur_net, x_test_neur_net = x_train.copy(), x_test.copy()


    models_list = []

# Here, we will train 3 models ( Logistic regression,  Random Forest, and Neural Network.
# They will be used later to predict with ensemble methods.

    log_reg_clf = create_model_logreg()
    random_forest = create_model_rand_for()
    neural_network = create_model_neur_net()

    models_list.extend(((log_reg_clf,"log_reg"), (random_forest,"rand_for"), (neural_network, "neur_net")))


    st.write("model instances created")

    # Now we apply a grid search in order to find the best parameters to our models

    st.write("Now applying preporcessing then grid search to find best params for each model")
    n= 1

    for model_ in models_list:

        st.write("---------------------Training new model---------------------")
        st.write(f" \n  model N° {n} / {len(models_list)} \n")



        if model_[1] == "log_reg":
            st.write("Model log_reg : \n")
            st.write("preprocessing data : Standardization")

            standardization(x_train_log_reg,x_test_log_reg)

            st.write("Standardisation done, begining grid search")
            params = {'solver': ('newton-cg', 'liblinear'),
                      'max_iter': [e for e in range(5000,15000,1000)]}

            log_reg_clf = GridSearchCV(model_[0], params, cv=5)

            log_reg_clf.fit(x_train_log_reg, y_train)

            st.write(" best estimator : ")
            print(log_reg_clf.best_estimator_)
            st.write(" best params : ")
            st.write(log_reg_clf.best_score_)

            st.write(" scoring on test dataD:s : ")
            st.write(log_reg_clf.score(x_test_log_reg, y_test))

            st.write("saving model")
            (save_model(log_reg_clf, 'log_reg_clf.SAV'))


        if model_[1] == "rand_for":
            st.write("Model rand_for : \n")


            params = {'n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200],
                      'criterion': ['gini', 'entropy', 'log_loss'],
                      'max_depth': [e for e in range(0, 32, 2)]}

            random_forest = GridSearchCV(model_[0], params, cv=5)

            random_forest.fit(x_train_rand_for, y_train)

            st.write(" best estimator : ")
            st.write(random_forest.best_estimator_)
            st.write(" best params : ")
            st.write(random_forest.best_score_)

            st.write(" scoring on test datas : ")
            st.write(random_forest.score(x_test_rand_for, y_test))

            st.write("saving model")
            save_model(random_forest, 'random_forest.SAV')


        if model_[1] == "neur_net":
            st.write("Model neur_net : \n")

            standardization(x_train_neur_net, x_test_neur_net)


            params = {'hidden_layer_sizes': (300, 400, 500),
                      'alpha' : [0.01, 0.001, 0.05, 0.1],
                      'max_iter': [e+1 for e in range(500,2000,250)],
                      'learning_rate': ['constant', 'adaptative'],
                      }

            neural_network = GridSearchCV(model_[0], params, cv=5)

            neural_network.fit(x_train_neur_net, y_train)

            st.write(" best estimator : ")
            st.write(neural_network.best_estimator_)
            st.write(" best params : ")
            st.write(neural_network.best_score_)

            st.write(" scoring on test datas : ")
            st.write(neural_network.score(x_test_neur_net, y_test))

            st.write("saving model")
            save_model(neural_network, 'neural_net.SAV')
        n += 1


#Creating voting classifier ( ensemble )
    st.write("Now creating voting classifier")

    log_reg = load_model_from_disk("model/log_reg_clf.SAV")
    rand_for = load_model_from_disk("model/random_forest.SAV")
    nn = load_model_from_disk("model/neural_net.SAV")

    eclf = VotingClassifier(
        estimators=[('lr', log_reg), ('rf', rand_for), ('nn', nn)],
        voting='soft')

    st.write(cross_val_score(eclf, x_train, y_train))
    st.write("Voting classifier score : ")
    st.write(eclf.score(x_test,y_test))
    save_model(eclf, 'voting_classifier.SAV')




def test_model():

    model = load_model_from_disk("model/svm_clf.SAV")


    x_train, x_test, y_train, y_test = preprocess.load_data_for_model_training()
    y_pred = model.predict(y_train)
    mesure_accuracy(y_test, y_pred)







# x_train,x_test,y_train,y_test=preprocess.load_data_for_model_training()
#
# model = create_model()
# #
# train_model(model, x_train, y_train)
# #
# y_pred = predict(x_test, model)
# #
# mesure_accuracy(y_test,y_pred)
# #
# save_model(model)




