import json
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import time
import itertools
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm


system_random = random.SystemRandom()
# source_Xy_train = 'data\\train\\GENERATED_transaction_vectors_train_1.json'
# source_Xy_test = 'data\\test\\GENERATED_transaction_vectors_test_1.json'
#source_Xy_train = 'data\\train\\GENERATED_classic_vectors_valid.json'
source_Xy_test = 'data\\test\\GENERATED_classic_vectors_test.json'
source_Xy_train = 'data\\train\\GENERATED_classic_vectors_train.json'
source_Xy_test_1 = 'data\\test\\GENERATED_transaction_vectors_test.json'
source_Xy_train_1 = 'data\\train\\GENERATED_transaction_vectors_train.json'
result_path = 'results\\'
classes = [1,2,3,4,5,6,7,8,9,10,11]
def unpack_data(path):
    transactions_list = []
    role_list = []
    query_list = []
    f = open(path, 'r', encoding='utf-8')
    data = json.load(f)
    for i in range(0,len(data)):
        transactions_list.append(data[i].get('transaction'))
        role_list.append(data[i].get('role'))
        query_list.append(np.array(data[i].get('query')))
    X = np.array(query_list)
    y = np.array(role_list)
    print('unpacked')
    return X,y#,transactions_list

def make_anomalies(y_test,percent):

    count = int(len(y_test)*percent/100)
    print(f'Будет заменено {count}  первых значений ролей из {len(y_test)}, процент измененных {percent}.')
    y_test_anomalies = y_test.copy()
    for i in range(0,count):
        list2 = classes.copy()
        list2.remove(y_test_anomalies[i])
        y_test_anomalies[i] = system_random.choice(list2)
    return y_test_anomalies,count


def plot_confusion_matrix(y_true,y_pred,name,vector_type):
    fig, ax = plt.subplots(figsize=(10,10))
    cf_matrix = confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cf_matrix, annot=True, cmap='OrRd')

    ax.set_title(' Матрица ошибок')
    ax.set_xlabel('Предсказанные значения классов')
    ax.set_ylabel(f'{name} значения классов')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)
    #fig.savefig(f'{result_path}{vector_type}_confus_{name}_SVM.png')




def RandomForest(X_train,y_train,X_test,y_test_anomalies,y_test,count,vector_type):
    print('start forest')
    model = RandomForestClassifier()
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = (time.time() - start_time)
    print('время обучения ',end_time)
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = (time.time() - start_time)
    print('время предсказания ', end_time)

    # performance evaluatio metrics
    print(classification_report(y_pred, y_test))
    anomalies_count = count
    print(y_pred[:count])
    print(y_test_anomalies[:count])
    diff_values = sum(el1 !=el2 for el1,el2 in zip(y_pred[:count],y_test_anomalies[:count]))
    print(diff_values)
    detection_rate = (diff_values/anomalies_count)*100
    print('anomalies count ',anomalies_count)
    print('detection rate ',detection_rate)

    accuracy_score(y_test, y_pred)
    print(f"The accuracy of the model is {round(accuracy_score(y_test, y_pred), 3) * 100} %")
    plot_confusion_matrix(y_test,y_pred,'Истинные',vector_type)
    plot_confusion_matrix(y_test_anomalies[:count],y_pred[:count],'Аномальные',vector_type)
    plt.show()
    #precision_recall_f1_metrics(y_test,y_pred)

def Gauss(X_train,y_train,X_test,y_test_anomalies,y_test,count,vector_type):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    accuracy_score(y_test, y_pred)

    print(classification_report(y_pred, y_test))
    anomalies_count = count
    print(y_pred[:count])
    print(y_test_anomalies[:count])
    diff_values = sum(el1 != el2 for el1, el2 in zip(y_pred[:count], y_test_anomalies[:count]))
    print(diff_values)
    detection_rate = (diff_values / anomalies_count) * 100
    print('anomalies count ', anomalies_count)
    print('detection rate ', detection_rate)
    plot_confusion_matrix(y_test, y_pred, 'Истинные',vector_type)
    plot_confusion_matrix(y_test_anomalies[:count], y_pred[:count], 'Аномальные',vector_type)
    plt.show()


def SVM(X_train,y_train,X_test,y_test_anomalies,y_test,count,vector_type):
    clf = svm.SVC(decision_function_shape='ovr')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_pred, y_test))
    anomalies_count = count
    print(y_pred[:count])
    print(y_test_anomalies[:count])
    diff_values = sum(el1 != el2 for el1, el2 in zip(y_pred[:count], y_test_anomalies[:count]))
    print(diff_values)
    detection_rate = (diff_values / anomalies_count) * 100
    print('anomalies count ', anomalies_count)
    print('detection rate ', detection_rate)
    plot_confusion_matrix(y_test, y_pred, 'Истинные', vector_type)
    plot_confusion_matrix(y_test_anomalies[:count], y_pred[:count], 'Аномальные', vector_type)
    plt.show()

X_train,y_train = unpack_data(source_Xy_train_1)
X_test, y_test = unpack_data(source_Xy_test_1)
y_test_anomalies,count = make_anomalies(y_test,25)


#RandomForest(X_train,y_train,X_test,y_test_anomalies,y_test,count,'quiplet')
#RandomForest(X_train,y_train,X_test,y_test_anomalies,y_test,count,'hexplet')
#Gauss(X_train,y_train,X_test,y_test_anomalies,y_test,count,'hexplet')
#SVM(X_train,y_train,X_test,y_test_anomalies,y_test,count,'quiplet')