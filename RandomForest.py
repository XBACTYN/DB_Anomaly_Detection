import json
import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import time
import itertools
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score

system_random = random.SystemRandom()
# source_Xy_train = 'data\\train\\GENERATED_transaction_vectors_train_1.json'
# source_Xy_test = 'data\\test\\GENERATED_transaction_vectors_test_1.json'
#source_Xy_train = 'data\\train\\GENERATED_classic_vectors_valid.json'   #110069
#source_Xy_train = 'data\\test\\GENERATED_classic_vectors_test.json'      # 11017
#source_Xy_train = 'data\\train\\GENERATED_classic_vectors_train.json'    #110069

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
    #print(transactions_list)
    print('unpacked')
    return X,y,transactions_list

def make_anomalies(y_test,percent):
    list1 = [1,2,3,4,5,6,7,8,9,10,11]
    count = int(len(y_test)*percent/100)
    print(f'Будет заменено {count}  первых значений ролей из {len(y_test)}, процент измененных {percent}.')
    y_test_anomalies = y_test.copy()
    for i in range(0,count):
        list2 = list1.copy()
        list2.remove(y_test_anomalies[i])
        y_test_anomalies[i] = system_random.choice(list2)
    return y_test_anomalies,count

def plot_confusion_matrix(cm, str,classes,normalize = False,
                          title='Confusion matrix',
                          cmap='OrRd'): # can change color
    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # Label the plot
    for i, j in itertools.product(range(cm.shape[0]),  range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
             fontsize = 20,
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
        plt.grid(None)
        plt.tight_layout()
        plt.ylabel(f'{str} label', size = 18)
        plt.xlabel('Predicted label', size = 18)
    plt.savefig(f'{str}.jpg')

def RandomForest(X_train,y_train,X_test,y_test_anomalies,y_test,count):
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
    cm = confusion_matrix(y_test, y_pred)
    cm2 = confusion_matrix(y_test_anomalies[:count],y_pred[:count])
    plot_confusion_matrix(cm,'True', classes=[1, 2,3,4,5,6,7,8,9,10,11],title='Confusion Matrix')
    plot_confusion_matrix(cm2, 'Anomalies',classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11], title='Confusion Matrix')
    plt.show()
    # param_grid = {
    #     'n_estimators': [25, 50, 100, 150],
    #     'max_features': ['sqrt', 'log2', None],
    #     'max_depth': [3, 6, 9],
    #     'max_leaf_nodes': [3, 6, 9],
    # }
    #
    # grid_search = GridSearchCV(RandomForestClassifier(),
    #                            param_grid=param_grid)
    # grid_search.fit(X_train, y_train)
    # print(grid_search.best_estimator_)

X_train,y_train,transactions = unpack_data(source_Xy_train)
print(len(transactions))
# print(len(X_train[0]))
# print(len(X_train))
# lst = list(zip(X_train,y_train))
# random.shuffle(lst)
# X_train,y_train = zip(*lst)
# X_test, y_test = unpack_data(source_Xy_test)
# y_test_anomalies,count = make_anomalies(y_test,15)



# RandomForest(X_train,y_train,X_test,y_test_anomalies,y_test,count)
