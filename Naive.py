import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
X_train_path = 'data\\train\\train_vectors1.csv'
X_test_path = 'data\\test\\test_vectors_anomaly1.csv'
Y_test_true_path = 'data\\test\\test_vectors1.csv'
def unpack_data(path):
    data = pd.read_csv(path, sep=',')
    data.columns = ["transaction", 'role', "query"]
    #print(data)
    X = data['query'].tolist()
    y_symb = data['role'].tolist()
    y_list=list(map(int, y_symb))
    y = np.array(y_list)
    return X,y

def preprocessing_data(S):
    res_arr =[]
    for el in S:
        el = el.replace('[', '')
        el = el.replace(']', '')
        el=  el.split(' ')
        el = [str for str in el if len(str)>0]
        res_arr.append(np.array(list(map(int, el))))
    #print(res_arr[4])
    res = np.vstack(res_arr)
    #print(res)
    return res

def Naive(X_train,y_train,X_test,y_test_anomalies,y_test_true):
    gnb = GaussianNB()
    # y_pred = gnb.fit(X_train, y_train).predict(X_test)
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    total = X_test.shape[0]
    #print(total)
    #print((y_test_true!=y_pred))
    #print(y_pred)
    #print(y_train)
    pred= list(y_pred)
    y_true= list(y_test_true)
    errors=0
    for i, j in zip(pred, y_true):
        if i != j:
            errors += 1

    #errors = (y_test_true != y_pred).sum(axis=0)
    print('From total ',total,'mislabled ',errors)
    #b1=(y_test_anomalies != y_pred).astype(int)

    # anomalies = (y_test_anomalies != y_pred).sum(axis=0)
    #anomalies = b1.sum(axis=0)
    #print('anomalies ',anomalies)
    return y_pred

def RandomForest(X_train,y_train,X_test,y_test_anomalies,y_test_true):
    clf = RandomForestClassifier(n_estimators=300, random_state=0)
    clf.fit(X_train, y_train)
    RandomForestClassifier(...)
    #print(clf.predict([[0, 0, 0, 0]]))
    y_predict = clf.predict(X_test)
    total=X_test.shape[0]
    pred = list(y_predict)
    print(pred)
    y_true = list(y_test_true)
    print(y_true)
    y_anom=list(y_test_anomalies)
    #a1 = (y_predict!=y_test_true).astype(int)
    errors = 0
    anomalies=0
    for i,j in zip(pred,y_true):
        if i!=j:
            errors+=1
    for i, j in zip(pred, y_anom):
        if i != j:
            anomalies += 1

    y = clf.predict(X_train)
    marker=0
    y_copy=list(y)
    for i,j in zip(y_copy,y_true):
        if i!=j:
            marker+=1
    print('From total ', total, 'mislabled ', errors)
    print('From total ', total, 'anomal is ', anomalies)
    print('From total ', total, 'marked ', marker)
    return y_predict
def main():
    X_tr,y_train = unpack_data(X_train_path)
    X_ts, y_test = unpack_data(X_test_path)
    X,y_test_true = unpack_data(Y_test_true_path)
    X_train = preprocessing_data(X_tr)
    X_test = preprocessing_data(X_ts)

    #y_pred = Naive(X_train,y_train,X_test,y_test,y_test_true)
    f_pred = RandomForest(X_train,y_train,X_test,y_test,y_test_true)




main()