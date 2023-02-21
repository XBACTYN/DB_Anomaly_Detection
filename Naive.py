import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
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


def RandomForest(X_train,y_train,X_test,y_test_anomalies,y_test_true):
    clf = RandomForestClassifier(n_estimators=1,max_depth=11)
    #pca = PCA(n_components=20)
    #XPCAreduced = pca.fit_transform(X_train)
    #clf.fit(XPCAreduced, y_train)
    clf.fit(X_train, y_train)

    #XPCAtest = pca.fit_transform(X_train)

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


    print('From total ', total, 'mislabled ', errors)
    print('From total ', total, 'anomal is ', anomalies)
    return y_predict
def main():
    X_tr,y_train = unpack_data(X_train_path)
    X_ts, y_test = unpack_data(X_test_path)
    X,y_test_true = unpack_data(Y_test_true_path)
    X_train = preprocessing_data(X_tr)
    X_test = preprocessing_data(X_ts)

    #naive_y = Naive(X_train,y_train,X_test,y_test,y_test_true)
    f_pred = RandomForest(X_train,y_train,X_test,y_test,y_test_true)




main()