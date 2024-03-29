import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle

class KNNregresion:
    def __init__(self, k:int):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        result = []
        for obj in X:
            distances = [self.count_distance(obj, x_i) for x_i in self.X_train]
            sorted_ind_dist = np.argsort(distances)
            k_first_indx = sorted_ind_dist[:self.k]
            k_nearests_obj = self.y_train[k_first_indx]
            prediction = np.mean(k_nearests_obj)
            result.append(prediction)
        return np.array(result)

    def predict_with_truth(self, X, thrr=0.1):
        result = []
        for obj in X:
            distances = [self.count_distance(obj, x_i) for x_i in self.X_train]
            sorted_indices = np.argsort(distances)[:self.k]
            k_nearest_responses = self.y_train[sorted_indices]
            dmax = np.argmax(obj)
            dk_max = [np.argmax(self.X_train[idx]) for idx in sorted_indices]
            diff_maxk = [dmax - dk for dk in dk_max]
            weights = [diff_maxk[k] + self.count_distance(obj, self.X_train[idx]) for k, idx in
                       enumerate(sorted_indices)]
            min_weight_idx = np.argmin(weights)
            most_trustworthy_response = k_nearest_responses[min_weight_idx]
            selected_responses = [response for response in k_nearest_responses if
                                  abs(response - most_trustworthy_response) < thrr]
            predicted_value = np.mean(selected_responses)
            result.append(predicted_value)
        return np.array(result)


    def count_distance(self, x, y):
        return np.sqrt(np.sum((x-y)**2))

    def predict_alpha(self, X):
        return round(self.predict(X)[0],2)


class StandardScaler:

    def __init__(self):
        self.mean_ = np.array([455.35635456, 366.12527101, 352.49062594, 344.55395252, 336.95385641,
                      333.36918912, 333.21945578, 336.36123774, 341.87669115, 344.25012948,
                      349.70505436, 357.07242353, 362.35481189, 372.03856933])
        self.std_ = np.array([192.24743675, 170.38173051, 159.12612375, 148.35943152, 140.63972472, 127.59722913, 116.76309691,
                     109.24324851, 100.82003821, 94.62715116, 89.15373475, 82.34669862, 77.98647965, 71.65253963])


    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


df = pd.read_csv('DataSet.csv', index_col=1)
df=df.drop('Unnamed: 0', axis = 1)
y = df['a_opt']
X=df.drop('a_opt', axis = 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=100)

normalizer = StandardScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)


knn = KNNregresion(6)
knn.fit(X_train_norm, y_train)
with open('KNNregression_model.pkl', 'wb') as file:
    pickle.dump(knn, file)

#y_pred=knn.predict(X_test_norm)
#mse_test = mean_squared_error(y_test, y_pred)
#print(mse_test)
