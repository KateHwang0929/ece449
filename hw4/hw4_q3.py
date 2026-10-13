import hw4_utils
import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def cross_validate_model(X,y,model,k_folds=5):
    errors=[]
    for train,val in KFold(n_splits=k_folds,shuffle=True,random_state=42).split(X):
        model_copy=deepcopy(model); model_copy.fit(X[train],y[train]); errors.append(mean_squared_error(y[val],model_copy.predict(X[val])))
    return np.mean(errors),np.std(errors)

def evaluate_polynomial_degrees(X,y):
    mse_list=[]
    for d in range(2,9):
        model=hw4_utils.train_polynomial_model(X,y,d); mean_mse,std_mse=cross_validate_model(X,y,model,k_folds=5); mse_list.append(mean_mse)
    return mse_list

if __name__=='__main__':
    X,y=hw4_utils.polynomial_data(); mse_list=evaluate_polynomial_degrees(X,y); hw4_utils.plot_mse(list(range(2,9)),mse_list); print('Saved mse.png. Include the plot and your discussion in the report.')
