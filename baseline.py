import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import math
from sklearn.metrics import mean_squared_error
import numpy as np
import os


def GetData(path):
    # type = {'Closing Price':np.float32, '': np.int32}
    # df_train = pd.read_csv(path, dtype=type)
    for name in ['train_v1', 'train_y', 'valid_v1', 'valid_y', 'test_v1']:
        if os.path.exists(path+name+'.csv'):
            continue
        else:
            df_train = pd.read_csv(path + '/train.csv')
            df_test = pd.read_csv(path + '/test.csv')
            df_train.pop('ID')
            df_test.pop('ID')
            li_valid = df_train.loc[df_train['Date'].str.startswith('2009-11', na=False)].index.tolist()
            df_valid = df_train.loc[df_train['Date'].str.startswith('2009-11', na=False)]
            df_train = df_train.drop(li_valid)
            df_train['label'] = 1
            df_valid['label'] = 0
            df_test['label'] = -1
            data = pd.concat([df_train, df_valid, df_test])
            data = label_encoding(data)
            df_train = data[data.label == 1]
            df_train_y = df_train['Closing Price']
            #df_train_y = df_train.pop('Closing Price')
            df_valid = data[data.label == 0]
            df_valid_y = df_valid['Closing Price']
            # df_valid_y = df_valid.pop('Closing Price')
            df_test = data[data.label == -1]
            del df_train['label'], df_valid['label'], df_test['label']
            del df_test['Opening Price'], df_test['Last Closing Price'], \
                df_test['Highest Price'], df_test['Lowest Price'], df_test['Closing Price']
            # l_code = df_train['Stock Code'].unique()
            df_train.to_csv('./data/train_v1.csv', index=False)
            df_train_y.to_csv('./data/train_y.csv', index=False)
            df_valid.to_csv('./data/valid_v1.csv', index=False)
            df_valid_y.to_csv('./data/valid_y.csv', index=False)
            df_test.to_csv('./data/test_v1.csv', index=False)
    train_v1 = pd.read_csv(path + 'train_v1.csv')
    train_y = pd.read_csv(path + 'train_y.csv', header=None)
    valid_v1 = pd.read_csv(path + 'valid_v1.csv')
    valid_y = pd.read_csv(path + 'valid_y.csv', header=None)
    test_v1 = pd.read_csv(path + 'test_v1.csv')

    return train_v1, train_y, valid_v1, valid_y, test_v1

def label_encoding(data):
    for feature in ['Date', 'Time']:
        try:
            data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
        except:
            data[feature] = LabelEncoder().fit_transform(data[feature])
    data['Date'] = data['Date'] + 1
    data['Time'] = data['Time'] + 1
    return data

def myLgbEval(y_pred, train_data):
    y_true = train_data.get_label()
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    min20 = y_true[np.argsort(y_true)[:20]]
    max20 = y_true[np.argsort(y_true)[-20:][::-1]]
    pred20 = y_true[np.argsort(y_pred)[-20:][::-1]]
    res = (pred20.mean() - min20.mean()) / (max20.mean() - min20.mean())
    return 'eval', res, True

def lgb_train(train, train_y, valid, valid_y, test):
    """LightGBM"""
    lgb_params = {
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'num_leaves': 61,
        'max_depth': -1,
        'learning_rate': 0.01,
        'verbose': 1,
        'seed': 2018,
    }
    lgb_train = lgb.Dataset(train, train_y)
    lgb_valid = lgb.Dataset(valid, valid_y)
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=10000, verbose_eval=10,
                          valid_sets=lgb_valid, early_stopping_rounds=100)#, feval=myLgbEval
    joblib.dump(lgb_model, 'lgb_train.pkl')
    test['Closing Price'] = lgb_model.predict(test, num_iteration = lgb_model.best_iteration)
    # test['Closing Price'] = test['Closing Price'].apply(lambda x: float('%.2f' % x))
    return test

def lgb_fit(train, train_y, valid, valid_y, test):
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1500,
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=100
    )
    train_y = train_y*100
    valid_y = valid_y*100
    clf.fit(train, train_y.astype('int'), eval_set=[(valid, valid_y.astype('int'))], eval_metric='rmse', early_stopping_rounds=100)
    joblib.dump(clf, 'lgb_fig.pkl')
    res = test
    res['proba'] = clf.predict_proba(test)[:, 1]
    res['predict'] = clf.predict(test)
    return res

def sk_Reg(train, train_y, valid, valid_y, test):
    del train['Opening Price'], train['Last Closing Price'], \
        train['Highest Price'], train['Lowest Price'], \
        valid['Opening Price'], valid['Last Closing Price'], \
        valid['Highest Price'], valid['Lowest Price']
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    estimator = lr
    estimator.fit(train, train_y)
    test['predict'] = estimator.predict(test)
    pred_valid = estimator.predict(valid)

    print('Train RMSE for %s is %f' % (
    'sk_Reg', math.sqrt(mean_squared_error(valid_y, pred_valid))))
    return test

def Keras_Reg(train, train_y, valid, valid_y, test):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor
    def baseline_model():
        # create model
        model = Sequential()
        model.add(Dense(20, input_dim=train.shape[1], kernel_initializer='uniform', activation='softplus'))
        model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
        # Compile model
        model.compile(loss='mse', optimizer='Nadam', metrics=['mse'])
        # model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    estimator = KerasRegressor(build_fn=baseline_model, verbose=1, epochs=5,
                               batch_size=55000)  # verbose: 详细信息模式，0 或者 1
    estimator.fit(train, train_y)
    test['predict'] = estimator.predict(test)
    pred_valid = estimator.predict(valid)
    print('Train RMSE for %s is %f' % (
    'Keras_Reg', math.sqrt(mean_squared_error(valid_y, pred_valid))))
    return test

if __name__ == '__main__':
    # Make sure the current directory is the parent directory of "data" !!!
    train, train_y, valid, valid_y, test = GetData('./data/')
    del train['Last Closing Price'], valid['Last Closing Price']
    # train['Closing Price'] = train_y
    # valid['Closing Price'] = valid_y
    #lgb_train
    pred = lgb_train(train, train_y, valid, valid_y, test)
    #lgb_fit
    #pred = lgb_fit(train, train_y, valid, valid_y, test)
    #sk_Reg
    #pred = sk_Reg(train, train_y, valid, valid_y, test)
    print('Valid RMSE for %s is %f' % (
        'Regression', math.sqrt(mean_squared_error(valid_y, pred))))
    pred.to_csv('./data/pred.csv', index=False)
    print('ok')

