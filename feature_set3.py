
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import category_encoders as en


from sklearn.pipeline import Pipeline, make_pipeline, make_union
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold

import lightgbm as lgb

from skopt import BayesSearchCV, gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt import dump

import pickle
import time
import os

def eval_top(y, preds):
    y = y.values
    n = int(len(preds) * 0.2)
    indices = np.argsort(preds)[::-1][:n]
    #print(len(indices))
    return sum(y[indices])/sum(y)


train = pd.read_csv("../data/train.csv", low_memory=False)
test = pd.read_csv("../data/test.csv", low_memory=False)

cat_cols = list(train.select_dtypes(['object']).columns.values)
for col in cat_cols:
    lb = LabelEncoder()
    lb.fit(pd.concat([train[col], test[col]]).astype(str).fillna('-1'))
    train[col] = lb.transform(train[col].astype(str).fillna('-1'))
    test[col] = lb.transform(test[col].astype(str).fillna('-1'))


num_feats = [f for f in train.columns if f not in ['UCIC_ID','Responders'] + list(cat_cols)]


train[num_feats] = train[num_feats].fillna(0)
test[num_feats] = test[num_feats].fillna(0)

X = train[num_feats + cat_cols]
y = train['Responders']
X_test = test[num_feats + cat_cols]

cvlist = list(StratifiedKFold(n_splits=5, random_state=5).split(X, y))

#Lets encode all categoricals with likelihood
#Writing cv method completely in pandas
def cv_feat_pd1col(data_df, eval_col, target_col, cvlist, func, thresh=3):
    cv_vals = np.ones(len(data_df)) * np.nan #Initialize
    for tr_index, val_index in cvlist:
        tr_func_dict = data_df.loc[tr_index].groupby(eval_col)[target_col].apply(lambda x: 
                                                    func(x) if len(x) >= thresh  else np.nan).to_dict()
        #print(tr_func_dict)
        cv_vals[val_index] = data_df.loc[val_index, eval_col].apply(lambda row: tr_func_dict[row]
                                                                     if row in tr_func_dict
                                                                     else func(data_df.loc[tr_index, target_col]))
    return cv_vals


#Get likehood features
ltrain = len(train)
ltest  = len(test)
test['Responders'] = np.nan
all_data = pd.concat([train, test]).reset_index(drop=True)
cvtraintest = [[np.arange(ltrain), np.arange(ltrain, ltrain + ltest)]]


for col in ['HNW_CATEGORY', 'OCCUP_ALL_NEW', 'city', 'FINAL_WORTH_prev1', 'EFT_SELF_TRANSFER_PrevQ1',
           'Charges_cnt_PrevQ1_N', 'gender_bin', 'RBI_Class_Audit', 'zip', 'brn_code', 'age']:
    print("Processing ", col)
    train[col+'_likelihood'] = cv_feat_pd1col(train, col, 'Responders', cvlist, np.mean, 50)
    test[col+'_likelihood'] = cv_feat_pd1col(all_data, col, 'Responders', cvtraintest, np.mean, 50)[ltrain:]
    
del all_data


train["EOP_prev1_cat"] = pd.qcut(train["EOP_prev1"], q = 50, labels = False, duplicates = "drop")
test["EOP_prev1_cat"] = pd.qcut(test["EOP_prev1"], q = 50, labels = False, duplicates = "drop")

train["age_cat"] = [1 if x <= 18 else 0 for x in train["age"]]
test["age_cat"] = [1 if x <= 18 else 0 for x in test["age"]]

cat_cols.append("EOP_prev1_cat")
cat_cols.append("age_cat")

#Mean of of drops; std in drops
train.loc[:,'EOP_mean_drop'] = train[['EOP_prev1', "EOP_prev2", 'EOP_prev3',
                               'EOP_prev4', "EOP_prev5", 'EOP_prev6',]].apply(
                                                            lambda r: np.mean(r), axis=1)
train.loc[:,'EOP_std_drop'] = train[['EOP_prev1', "EOP_prev2", 'EOP_prev3',
                               'EOP_prev4', "EOP_prev5", 'EOP_prev6',]].apply(
                                                            lambda r: np.std(r), axis=1)

test.loc[:,'EOP_mean_drop'] = test[['EOP_prev1', "EOP_prev2", 'EOP_prev3',
                               'EOP_prev4', "EOP_prev5", 'EOP_prev6',]].apply(
                                                            lambda r: np.mean(r), axis=1)
test.loc[:,'EOP_std_drop'] = test[['EOP_prev1', "EOP_prev2", 'EOP_prev3',
                               'EOP_prev4', "EOP_prev5", 'EOP_prev6',]].apply(
                                                            lambda r: np.std(r), axis=1)



#Mean of of drops; std in drops
train.loc[:,'CR_mean_drop'] = train[['CR_AMB_Drop_Build_1', 'CR_AMB_Drop_Build_2', 
                               'CR_AMB_Drop_Build_3', 'CR_AMB_Drop_Build_4',
                              'CR_AMB_Drop_Build_5']].apply(lambda r: np.mean(r), axis=1)
train.loc[:,'CR_std_drop'] = train[['CR_AMB_Drop_Build_1', 'CR_AMB_Drop_Build_2', 
                               'CR_AMB_Drop_Build_3', 'CR_AMB_Drop_Build_4',
                              'CR_AMB_Drop_Build_5']].apply(lambda r: np.std(r), axis=1)

test.loc[:,'CR_mean_drop'] = test[['CR_AMB_Drop_Build_1', 'CR_AMB_Drop_Build_2', 
                               'CR_AMB_Drop_Build_3', 'CR_AMB_Drop_Build_4',
                              'CR_AMB_Drop_Build_5']].apply(lambda r: np.mean(r), axis=1)
test.loc[:,'CR_std_drop'] = test[['CR_AMB_Drop_Build_1', 'CR_AMB_Drop_Build_2', 
                               'CR_AMB_Drop_Build_3', 'CR_AMB_Drop_Build_4',
                              'CR_AMB_Drop_Build_5']].apply(lambda r: np.std(r), axis=1)


#Sum of drops
train.loc[:,'CR_sum_drop'] = train[['CR_AMB_Drop_Build_1', 'CR_AMB_Drop_Build_2', 
                               'CR_AMB_Drop_Build_3', 'CR_AMB_Drop_Build_4',
                              'CR_AMB_Drop_Build_5']].apply(lambda r: np.sum(r), axis=1)

test.loc[:,'CR_sum_drop'] = test[['CR_AMB_Drop_Build_1', 'CR_AMB_Drop_Build_2', 
                               'CR_AMB_Drop_Build_3', 'CR_AMB_Drop_Build_4',
                              'CR_AMB_Drop_Build_5']].apply(lambda r: np.sum(r), axis=1)


#Ratio average balance to EOB and CR of last month
train.loc[:,'EOB_rat1'] = train['EOP_prev1']/(1 + train['I_AQB_PrevQ1'])
train.loc[:,'CR_rat1'] = train['CR_AMB_Drop_Build_1']/(1 + train['I_AQB_PrevQ1'])

test.loc[:,'EOB_rat1'] = test['EOP_prev1']/(1 + test['I_AQB_PrevQ1'])
test.loc[:,'CR_rat1'] = test['CR_AMB_Drop_Build_1']/(1 + test['I_AQB_PrevQ1'])


train.loc[:,'EOB_rat2'] = train['EOP_prev1']/(1 + train['I_AQB_PrevQ2'].clip(0, 10**9))
train.loc[:,'CR_rat2'] = train['CR_AMB_Drop_Build_1']/(1 + train['I_AQB_PrevQ2'].clip(0, 10**9))

test.loc[:,'EOB_rat2'] = test['EOP_prev1']/(1 + test['I_AQB_PrevQ2'].clip(0, 10**9))
test.loc[:,'CR_rat2'] = test['CR_AMB_Drop_Build_1']/(1 + test['I_AQB_PrevQ2'].clip(0, 10**9))

#ratio of balances
train.loc[:,'bal_rats'] = train['I_AQB_PrevQ2']/(1 + train['I_AQB_PrevQ1'])
test.loc[:,'bal_rats'] = test['I_AQB_PrevQ2']/(1 + test['I_AQB_PrevQ1'])

#log values
train.loc[:,'EOP_prev1log'] = np.log1p(train['EOP_prev1'].clip(0, 10**9))
test.loc[:,'EOP_prev1log'] = np.log1p(test['EOP_prev1'].clip(0, 10**9))

#Encode branch code and city by average balnces of their customers
train.loc[:,'brn_bal'] = train.groupby('brn_code')['I_AQB_PrevQ1'].transform(np.mean)
test.loc[:,'brn_bal'] = test.groupby('brn_code')['I_AQB_PrevQ1'].transform(np.mean)

train.loc[:,'zip_bal'] = train.groupby('zip')['I_AQB_PrevQ1'].transform(np.mean)
test.loc[:,'zip_bal'] = test.groupby('zip')['I_AQB_PrevQ1'].transform(np.mean)

train.loc[:,'brn_churn'] = train.groupby('brn_code')['CR_rat1'].transform(np.mean)
test.loc[:,'brn_churn'] = test.groupby('brn_code')['CR_rat1'].transform(np.mean)

train.loc[:,'zip_churn'] = train.groupby('zip')['CR_rat1'].transform(np.mean)
test.loc[:,'zip_churn'] = test.groupby('zip')['CR_rat1'].transform(np.mean)


#Decline features
train.loc[:,'bal_decline6'] = (train['BAL_prev6'] - train['BAL_prev1'])/(1 + train['I_AQB_PrevQ2'].clip(0, 10**9))
train.loc[:,'bal_decline5'] = (train['BAL_prev5'] - train['BAL_prev1'])/(1 + train['I_AQB_PrevQ2'].clip(0, 10**9))
train.loc[:,'bal_decline4'] = (train['BAL_prev4'] - train['BAL_prev1'])/(1 + train['I_AQB_PrevQ2'].clip(0, 10**9))
train.loc[:,'bal_decline3'] = (train['BAL_prev3'] - train['BAL_prev1'])/(1 + train['I_AQB_PrevQ2'].clip(0, 10**9))
train.loc[:,'bal_decline2'] = (train['BAL_prev2'] - train['BAL_prev1'])/(1 + train['I_AQB_PrevQ2'].clip(0, 10**9))

test.loc[:,'bal_decline6'] = (test['BAL_prev6'] - test['BAL_prev1'])/(1 + test['I_AQB_PrevQ2'].clip(0, 10**9))
test.loc[:,'bal_decline5'] = (test['BAL_prev5'] - test['BAL_prev1'])/(1 + test['I_AQB_PrevQ2'].clip(0, 10**9))
test.loc[:,'bal_decline4'] = (test['BAL_prev4'] - test['BAL_prev1'])/(1 + test['I_AQB_PrevQ2'].clip(0, 10**9))
test.loc[:,'bal_decline3'] = (test['BAL_prev3'] - test['BAL_prev1'])/(1 + test['I_AQB_PrevQ2'].clip(0, 10**9))
test.loc[:,'bal_decline2'] = (test['BAL_prev2'] - test['BAL_prev1'])/(1 + test['I_AQB_PrevQ2'].clip(0, 10**9))


#Mean, std and ratio of debits
train.loc[:,'debits_mean'] = train[['D_prev1', 'D_prev2', 'D_prev3', 'D_prev4',
                              'D_prev5', 'D_prev6']].apply(lambda r: np.mean(r), axis=1) 
train.loc[:,'debits_std'] = train[['D_prev1', 'D_prev2', 'D_prev3', 'D_prev4',
                              'D_prev5', 'D_prev6']].apply(lambda r: np.std(r), axis=1)
train.loc[:,'debit1_rat'] = train['D_prev1']/(1 + train['debits_mean'].clip(0, 10**8))

test.loc[:,'debits_mean'] = test[['D_prev1', 'D_prev2', 'D_prev3', 'D_prev4',
                              'D_prev5', 'D_prev6']].apply(lambda r: np.mean(r), axis=1) 
test.loc[:,'debits_std'] = test[['D_prev1', 'D_prev2', 'D_prev3', 'D_prev4',
                             'D_prev5', 'D_prev6']].apply(lambda r: np.std(r), axis=1)
test.loc[:,'debit1_rat'] = test['D_prev1']/(1 + test['debits_mean'].clip(0, 10**8))


#Likelihood of EOBprev1 ratio and EOBprev1 after splitting them into categories
_, bins1 = pd.qcut(pd.concat([train['EOB_rat1'], test['EOB_rat1']]), 
                   50, duplicates='drop', retbins=True, labels=False)
_, bins2 = pd.qcut(pd.concat([train['EOP_prev1'], test['EOP_prev1']]),
                   50, duplicates='drop', retbins=True, labels=False)

train.loc[:,'EOB_rat1qcut'] = pd.cut(train['EOB_rat1'], bins=bins1, labels=False)
train.loc[:,'EOB_prev1qcut'] = pd.cut(train['EOP_prev1'], bins=bins2, labels=False)

test.loc[:,'EOB_rat1qcut'] = pd.cut(test['EOB_rat1'], bins=bins1, labels=False)
test.loc[:,'EOB_prev1qcut'] = pd.cut(test['EOP_prev1'], bins=bins2, labels=False)

#user acc features
growth_preclose_feats = ['RD_PREM_CLOSED_PREVQ1' ,'FD_PREM_CLOSED_PREVQ1']

#Premature loan closure
loan_preclose_feats = [f for f in train.columns if ('_PREM_CLOSED' in f) & (f not in growth_preclose_feats)]

#Closure of growth accounts in last quarter
growth_close_feats = ['RD_CLOSED_PREVQ1', 'FD_CLOSED_PREVQ1']

#CLosure of growth accounts in last year
growth_close_feats2 = ['DEMAT_CLOSED_PREV1YR', 'SEC_ACC_CLOSED_PREV1YR']

#Closure of loan accounts in last quarter
loan_close_feats = train.filter(regex='(?<!PREM)(_Closed_PrevQ1|_CLOSED_PREVQ1)').columns.tolist() + ['CC_CLOSED_PREVQ1']
print(len(loan_close_feats))

#Live growth accounts
growth_accounts_live = ['DEMAT_TAG_LIVE', 'SEC_ACC_TAG_LIVE', 'MF_TAG_LIVE', 'RD_TAG_LIVE', 'FD_TAG_LIVE']
print(len(growth_accounts_live))

#Live loan accounts
loan_accounts_live = train.filter(regex='(_LIVE|live)').columns.tolist()
loan_accounts_live = [f for f in loan_accounts_live if f not in growth_accounts_live]
print(len(loan_accounts_live))

#Loan flags
loan_flags = train.filter(regex='_DATE$').columns
print(len(loan_flags))

train.loc[:,'acc_prem_close'] = train[loan_preclose_feats].fillna(0).apply(lambda row: np.any(row), axis=1)
test.loc[:,'acc_prem_close'] = test[loan_preclose_feats].fillna(0).apply(lambda row: np.any(row), axis=1)

train.loc[:,'acc_prem_close_sum'] = train[loan_preclose_feats].fillna(0).apply(lambda row: sum(row), axis=1).clip(0,2)
test.loc[:,'acc_prem_close_sum'] = test[loan_preclose_feats].fillna(0).apply(lambda row: sum(row), axis=1).clip(0,2)

train.loc[:,'loan_close_num'] = train[loan_close_feats].fillna(0).apply(lambda row: sum(row), axis=1).clip(0,3)
test.loc[:,'loan_close_num'] = test[loan_close_feats].fillna(0).apply(lambda row: sum(row), axis=1).clip(0,3)

train.loc[:,'growth_close_num'] = train[growth_close_feats].fillna(0).apply(lambda row: sum(row), axis=1)
test.loc[:,'growth_close_num'] = test[growth_close_feats].fillna(0).apply(lambda row: sum(row), axis=1)

train.loc[:,'loan_live'] = train[loan_accounts_live].replace('Y',1).fillna(0).apply(lambda row: sum(row), axis=1).clip(0, 3)
test.loc[:,'loan_live'] = test[loan_accounts_live].replace('Y',1).fillna(0).apply(lambda row: sum(row), axis=1).clip(0, 3)

train[loan_flags] = train[loan_flags].fillna(0)
train.loc[:,'loan_flag_sum1'] = train[loan_flags].apply(lambda row: sum([r==1 for r in row]), 
                                                  axis=1)
train.loc[:,'loan_flag_sum2'] = train[loan_flags].apply(lambda row: sum([r==2 for r in row]), 
                                                  axis=1)

test[loan_flags] = test[loan_flags].fillna(0)
test.loc[:,'loan_flag_sum1'] = test[loan_flags].apply(lambda row: sum([r==1 for r in row]), 
                                                  axis=1)
test.loc[:,'loan_flag_sum2'] = test[loan_flags].apply(lambda row: sum([r==2 for r in row]), 
                                                  axis=1)


#Invest features
train.loc[:,'FD_Q1_Q2_diffabs'] = (train['NO_OF_FD_BOOK_PrevQ1'] - train['NO_OF_FD_BOOK_PrevQ2']).clip(-7,7)
train.loc[:,'NO_OF_FD_BOOK_PrevQ1'] = train['NO_OF_FD_BOOK_PrevQ1'].clip(0, 25)
train.loc[:,'NO_OF_FD_BOOK_PrevQ2'] = train['NO_OF_FD_BOOK_PrevQ2'].clip(0, 25)

test.loc[:,'FD_Q1_Q2_diffabs'] = (test['NO_OF_FD_BOOK_PrevQ1'] - test['NO_OF_FD_BOOK_PrevQ2']).clip(-7,7)
test.loc[:,'NO_OF_FD_BOOK_PrevQ1'] = test['NO_OF_FD_BOOK_PrevQ1'].clip(0, 25)
test.loc[:,'NO_OF_FD_BOOK_PrevQ2'] = test['NO_OF_FD_BOOK_PrevQ2'].clip(0, 25)

train.loc[:,'NO_OF_RD_BOOK_PrevQ1'] = train['NO_OF_RD_BOOK_PrevQ1'].clip(0, 20)
train.loc[:,'NO_OF_RD_BOOK_PrevQ2'] = train['NO_OF_RD_BOOK_PrevQ2'].clip(0, 20)

test.loc[:,'NO_OF_RD_BOOK_PrevQ1'] = test['NO_OF_RD_BOOK_PrevQ1'].clip(0, 20)
test.loc[:,'NO_OF_RD_BOOK_PrevQ2'] = test['NO_OF_RD_BOOK_PrevQ2'].clip(0, 20)

train.loc[:,'FD_amt_diff'] = train['FD_AMOUNT_BOOK_PrevQ1'] - train['FD_AMOUNT_BOOK_PrevQ2']
test.loc[:,'FD_amt_diff'] = test['FD_AMOUNT_BOOK_PrevQ1'] - test['FD_AMOUNT_BOOK_PrevQ2']

train.loc[:,'DM_amt_diff'] = train['Dmat_Investing_PrevQ1'] - train['Dmat_Investing_PrevQ2']
train.loc[:,'MF_amt_diff'] = train['Total_Invest_in_MF_PrevQ1'] - train['Total_Invest_in_MF_PrevQ2']

test.loc[:,'DM_amt_diff'] = test['Dmat_Investing_PrevQ1'] - test['Dmat_Investing_PrevQ2']
test.loc[:,'MF_amt_diff'] = test['Total_Invest_in_MF_PrevQ1'] - test['Total_Invest_in_MF_PrevQ2']

## Credit to Debit ratios

train.loc[:, "C_D_prev1_ratio"] = train["C_prev1"]/(1+train["D_prev1"])
test.loc[:, "C_D_prev1_ratio"] = test["C_prev1"]/(1+test["D_prev1"])

train.loc[:, "C_D_prev2_ratio"] = train["C_prev2"]/(1+ train["D_prev2"])
test.loc[:, "C_D_prev2_ratio"] = test["C_prev2"]/(1+ test["D_prev2"])

train.loc[:, "C_D_prev3_ratio"] = train["C_prev3"]/(1+train["D_prev3"])
test.loc[:, "C_D_prev3_ratio"] = test["C_prev3"]/(1+test["D_prev3"])

train.loc[:, "C_D_prev4_ratio"] = train["C_prev4"]/(1+train["D_prev4"])
test.loc[:, "C_D_prev4_ratio"] = test["C_prev4"]/(1+test["D_prev4"])

train.loc[:, "C_D_prev5_ratio"] = train["C_prev5"]/(1+train["D_prev5"])
test.loc[:, "C_D_prev5_ratio"] = test["C_prev5"]/(1+test["D_prev5"])

train.loc[:, "C_D_prev6_ratio"] = train["C_prev6"]/(1+train["D_prev6"])
test.loc[:, "C_D_prev6_ratio"] = test["C_prev6"]/(1+test["D_prev6"])


time_periods = ["prev1", "prev2", "prev3", "prev4", "prev5", "prev6"]
useful_channel = ["CNR_", "BAL_", "EOP_"]
useful_cr_amb = ["CR_AMB_Prev1", "CR_AMB_Prev2", "CR_AMB_Prev3", "CR_AMB_Prev4", "CR_AMB_Prev5", "CR_AMB_Prev6"]
diff_channel_C_columns = ["count_C_", "COUNT_BRANCH_C_", "custinit_CR_cnt_"]
diff_channel_D_columns = ["count_D_", "COUNT_ATM_D_", "COUNT_BRANCH_D_", "COUNT_IB_D_", "custinit_DR_cnt_"]
useful_C_channel = ["BRANCH_C_", "custinit_CR_amt_"]
useful_D_channel = ["ATM_D_", "BRANCH_D_", "IB_D_", "POS_D_", "custinit_DR_amt_"]

new_feats_imp = []
for d in useful_channel:
    cols = []
    for t in time_periods:
        cols.append(d+t)
    a = "Total_"+d
    train.loc[:, a] = np.average(train[cols], axis = 1, weights=[0.5, 0.25, 0.1, 0.1, 0.05 ,0])
    test.loc[:, a] = np.average(test[cols], axis = 1, weights=[0.5, 0.25, 0.1, 0.1, 0.05 ,0])
    new_feats_imp.append(a)

new_feats_imp.append("Total_CR_AMB_")
train.loc[:, "Total_CR_AMB_"] = np.average(train[useful_cr_amb], axis = 1, weights=[0.5, 0.25, 0.1, 0.1, 0.05 ,0])
test.loc[:, "Total_CR_AMB_"] = np.average(test[useful_cr_amb], axis = 1, weights=[0.5, 0.25, 0.1, 0.1, 0.05 ,0])

new_feats_cnt = []
for d in diff_channel_C_columns:
    cols = []
    for t in time_periods:
        cols.append(d+t)
    a = "Total_"+d
    train.loc[:, a] = np.average(train[cols], axis = 1, weights=[0.3, 0.25, 0.2, 0.15, 0.1 ,0.1])
    test.loc[:, a] = np.average(test[cols], axis = 1, weights=[0.3, 0.25, 0.2, 0.15, 0.1 ,0.1])
    new_feats_cnt.append(a)
for d in diff_channel_D_columns:
    cols = []
    for t in time_periods:
        cols.append(d+t)
    a = "Total_"+d
    train.loc[:, a] = np.average(train[cols], axis = 1, weights=[0.3, 0.25, 0.2, 0.15, 0.1 ,0.1])
    test.loc[:, a] = np.average(test[cols], axis = 1, weights=[0.3, 0.25, 0.2, 0.15, 0.1 ,0.1])
    new_feats_cnt.append(a)

new_feats_amt = []
for d in useful_C_channel:
    cols = []
    for t in time_periods:
        cols.append(d+t)
    a = "Total_"+d
    train.loc[:, a] = np.average(train[cols], axis = 1, weights=[0.3, 0.25, 0.2, 0.15, 0.1 ,0.1])
    test.loc[:, a] = np.average(test[cols], axis = 1, weights=[0.3, 0.25, 0.2, 0.15, 0.1 ,0.1])
    new_feats_amt.append(a)
for d in useful_C_channel:
    cols = []
    for t in time_periods:
        cols.append(d+t)
    a = "Total_"+d
    train.loc[:, a] = np.average(train[cols], axis = 1, weights=[0.3, 0.25, 0.2, 0.15, 0.1 ,0.1])
    test.loc[:, a] = np.average(test[cols], axis = 1, weights=[0.3, 0.25, 0.2, 0.15, 0.1 ,0.1])
    new_feats_amt.append(a)



likeli_feats = [f for f in train.columns if '_likelihood' in f]
new_feats = ['CR_mean_drop', 'CR_std_drop', 'EOP_mean_drop', 'EOP_std_drop', 'CR_sum_drop',
            'EOB_rat1', 'CR_rat1', 'bal_rats', 'EOP_prev1log', 'brn_bal', 'zip_bal',
            'brn_churn', 'zip_churn','bal_decline6', 'bal_decline5', 'bal_decline5',
            'bal_decline3','bal_decline2', 'EOB_rat2', 'CR_rat2', 'debits_std', 
             'debits_mean', 'debit1_rat', 'EOB_rat1qcut', 'EOB_prev1qcut',
            'acc_prem_close', 'acc_prem_close_sum', 'loan_close_num', 'growth_close_num',
            'loan_live', 'loan_flag_sum1', 'loan_flag_sum2',
            'FD_amt_diff', 'DM_amt_diff', 'MF_amt_diff',  'FD_Q1_Q2_diffabs', "C_D_prev1_ratio",
            "C_D_prev2_ratio", "C_D_prev3_ratio", "C_D_prev4_ratio" ,"C_D_prev5_ratio", "C_D_prev6_ratio"]
all_feats = num_feats + cat_cols + likeli_feats + new_feats + new_feats_imp + new_feats_cnt + new_feats_amt
X = train[all_feats]
X_test = test[all_feats]
y = train["Responders"]


train[all_feats + ['Responders','UCIC_ID']].to_csv('../utility/train_wfeats3.csv', index=False, compression='gzip')
test[all_feats + ['Responders', 'UCIC_ID']].to_csv('../utility/test_wfeats3.csv', index=False, compression='gzip')




# In[ ]:


#lgb_params = lgb_params = {
#    'learning_rate': 0.018,
#    'max_depth': -1,
#    'num_leaves': 255,
#    'n_estimators': 500,
    #'min_child_weight': 11,
    #'min_child_samples': 200,
#    'subsample':0.95,
#    'colsample_bytree':0.6,
    #'min_sum_hessian_in_leaf':20,
    #'reg_lambda': 1,
    #'reg_alpha':1,
    #'is_unbalance':True,
    #'verbose':1,
#}
#lgb_preds = cross_val_predict(lgb.LGBMClassifier(**lgb_params), X, y, cv=cvlist, method='predict_proba_corr', verbose=10)


# In[ ]:


#eval_top(y, lgb_preds)


# In[ ]:


#preds_df = pd.DataFrame({'UCIC_ID':train['UCIC_ID'], 'Responders':lgb_preds})
#preds_df.head()


# In[ ]:


#lgb_preds_test = lgb.LGBMClassifier(**lgb_params).set_params(n_estimators=550).fit(X,y).predict_proba_corr(X_test)


# In[ ]:


#sns.distplot(lgb_preds)
#sns.distplot(lgb_preds_test)
#plt.show()


# In[ ]:


#preds_test_df = pd.DataFrame({'UCIC_ID':test['UCIC_ID'].values, 'Responders':lgb_preds_test})
#preds_test_df.head()


# In[ ]:


#preds_df.to_csv("../utility/oof_lgb_feat3_v1.csv", index=False)
#preds_test_df.to_csv("../utility/test_lgb_feat3_v1.csv", index=False)

