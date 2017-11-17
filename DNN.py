
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn import metrics

import category_encoders as en
from sklearn.pipeline import Pipeline, make_pipeline, make_union
from datetime import datetime
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import keras
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Reshape
from keras.layers import Input, concatenate, BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
from keras.wrappers.scikit_learn import KerasClassifier


# In[2]:


def split_features(X):
    X_list = []
    
    cols = [user_feats, acc_feats, invest_feats, misc_feats, percent_feats, balance_feats,
                      new_feats, C_prev, D_prev, all_credit, all_debit, custint_credit, custint_debit,
                      cashwithdraw, cashdeposit, cnr, bal, eop, cr_amb, atm_amt]
    
    for col in cols:
        X_list.append(X[col].values)
    
    return X_list


# In[3]:


class roc_auc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(split_features(self.x), verbose=0)
        roc = roc_auc_score(self.y, y_pred)
        logs['roc_auc'] = roc_auc_score(self.y, y_pred)
        logs['norm_gini'] = ( roc_auc_score(self.y, y_pred) * 2 ) - 1

        y_pred_val = self.model.predict(split_features(self.x_val), verbose=0)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        logs['roc_auc_val'] = roc_auc_score(self.y_val, y_pred_val)
        logs['norm_gini_val'] = ( roc_auc_score(self.y_val, y_pred_val) * 2 ) - 1

        print('\rroc_auc: %s - roc_auc_val: %s - norm_gini: %s - norm_gini_val: %s' % (str(round(roc,5)),str(round(roc_val,5)),str(round((roc*2-1),5)),str(round((roc_val*2-1),5))), end=10*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


# In[4]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))

def scale_data(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


# In[5]:


# train and test data path
DATA_TRAIN_PATH = '../utility/train_wfeats3.csv'
DATA_TEST_PATH = '../utility/test_wfeats3.csv'

def load_data(path_train=DATA_TRAIN_PATH, path_test=DATA_TEST_PATH):
    train_loader = pd.read_csv(path_train, dtype={'Responders': np.int8, 'UCIC_ID': np.int32}, compression="gzip")
    train = train_loader.drop(['Responders', 'UCIC_ID'], axis=1)
    train_labels = train_loader['Responders'].values
    train_ids = train_loader['UCIC_ID'].values
    print('\n Shape of raw train data:', train.shape)

    test_loader = pd.read_csv(path_test, dtype={'UCIC_ID': np.int32}, compression="gzip")
    test = test_loader.drop(['UCIC_ID'], axis=1)
    test_ids = test_loader['UCIC_ID'].values
    print(' Shape of raw test data:', test.shape)

    return train, train_labels, test, train_ids, test_ids


# In[6]:


folds = 5
runs = 2

cv_LL = 0
cv_AUC = 0
cv_gini = 0
fpred = []
avpred = []
avreal = []
avids = []


# In[7]:


train, target, test, tr_ids, te_ids = load_data()

#Lets drop calc columns (why?)
usefull_cols = [col for col in train.columns if train[col].var() > 0.001]
train = train[usefull_cols].fillna(0)
test  = test[usefull_cols].fillna(0)
from sklearn.preprocessing import QuantileTransformer
qnt = QuantileTransformer(output_distribution='normal')
train = pd.DataFrame(qnt.fit_transform(train), columns=train.columns)
test = pd.DataFrame(qnt.transform(test), columns=train.columns)
n_train = train.shape[0]

#Lets treat -1 as separate label and convert categoricals to one hot encoding for neural network
#cat_cols = [col for col in train.columns if 'cat' in col]
#not_cat_cols = [col for col in train.columns if col not in cat_cols]

#train_test = pd.concat((train, test)).reset_index(drop=True)

#train_test_noncat = train_test[not_cat_cols]
#train_test_cat = train_test[cat_cols]

#First lets do label encoding to get rid of -1 (causes problems with one hot encoding)
#for col in cat_cols:
#    lb = LabelEncoder()
#    lb.fit(train_test_cat[col]) #Fit on all data 
#    train_test_cat[col] = lb.transform(train_test_cat[col]) + 1
    #train[col] = lb.transform(train[col]) + 1  #Keras embedding layer needs labels starting 1
    #test[col]  = lb.transform(test[col]) + 1   #Keras embedding layer needs labels starting 1
    
#print("Unique values in each of caterical column", train[cat_cols].apply(lambda x: x.nunique()))

#train_test_noncat = train_test[not_cat_cols]
#train_test_noncat_scaled, scaler = scale_data(train_test_noncat)

#train_test_noncat_scaled = pd.DataFrame(train_test_noncat_scaled, columns=not_cat_cols)

## After scaling and label encoding
#train_test = pd.concat([train_test_noncat_scaled,train_test_cat], axis=1)

#train = train_test.loc[:(n_train-1), :]
#test = train_test.loc[n_train:, :]

#train_test_scaled, scaler = scale_data(train_test)

#train = train_test_scaled[:n_train, :]
#test = train_test_scaled[n_train:, :]

#n_train = train.shape[0]
#train_test = pd.concat((train, test)).reset_index(drop=True)
#col_to_drop = train.columns[train.columns.str.endswith('_cat')]
#col_to_dummify = train.columns[train.columns.str.endswith('_cat')].astype(str).tolist()

#for col in col_to_dummify:
#    dummy = pd.get_dummies(train_test[col].astype('category'))
#    columns = dummy.columns.astype(str).tolist()
#    columns = [col + '_' + w for w in columns]
#    dummy.columns = columns
#    train_test = pd.concat((train_test, dummy), axis=1)

#train_test.drop(col_to_dummify, axis=1, inplace=True)
#train_test_scaled, scaler = scale_data(train_test)
#train = train_test_scaled[:n_train, :]
#test = train_test_scaled[n_train:, :]
print('\n Shape of processed train data:', train.shape)
print(' Shape of processed test data:', test.shape)


# In[8]:


train.columns.values


# In[9]:


user_feats = ['NO_OF_Accs', 'vintage', 'dependents', 'zip', 'HNW_CATEGORY',
       'EMAIL_UNSUBSCRIBE', 'OCCUP_ALL_NEW', 'city', 'gender_bin', 'Billpay_Active_PrevQ1',
       'Billpay_Reg_ason_Prev1', 'brn_code', 'age', 'ENGAGEMENT_TAG_prev1', 'EFT_SELF_TRANSFER_PrevQ1']
print(len(user_feats))

invest_feats = ['FD_AMOUNT_BOOK_PrevQ1',
       'FD_AMOUNT_BOOK_PrevQ2', 'NO_OF_FD_BOOK_PrevQ1',
       'NO_OF_FD_BOOK_PrevQ2', 'NO_OF_RD_BOOK_PrevQ1',
       'NO_OF_RD_BOOK_PrevQ2', 'RD_AMOUNT_BOOK_PrevQ1',
       'RD_AMOUNT_BOOK_PrevQ2', 'Total_Invest_in_MF_PrevQ1',
       'Total_Invest_in_MF_PrevQ2', 'count_No_of_MF_PrevQ1',
       'count_No_of_MF_PrevQ2', 'Dmat_Investing_PrevQ1',
       'Dmat_Investing_PrevQ2']
print(len(invest_feats))

acc_feats = ['AL_PREM_CLOSED_PREVQ1',
       'OTHER_LOANS_PREM_CLOSED_PREVQ1', 'RD_PREM_CLOSED_PREVQ1',
       'FD_PREM_CLOSED_PREVQ1', 'AL_Closed_PrevQ1', 'CC_CLOSED_PREVQ1',
       'OTHER_LOANS_Closed_PrevQ1', 'RD_CLOSED_PREVQ1', 'FD_CLOSED_PREVQ1',
       'DEMAT_CLOSED_PREV1YR', 'SEC_ACC_CLOSED_PREV1YR', 'AL_CNC_DATE',
       'AL_DATE', 'BL_DATE', 'CE_DATE', 'CV_DATE', 'GL_DATE', 'LAP_DATE',
       'OTHER_LOANS_DATE', 'PL_DATE', 'TWL_DATE','AL_CNC_TAG_LIVE', 'AL_TAG_LIVE', 'BL_TAG_LIVE', 'CC_TAG_LIVE',
       'CV_TAG_LIVE', 'DEMAT_TAG_LIVE', 'GL_TAG_LIVE', 'HL_TAG_LIVE',
       'SEC_ACC_TAG_LIVE', 'INS_TAG_LIVE', 'MF_TAG_LIVE',
       'OTHER_LOANS_TAG_LIVE', 'PL_TAG_LIVE', 'RD_TAG_LIVE', 'FD_TAG_LIVE',
       'TWL_TAG_LIVE', 'lap_tag_live']
print(len(acc_feats))

misc_feats = ['Charges_PrevQ1',
       'Charges_cnt_PrevQ1', 'NO_OF_COMPLAINTS', 'CASH_WD_AMT_Last6',
       'CASH_WD_CNT_Last6','Recency_of_CR_TXN',
       'Recency_of_DR_TXN', 'Recency_of_IB_TXN', 'Recency_of_ATM_TXN',
       'Recency_of_BRANCH_TXN', 'Recency_of_POS_TXN', 'Recency_of_MB_TXN',
       'Recency_of_Activity','Req_Logged_PrevQ1',
       'Query_Logged_PrevQ1', 'NO_OF_CHEQUE_BOUNCE_V1','RBI_Class_Audit',
             'Req_Resolved_PrevQ1',
       'Query_Resolved_PrevQ1', 'Complaint_Resolved_PrevQ1']
print(len(misc_feats))

percent_feats = ['Percent_Change_in_FT_Bank',
       'Percent_Change_in_FT_outside', 'Percent_Change_in_Self_Txn',
       'Percent_Change_in_Big_Expenses']
print(len(percent_feats))

balance_feats = ['I_AQB_PrevQ1', 'I_AQB_PrevQ2',
       'I_CR_AQB_PrevQ1', 'I_CR_AQB_PrevQ2', 'I_CNR_PrevQ1',
       'I_CNR_PrevQ2', 'I_NRV_PrevQ1', 'I_NRV_PrevQ2',]
print(len(balance_feats))

new_feats = ['FINAL_WORTH_prev1_likelihood', 'zip_likelihood',
       'brn_code_likelihood','CR_mean_drop', 'CR_std_drop',
       'EOP_mean_drop', 'EOP_std_drop', 'CR_sum_drop', 'EOB_rat1',
       'bal_rats', 'EOP_prev1log', 'brn_bal', 'zip_bal', 'bal_decline6',
       'bal_decline5', 'bal_decline5.1', 'bal_decline3', 'bal_decline2',
       'EOB_rat2', 'CR_rat2', 'debits_std', 'debits_mean', 'debit1_rat',
       'EOB_rat1qcut', 'EOB_prev1qcut', 'acc_prem_close',
       'acc_prem_close_sum', 'loan_close_num', 'growth_close_num',
       'loan_flag_sum1', 'loan_flag_sum2', 'FD_amt_diff', 'DM_amt_diff',
       'MF_amt_diff', 'FD_Q1_Q2_diffabs']
print(len(new_feats))
C_prev = train.filter(regex="(^C_prev?|^count_C_prev?)").columns
D_prev = train.filter(regex="(^C_prev?|^count_C_prev?)").columns
print(len(C_prev), len(D_prev))

all_credit = train.filter(regex="^[A-Z_]{2,13}_C_prev?").columns
print(len(all_credit))

all_debit = train.filter(regex="^[A-Z_]{2,13}_D_prev?").columns
print(len(all_debit))

custint_credit = train.filter(regex="custinit_C").columns
print(len(custint_credit))

custint_debit = train.filter(regex="custinit_D").columns
print(len(custint_debit))

cashwithdraw = train.filter(regex="_CW_").columns
print(len(cashwithdraw))

cashdeposit = train.filter(regex="CASH_Dep*").columns
print(len(cashdeposit))

cnr = train.filter(regex="CNR_prev?").columns
print(len(cnr))

bal = train.filter(regex="BAL_prev?").columns
print(len(bal))

eop = train.filter(regex="EOP_prev[0-6]$").columns
print(len(eop))

cr_amb = train.filter(regex="CR_AMB_?").columns
print(len(cr_amb))

atm_amt = train.filter(regex='ATM_amt_').columns
print(len(atm_amt))


# In[10]:


train = train.fillna(0)
test = test.fillna(0)


# In[ ]:


from tqdm import tqdm, tqdm_notebook
tqdm_notebook()


# In[ ]:


patience = 10
batchsize = 128

skf = StratifiedKFold(n_splits=folds, random_state=5)
starttime = timer(None)
for i, (train_index, test_index) in enumerate(skf.split(train, target)):
    start_time = timer(None)
    X_train, X_val = train.iloc[train_index,:], train.iloc[test_index,:]
    y_train, y_val = target[train_index], target[test_index]
    train_ids, val_ids = tr_ids[train_index], tr_ids[test_index]
    
    #Compile
    def baseline_model():
        #Define our model
        input_userf = Input(shape= (15,), name='user_feats')
        em_userf = Dense(10, activation='relu')(input_userf)
        
        input_invest = Input(shape= (14,), name='invest_feats')
        em_invest = Dense(10, activation='relu')(input_invest)
        
        input_acc = Input(shape= (38,), name='acc_feats')
        em_acc = Dense(6, activation='relu')(input_acc)

        input_misc = Input(shape= (20,), name='misc_feats')
        em_misc = Dense(3, activation='relu')(input_misc)
    
        input_pct = Input(shape= (4,), name='percent_feats')
        em_pct = Dense(3, activation='relu')(input_pct)

        input_bal = Input(shape= (8,), name='balace_feats')
        em_bal = Dense(4, activation='relu')(input_bal)
        
        input_ind02 = Input(shape= (12,), name='C_prev')
        em_ind02 = Dense(20, activation='relu')(input_ind02)
        #em_ind02 = Reshape(target_shape=(3,))(em_ind02)

        input_ind04 = Input(shape= (12,), name='D_prev')
        em_ind04 = Dense(20, activation='relu')(input_ind04)
        #em_ind04 = Reshape(target_shape=(3,))(em_ind04)

        input_ind05 = Input(shape= (24,), name='All_channel_credit')
        em_ind05 = Dense(20, activation='relu')(input_ind05)
        #em_ind05 = Reshape(target_shape=(10,))(em_ind05)

        input_car01 = Input(shape= (54,), name='All_channel_debit')
        em_car01 = Dense(20, activation='relu')(input_car01)
        #em_car01 = Reshape(target_shape=(10,))(em_car01)


        input_car04 = Input(shape= (15,), name='custinit_credit')
        em_car04 = Dense(3, activation='relu')(input_car04)
        #em_car04 = Reshape(target_shape=(3,))(em_car04)

        input_car05 = Input(shape= (13,), name='custinit_debit')
        em_car05 = Dense(3, activation='relu')(input_car05)
        #em_car05 = Reshape(target_shape=(3,))(em_car05)

        input_car06 = Input(shape= (24,), name='cashwithdraw')
        em_car06 = Dense(6, activation='relu')(input_car06)
        #em_car06 = Reshape(target_shape=(6,))(em_car06)

        input_car07 = Input(shape= (12,), name='cashdeposit')
        em_car07 = Dense(4, activation='relu')(input_car07)
        #em_car07 = Reshape(target_shape=(4,))(em_car07)

        input_car08 = Input(shape= (6,), name='cnr')
        em_car08 = Dense(20, activation='relu')(input_car08)
        #em_car08 = Reshape(target_shape=(3,))(em_car08)

        input_car09 = Input(shape= (6,), name='bal')
        em_car09 = Dense(20, activation='relu')(input_car09)
        #em_car09 = Reshape(target_shape=(3,))(em_car09)

        input_car10 = Input(shape= (6,), name='eop')
        em_car10 = Dense(20, activation='relu')(input_car10)
        #em_car10 = Reshape(target_shape=(3,))(em_car10)

        input_car11 = Input(shape= (12,), name='cr_amb')
        em_car11 = Dense(10, activation='relu')(input_car11)
        #em_car11 = Reshape(target_shape=(3,))(em_car11)

        input_car12 = Input(shape= (6,), name='atm_amt')
        em_car12 = Dense(3, activation='relu')(input_car12)
        
        newfeats = Input(shape=(35,), name='new_feats')

        x = concatenate([em_userf, em_acc, em_invest, em_misc, em_pct, em_bal, newfeats, em_ind02,
                         em_ind04, em_ind05, em_car01, em_car04, em_car05,
                        em_car06, em_car07, em_car08, em_car09, em_car10, em_car11, em_car12
                    ])



        #x = BatchNormalization()(x)
        #x = Dropout(rate=0.01)(x)
        #x = Dense(300, activation='relu', kernel_regularizer=regularizers.l2(0.00001))(x)
        x = Dense(200, activation='relu', kernel_initializer="glorot_normal")(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.5)(x)
        
        
        #x = BatchNormalization()(x)
        #x = Dropout(rate=0.01)(x)
        #x = Dense(150, activation='relu', kernel_regularizer=regularizers.l2(0.0))(x)
        x = Dense(100, activation='relu', kernel_initializer="glorot_normal")(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.25)(x)
        

        #x = BatchNormalization()(x)
        #x = Dropout(rate=0.01)(x)
        #x = Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.00))(x)
        #x4 = Dense(5, activation='relu')(x)
        
        x = Dense(50, activation='relu', kernel_initializer="glorot_normal")(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.15)(x)
        
        x = Dense(25, activation='relu', kernel_initializer="glorot_normal")(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=0.1)(x)
        

        main_output = Dense(1, activation='sigmoid', name='main_output')(x)
        
        model = Model(inputs=[input_userf, input_acc, input_invest, input_misc, input_pct, input_bal,
                      newfeats,  input_ind02, input_ind04, input_ind05, input_car01, input_car04,
                      input_car05, input_car06, input_car07, input_car08, input_car09, input_car10,
                      input_car11, input_car12], 
              outputs=main_output)
        
        adam = keras.optimizers.Adam(lr = 0.001, decay=1e-7)
        model.compile(optimizer=adam, loss='binary_crossentropy', metrics = [])
        
        return model


# This is where we repeat the runs for each fold. If you choose runs=1 above, it will run a 
# regular N-fold procedure.

#########
# It is important to leave the call to random seed here, so each run starts with a different seed.
#########

    for run in range(runs):
        print('\n Fold %d - Run %d\n' % ((i + 1), (run + 1)))
        np.random.seed()
        
        callbacks = [
        roc_auc_callback(training_data=(X_train, y_train),validation_data=(X_val, y_val)),  # call this before EarlyStopping
        EarlyStopping(monitor='norm_gini_val', patience=patience, mode='max', verbose=1),
        CSVLogger('keras-5fold-run-01-v1-epochs.log', separator=',', append=False),
        ModelCheckpoint(
                'keras-5fold-run-01-v1-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check',
                monitor='norm_gini_val', mode='max', # mode must be set to max or Keras will be confused
                save_best_only=True,
                verbose=1)
        ]

# The classifier is defined here. Epochs should be be set to a very large number (not 3 like below) which 
# will never be reached anyway because of early stopping. I usually put 5000 there. Because why not.

        nnet = KerasClassifier(
            build_fn=baseline_model,
# Epoch needs to be set to a very large number ; early stopping will prevent it from reaching
#            epochs=5000,
            epochs=25,
            batch_size=batchsize,
            validation_data=(split_features(X_val), y_val),
            verbose=2,
            shuffle=True,
        callbacks = callbacks)
        
        if i > 4:
            fit = nnet.fit(split_features(X_train), y_train)
        
# We want the best saved model - not the last one where the training stopped. So we delete the old 
# model instance and load the model from the last saved checkpoint. Next we predict values both for 
# validation and test data, and create a summary of parameters for each run.

        del nnet
        nnet = load_model('keras-5fold-run-01-v1-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check')
        scores_val_run = nnet.predict(split_features(X_val), verbose=0)
        LL_run = log_loss(y_val, scores_val_run)
        print('\n Fold %d Run %d Log-loss: %.5f' % ((i + 1), (run + 1), LL_run))
        AUC_run = roc_auc_score(y_val, scores_val_run)
        print(' Fold %d Run %d AUC: %.5f' % ((i + 1), (run + 1), AUC_run))
        print(' Fold %d Run %d normalized gini: %.5f' % ((i + 1), (run + 1), AUC_run*2-1))
        y_pred_run = nnet.predict(split_features(test), verbose=0)
        if run > 0:
            scores_val = scores_val + scores_val_run
            y_pred = y_pred + y_pred_run
        else:
            scores_val = scores_val_run
            y_pred = y_pred_run
            
# We average all runs from the same fold and provide a parameter summary for each fold. Unless something 
# is wrong, the numbers printed here should be better than any of the individual runs.

    scores_val = scores_val / runs
    y_pred = y_pred / runs
    LL = log_loss(y_val, scores_val)
    print('\n Fold %d Log-loss: %.5f' % ((i + 1), LL))
    AUC = roc_auc_score(y_val, scores_val)
    print(' Fold %d AUC: %.5f' % ((i + 1), AUC))
    print(' Fold %d normalized gini: %.5f' % ((i + 1), AUC*2-1))
    timer(start_time)
    
# We add up predictions on the test data for each fold. Create out-of-fold predictions for validation data.

    if i > 0:
        fpred = pred + y_pred
        avreal = np.concatenate((avreal, y_val), axis=0)
        avpred = np.concatenate((avpred, scores_val), axis=0)
        avids = np.concatenate((avids, val_ids), axis=0)
    else:
        fpred = y_pred
        avreal = y_val
        avpred = scores_val
        avids = val_ids
    pred = fpred
    cv_LL = cv_LL + LL
    cv_AUC = cv_AUC + AUC
    cv_gini = cv_gini + (AUC*2-1)


# In[ ]:


LL_oof = log_loss(avreal, avpred)
print('\n Average Log-loss: %.5f' % (cv_LL/folds))
print(' Out-of-fold Log-loss: %.5f' % LL_oof)
AUC_oof = roc_auc_score(avreal, avpred)
print('\n Average AUC: %.5f' % (cv_AUC/folds))
print(' Out-of-fold AUC: %.5f' % AUC_oof)
print('\n Average normalized gini: %.5f' % (cv_gini/folds))
print(' Out-of-fold normalized gini: %.5f' % (AUC_oof*2-1))
score = str(round((AUC_oof*2-1), 5))
timer(starttime)
mpred = pred / folds


# In[ ]:


def predict_proba_corr2(preds):
    d0 = 0.5
    d1 = 1 - d0
    r0 = np.mean(preds)
    r1 = 1 - r0
    gamma_0 = r0/d0
    gamma_1 = r1/d1
    return gamma_1*preds/(gamma_1*preds + gamma_0*(1 - preds))

avpred_corr = predict_proba_corr2(avpred[:,0])
mpred_corr = predict_proba_corr2(mpred)


# In[ ]:


def eval_top(y, preds):
    y = np.array(y)
    n = int(len(preds) * 0.2)
    if np.ndim(preds) == 1:
        indices = np.argsort(preds)[::-1][:n]
    else:
        indices = np.argsort(preds[:, 1])[::-1][:n]
    return sum(y[indices])/sum(y)

print(eval_top(avreal, avpred_corr))


# In[ ]:


print('#\n Writing results')
now = datetime.now()
oof_result = pd.DataFrame(avreal, columns=['Responders_real'])
oof_result['Responders'] = avpred_corr
oof_result['UCIC_ID'] = avids
oof_result.sort_values('UCIC_ID', ascending=True, inplace=True)
oof_result = oof_result.set_index('UCIC_ID')
sub_file = 'train_10fold-keras-run-05-v1-oof_' + str(score) + '_' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'
print('\n Writing out-of-fold file:  %s' % sub_file)
oof_result.to_csv(sub_file, index=True, index_label='UCIC_ID')


# In[ ]:


result = pd.DataFrame(mpred_corr, columns=['Responders'])
result['UCIC_ID'] = te_ids
result = result.set_index('UCIC_ID')
print('\n First 10 lines of your 5-fold average prediction:\n')
print(result.head(10))
sub_file = 'submission_10fold-average-keras-run-05-v1_' + str(score) + '_' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'
print('\n Writing submission:  %s' % sub_file)
result.to_csv(sub_file, index=True, index_label='UCIC_ID')



# In[ ]:


#import matplotlib.pyplot as plt
#import seaborn as sns
#get_ipython().magic('matplotlib inline')


# In[ ]:


#plt.figure(figsize=(9,9))
#sns.distplot(result["Responders"])

