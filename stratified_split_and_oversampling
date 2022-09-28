""" Example of encapsulating the oversampling step within the stratified k-fold split
"""


import pandas as pd
import numpy as np
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics


df = pd.read_csv('dataset_end_ver.csv', index_col=[0])

x_columns = ['AGE', 'SEX', 'INF_ANAM', 'STENOK_AN', 'FK_STENOK', 'IBS_POST', 'GB', 'SIM_GIPERT', 'DLIT_AG', 
          'ZSN_A', 'nr_11', 'nr_01', 'nr_02', 'nr_03', 'nr_04', 'nr_07', 'nr_08', 'np_01', 'np_04', 
          'np_05', 'np_07', 'np_08', 'np_09', 'np_10', 'endocr_01', 'endocr_02', 'endocr_03', 
          'zab_leg_01', 'zab_leg_02', 'zab_leg_03', 'zab_leg_04', 'zab_leg_06', 'S_AD_KBRIG', 
          'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT', 'O_L_POST', 'K_SH_POST', 'MP_TP_POST', 
          'SVT_POST', 'GT_POST', 'FIB_G_POST', 'ant_im', 'lat_im', 'inf_im', 'post_im', 'IM_PG_P', 
          'ritm_ecg_p_01', 'ritm_ecg_p_02', 'ritm_ecg_p_04', 'ritm_ecg_p_06', 'ritm_ecg_p_07', 
          'ritm_ecg_p_08', 'n_r_ecg_p_01', 'n_r_ecg_p_02', 'n_r_ecg_p_03', 'n_r_ecg_p_04', 
          'n_r_ecg_p_05', 'n_r_ecg_p_06', 'n_r_ecg_p_08', 'n_r_ecg_p_09', 'n_r_ecg_p_10', 
          'n_p_ecg_p_01', 'n_p_ecg_p_03', 'n_p_ecg_p_04', 'n_p_ecg_p_05', 'n_p_ecg_p_06', 
          'n_p_ecg_p_07', 'n_p_ecg_p_08', 'n_p_ecg_p_09', 'n_p_ecg_p_10', 'n_p_ecg_p_11', 'n_p_ecg_p_12', 
          'fibr_ter_01', 'fibr_ter_02', 'fibr_ter_03', 'fibr_ter_05', 'fibr_ter_06', 'fibr_ter_07', 'fibr_ter_08', 
          'GIPO_K', 'K_BLOOD', 'GIPER_NA', 'NA_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'L_BLOOD', 'ROE', 'TIME_B_S', 
          'NA_KB', 'NOT_NA_KB', 'LID_KB', 'NITR_S', 'LID_S_n', 'B_BLOK_S_n', 'ANT_CA_S_n', 'GEPAR_S_n', 'ASP_S_n', 'TIKL_S_n', 'TRENT_S_n']

y_columns = ['FIBR_PREDS','PREDS_TAH','JELUD_TAH','FIBR_JELUD',
          'A_V_BLOK','OTEK_LANC','RAZRIV','DRESSLER','ZSN','REC_IM','P_IM_STEN',
          'alive', 'cardiogenic shock', 'pulmonary edema','myocardial rupture', 
          'progress of congestive heart failure','thromboembolism', 'asystole', 'ventricular fibrillation']

def create_split(nfolds=5, order=2):
    
    k_fold = IterativeStratification(n_splits=nfolds, order=order)

    splits = list(k_fold.split(X, y))

    fold_splits = np.zeros(df.shape[0]).astype(np.int)

    for i in range(nfolds):
        fold_splits[splits[i][1]] = i

    df['Split'] = fold_splits    

    df_folds = []

    for fold in range(nfolds):

        df_fold = df.copy()
        global train_df
        global test_df
        train_df = df_fold[df_fold.Split != fold].drop('Split', axis=1).reset_index(drop=True)
        
        test_df = df_fold[df_fold.Split == fold].drop('Split', axis=1).reset_index(drop=True)
        
        df_folds.append((train_df, test_df))

    return df_folds
    
create_split(nfolds=5, order=2)

for i, split in enumerate(splits):
    train_df, test_df = split
    full_counts = {}
    for lbl in labels:
        count = train_df[lbl].sum()
        full_counts[lbl] = count

    label_counts = list(zip(full_counts.keys(), full_counts.values()))
    label_counts = np.array(sorted(label_counts, key=lambda x:-x[1]))
    label_counts = pd.DataFrame(label_counts, columns=['label', 'full_count'])
    label_counts.set_index('label', inplace=True)

    label_counts['full_count'] = pd.to_numeric(counts['full_count'])
    total = label_counts['full_count'].sum()
    avg = int(label_counts['full_count'].sum())/len(labels)
    #print(avg)

    def find_sample_ratio(x):
        x = int(x)
        if x >= avg: return 1
        else: return int(np.round(avg / x))

    label_counts['oversampling_ratio'] = label_counts['full_count'].apply(find_sample_ratio)

    def get_sample_ratio(row):
        ratio = 1
        for l in labels:
            r = label_counts.oversampling_ratio.loc[l]
            if r > ratio: ratio = r
        return ratio
        

    rows = train_df.values.tolist()
    print("Starting rows:", len(rows))
    oversampled_rows = [row for row in rows for _ in range(get_sample_ratio(row))]
    print("Oversampled total:",len(oversampled_rows))
    
    train_df = pd.DataFrame(oversampled_rows, columns=train_df.columns)
    
    X_train = train_df[x_columns]
    y_train = train_df[y_columns]
    X_test = test_df[x_columns]
    y_test = test_df[y_columns]
    
    #print(X_train.shape)
    
    # choosing the classifier and transformation method, for example BinaryRelevance
    classifier = BinaryRelevance(
        classifier = RandomForestClassifier(),
        require_dense = [False, True]
    )

    classifier.fit(X_train, y_train)

    y_hat=classifier.predict(X_test)

    # calling the metrics, for example f1 score and hamming loss
    br_f1=metrics.f1_score(y_test, y_hat, average='micro')
    br_hamm=metrics.hamming_loss(y_test,y_hat)
    print('split number:', i)
    print('Binary Relevance F1-score:',round(br_f1,3))
    print('Binary Relevance Hamming Loss:',round(br_hamm,3))
