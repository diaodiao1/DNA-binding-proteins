# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 21:19:01 2019

@author: diao
"""

#%%
import re
import itertools
import numpy as np
import collections
from sklearn.externals import joblib

seed = 13
np.random.seed(seed)
from sklearn.metrics import roc_auc_score
from keras.models import load_model
import argparse


def TransDict_from_list(groups):
    tar_list = ['0', '1', '2', '3', '4', '5', '6']
    result = {}
    index = 0
    for group in groups:
        g_members = sorted(group) #Alphabetically sorted list
        for c in g_members:
            result[c] = str(tar_list[index]) #K:V map, use group's first letter as represent.
        index = index + 1
    return result


def translate_sequence (seq, TranslationDict):
#    import string
    from_list = []
    to_list = []
    for k,v in TranslationDict.items():
        from_list.append(k)
        to_list.append(v)
    TRANS_seq = seq.translate(str.maketrans(str(from_list), str(to_list)))
    return TRANS_seq


def get_composition(seq):
    aac=['G','A','V','L','I','F','W','Y','D','N','E','K','Q','M','S','T','C','P','H','R']
    length=len(seq)
    vector=[]
    
    '''
    four-parts composition 
    '''
    one_part=length//4
    dipeptide=[]
    chars = ['0', '1', '2', '3', '4', '5', '6']
    groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
    group_dict = TransDict_from_list(groups)
    iter = itertools.product(chars,repeat=2)
    for dip in list(iter):
        dipeptide.append(''.join(dip))
    #the first part vector        
    one_part_seq=seq[0:one_part]
    one_part_vector=[]
    for str in aac:
        one_part_vector.append(len(re.findall(r'(?='+str+')', one_part_seq))/one_part)
    one_part_seq = translate_sequence(one_part_seq, group_dict)
    for str in chars:
        one_part_vector.append(len(re.findall(r'(?='+str+')', one_part_seq))/one_part)
    for str in dipeptide:
        one_part_vector.append(len(re.findall(r'(?='+str+')', one_part_seq))/(one_part-1))
    vector.extend(one_part_vector)
    #the second part vector
    two_part_seq=seq[:one_part*2]
    two_part_vector=[]
    for str in aac:
        two_part_vector.append(len(re.findall(r'(?='+str+')', two_part_seq))/(one_part*2))
    two_part_seq = translate_sequence(two_part_seq, group_dict)
    for str in chars:
        two_part_vector.append(len(re.findall(r'(?='+str+')', two_part_seq))/(one_part*2))
    for str in dipeptide:
        two_part_vector.append(len(re.findall(r'(?='+str+')', two_part_seq))/(one_part*2 - 1))
    vector.extend(two_part_vector)
    #the third part vector
    three_part_seq=seq[:one_part*3]
    three_part_vector=[]
    for str in aac:
        three_part_vector.append(len(re.findall(r'(?='+str+')', three_part_seq))/(one_part*3))
    three_part_seq = translate_sequence(three_part_seq, group_dict)
    for str in chars:
        three_part_vector.append(len(re.findall(r'(?='+str+')', three_part_seq))/(one_part*3))
    for str in dipeptide:
        three_part_vector.append(len(re.findall(r'(?='+str+')', three_part_seq))/(one_part*3 - 1))
    vector.extend(three_part_vector)
    #the forth paty vector
    four_part_seq=seq
    four_part_vector=[]
    for str in aac:
        four_part_vector.append(len(re.findall(r'(?='+str+')', four_part_seq))/length)
    four_part_seq = translate_sequence(four_part_seq, group_dict)
    for str in chars:
        four_part_vector.append(len(re.findall(r'(?='+str+')', four_part_seq))/length)
    for str in dipeptide:
        four_part_vector.append(len(re.findall(r'(?='+str+')', four_part_seq))/(length-1))
    vector.extend(four_part_vector)
    
    return vector


def use_composition(rbp_dict_pos, rbp_dict_neg):
    pos_nueeric = []
    neg_nueeric = []
    for k, v in rbp_dict_pos.items():
        pos_nueeric.append(get_composition(v))
    for k, v in rbp_dict_neg.items():
        neg_nueeric.append(get_composition(v))
    poslabel=np.ones(len(pos_nueeric))
    neglabel=np.zeros(len(neg_nueeric))
    dataset=np.concatenate((pos_nueeric,neg_nueeric), axis = 0)
    label=np.concatenate((poslabel,neglabel), axis = 0)
    
    return dataset,label


def use_composition_pos(rbp_dict_pos):
    pos_nueeric = []
    for k, v in rbp_dict_pos.items():
        pos_nueeric.append(get_composition(v))
    poslabel=np.ones(len(pos_nueeric))
    return pos_nueeric,poslabel


def use_composition_neg(rbp_dict_neg):
    neg_nueeric = []
    for k, v in rbp_dict_neg.items():
        neg_nueeric.append(get_composition(v))
    poslabel=np.zeros(len(neg_nueeric))
    
    return neg_nueeric,poslabel


def read_fasta_file(fasta_file):
    seq_dict = collections.OrderedDict()
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        if line == '':
            continue
        if line[0] == '>':  # or line.startswith('>')
            name = line[1:].upper()  # discarding the initial >
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line
    fp.close()
    return seq_dict


def read_fasta_file_pos(fasta_file):
    seq_dict = collections.OrderedDict()
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        line = line.rstrip()
        if line == '':
            continue
        if line[0] == '>':  # or line.startswith('>')
            line = line.split('|')
            name = line[1].upper()  # discarding the initial >
            seq_dict[name] = ''
        else:
            seq_dict[name] = seq_dict[name] + line
    fp.close()
    return seq_dict


def calculate_performace(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1        
                              
    accuracy = float(tp + tn)/test_num
    precision = float(tp)/(tp+ fp + 1e-06)
    sensitivity = float(tp)/ (tp + fn + 1e-06)
    specificity = float(tn)/(tn + fp + 1e-06)
    f1_score = float(2*tp)/(2*tp + fp + fn + 1e-06)
    MCC = float(tp*tn-fp*fn)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score


def calculate_performace_DBP(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1        
                              
    sensitivity = float(tp)/ (tp + fn + 1e-06)
    return tp, fn, sensitivity


def calculate_performace_NDBP(test_num, pred_y,  labels):
    tp =0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] ==1:
            if labels[index] == pred_y[index]:
                tp = tp +1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn +1
            else:
                fp = fp + 1        
                              
    FPR = float(fp)/(tn + fp + 1e-06)
    return tn, fp, FPR
#%%

def predict_PDB2272():
    rbp_pos_dict = read_fasta_file('data/PDB2272_P.txt')
    rbp_neg_dict = read_fasta_file('data/PDB2272_N.txt')
    
    com_fea, Y = use_composition(rbp_pos_dict, rbp_neg_dict)
    
    scaler = joblib.load('model/MsDBN_PDB2272_scaler.h5')
    fea = scaler.transform(com_fea)
     
    model = load_model('model/MsDBN_PDB2272_model.h5')
    proba = model.predict(fea)
    pred = (proba > 0.5).astype('int32')
    roc_auc = roc_auc_score(Y, proba)
    print('========================')
    tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = calculate_performace(len(pred), pred, Y)
    print('\ttp=%0.0f,fp=%0.0f,tn=%0.0f,fn=%0.0f'%(tp, fp, tn, fn))
    print('\tacc=%0.2f%%,pre=%0.2f%%,sn=%0.2f%%,sp=%0.2f%%,mcc=%0.2f%%,f1=%0.2f%%, roc_auc = %0.2f%%'
    			  %(accuracy*100, precision*100, sensitivity*100, specificity*100, MCC*100, f1_score*100, roc_auc*100))
    				  	

def predict_new_sample(dataset_P, dataset_N):
    rbp_pos_dict = read_fasta_file(dataset_P)
    rbp_neg_dict = read_fasta_file(dataset_N)
    
    com_fea, Y = use_composition(rbp_pos_dict, rbp_neg_dict)
    scaler = joblib.load('model/MsDBN_scaler.h5')
    fea = scaler.transform(com_fea)
     
    model = load_model('model/MsDBN_model.h5')
    proba = model.predict(fea)
    pred = (proba > 0.5).astype('int32')
    roc_auc = roc_auc_score(Y, proba)
    print('========================')
    tp, fp, tn, fn, accuracy, precision, sensitivity, specificity, MCC, f1_score = calculate_performace(len(pred), pred, Y)
    print('\ttp=%0.0f,fp=%0.0f,tn=%0.0f,fn=%0.0f'%(tp, fp, tn, fn))
    print('\tacc=%0.2f%%,pre=%0.2f%%,sn=%0.2f%%,sp=%0.2f%%,mcc=%0.2f%%,f1=%0.2f%%, roc_auc = %0.2f%%'
    			  %(accuracy*100, precision*100, sensitivity*100, specificity*100, MCC*100, f1_score*100, roc_auc*100))
    print('========================')


def MsDBP(dataset_P, dataset_N):
    if dataset_P == 'PDB2272_P' and dataset_N == 'PDB2272_N':
        predict_PDB2272()
    else:
        predict_new_sample(dataset_P, dataset_N)


def predict_DBP2858():
    rbp_pos_dict_h = read_fasta_file_pos('data/human_id_seq.txt')
    com_fea_h, Y_h = use_composition_pos(rbp_pos_dict_h)

    rbp_pos_dict_t = read_fasta_file_pos('data/thaliana_id_seq.txt')
    com_fea_t, Y_t = use_composition_pos(rbp_pos_dict_t)
    
    rbp_pos_dict_m = read_fasta_file_pos('data/mouse_id_seq.txt')
    com_fea_m, Y_m = use_composition_pos(rbp_pos_dict_m)
    
    rbp_pos_dict_c = read_fasta_file_pos('data/cerevisiae_id_seq.txt')
    com_fea_c, Y_c = use_composition_pos(rbp_pos_dict_c)
    
    rbp_pos_dict_f = read_fasta_file_pos('data/Fruit_fly_id_seq.txt')
    com_fea_f, Y_f = use_composition_pos(rbp_pos_dict_f)
    
    scaler = joblib.load('model/MsDBN_scaler.h5')
    
    fea_h = scaler.transform(com_fea_h)
    fea_t = scaler.transform(com_fea_t)
    fea_m = scaler.transform(com_fea_m)
    fea_c = scaler.transform(com_fea_c)
    fea_f = scaler.transform(com_fea_f)
     
    model = load_model('model/MsDBN_model.h5')
    proba_h = model.predict(fea_h)
    proba_t = model.predict(fea_t)
    proba_m = model.predict(fea_m)
    proba_c = model.predict(fea_c)
    proba_f = model.predict(fea_f)
    
    pred_h = (proba_h > 0.5).astype('int32')
    pred_t = (proba_t > 0.5).astype('int32')
    pred_m = (proba_m > 0.5).astype('int32')
    pred_c = (proba_c > 0.5).astype('int32')
    pred_f = (proba_f > 0.5).astype('int32')
    
    tp_h, fn_h, sensitivity_h = calculate_performace_DBP(len(pred_h), pred_h, Y_h)
    tp_t, fn_t, sensitivity_t = calculate_performace_DBP(len(pred_t), pred_t, Y_t)
    tp_m, fn_m, sensitivity_m = calculate_performace_DBP(len(pred_m), pred_m, Y_m)
    tp_c, fn_c, sensitivity_c = calculate_performace_DBP(len(pred_c), pred_c, Y_c)
    tp_f, fn_f, sensitivity_f = calculate_performace_DBP(len(pred_f), pred_f, Y_f)
    
    print('***************************************')
    print('\ttp_human=%0.0f,fn_human=%0.0f'%(tp_h, fn_h))
    print('\tsn_human=%0.2f'% (sensitivity_h*100))
    print('***************************************')
    print('\ttp_thaliana=%0.0f,fn_thaliana=%0.0f'%(tp_t, fn_t))
    print('\tsn_thaliana=%0.2f'% (sensitivity_t*100))
    print('***************************************')
    print('\ttp_mouse=%0.0f,fn_mouse=%0.0f'%(tp_m, fn_m))
    print('\tsn_mouse=%0.2f'% (sensitivity_m*100))
    print('***************************************')
    print('\ttp_cerevisiae=%0.0f,fn_cerevisiae=%0.0f'%(tp_c, fn_c))
    print('\tsn_cerevisiae=%0.2f'% (sensitivity_c*100))
    print('***************************************')
    print('\ttp_fruit_fly=%0.0f,fn_fruit_fly=%0.0f'%(tp_f, fn_f))
    print('\tsn_fruit_fly=%0.2f'% (sensitivity_f*100))
    print('***************************************')
        

def predict_new(dataset_P):
    rbp_pos_dict = read_fasta_file(dataset_P)
    
    com_fea, Y = use_composition_pos(rbp_pos_dict)
    
    scaler = joblib.load('model/MsDBN_scaler.h5')
    
    fea = scaler.transform(com_fea)
     
    model = load_model('model/MsDBN_model.h5')
    proba = model.predict(fea)
    
    pred = (proba > 0.5).astype('int32')
    
    tp, fn, sensitivity = calculate_performace_DBP(len(pred), pred, Y)
    print('***************************************')
    print('\ttp=%0.0f,fn=%0.0f'%(tp, fn))
    print('\tsn=%0.2f'% (sensitivity*100))
    print('***************************************')

def predict_new_DBP(dataset_P):
    if dataset_P == 'DBP2858':
        predict_DBP2858()
    else:
        predict_new(dataset_P)
    

def predict_NDBP3723():
    rbp_neg_dict = read_fasta_file('data/NDBP3723.txt')
    com_fea, Y = use_composition_neg(rbp_neg_dict)
    
    scaler = joblib.load('model/MsDBN_scaler.h5')
    fea = scaler.transform(com_fea)
     
    model = load_model('model/MsDBN_model.h5')
    proba = model.predict(fea)
    pred = (proba > 0.5).astype('int32')
    tn, fp, FPR = calculate_performace_NDBP(len(pred), pred, Y)
    print('========================')
    print('\ttn=%0.0f,fp=%0.0f'%(tn, fp))
    print('\tFPR=%0.2f%%' % (FPR*100))
    print('========================')


def predict_new_fpr(dataset_N):
    rbp_neg_dict = read_fasta_file(dataset_N)
    
    com_fea, Y = use_composition_neg(rbp_neg_dict)
    
    scaler = joblib.load('model/MsDBN_scaler.h5')
    fea = scaler.transform(com_fea)
     
    model = load_model('model/MsDBN_model.h5')
    proba = model.predict(fea)
    pred = (proba > 0.5).astype('int32')
    tn, fp, FPR = calculate_performace_NDBP(len(pred), pred, Y)
    print('========================')
    print('\ttn=%0.0f,fp=%0.0f'%(tn, fp))
    print('\tFPR=%0.2f%%' % (FPR*100))
    print('========================')


def predict_fpr(dataset_N):
    if dataset_N == 'NDBP3723':
        predict_NDBP3723()
    else:
        predict_new_fpr(dataset_N)
        

#%%
parser = argparse.ArgumentParser(description = """MsDBP: Exploring DNA-binding Proteins by Integrating Multi-scale Sequence Information with Deep Neural Network""")

parser.add_argument('-dataset_P',
                    type=str, help='which positive samples you want to predict')

parser.add_argument('-dataset_N',
                    type=str, help='which negative samples you want to predict')

args = parser.parse_args()
dataset_P = args.dataset_P
dataset_N = args.dataset_N
if dataset_P is not None and dataset_N is not None:
    MsDBP(dataset_P, dataset_N)
else:
    if dataset_P is not None:
        predict_new_DBP(dataset_P)
    if dataset_N is not None:
        predict_fpr(dataset_N)
    if dataset_P is None and dataset_N is None:
        print('you must input fasta file')




























