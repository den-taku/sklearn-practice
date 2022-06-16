#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ignore warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#... import sklearn stuff...

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn import svm
import sys, time

Times = 10
Fold = 5
ZERO_TOL = 0.000001

################################################
def read_dataset(data_csv, value_txt):
    
    ### read files ###
    # read the csv and the observed values
    fv = pd.read_csv(data_csv) 
    value = pd.read_csv(value_txt)      

    ### prepare data set ###
    # prepare CIDs
    CIDs = np.array(fv['CID'])
    # prepare target, train, test arrays
    target = np.array(value['a'])
    # construct dictionary: CID to feature vector
    fv_dict = {}
    for cid,row in zip(CIDs, fv.values[:,1:]):
        fv_dict[cid] = row
    # construct dictionary: CID to target value
    target_dict = {}
    for cid, val in zip(np.array(value['CID']), np.array(value['a'])):
        target_dict[cid] = val
    # check CIDs: target_values_filename should contain all CIDs that appear in descriptors_filename
    for cid in CIDs:
        if cid not in target_dict:
            sys.stderr.write('error: {} misses the target value of CID {}\n'.format(target_values_filename, cid))
            exit(1)
    # construct x and y so that the CIDs are ordered in ascending order
    CIDs.sort()
    x = np.array([fv_dict[cid] for cid in CIDs])
    y = np.array([target_dict[cid] for cid in CIDs])
    return (CIDs,x,y)

    
################################################
def learn_Lasso(x_train, y_train, x_test, y_test, a=1.0):
    lasso = Lasso(alpha=a, max_iter=10**5)
    lasso.fit(x_train, y_train)
    r2train = lasso.score(x_train,y_train)
    r2test = lasso.score(x_test,y_test)
    nonzero = len([w for w in lasso.coef_ if abs(w)>=ZERO_TOL])
    return (lasso, nonzero, r2train, r2test)

################################################

################################################
def learn_Svm(x_train, y_train, x_test, y_test):
    machine = svm.SVR(kernel="linear", max_iter=10**2)
    machine.fit(x_train, y_train)
    r2train = machine.score(x_train,y_train)
    r2test = machine.score(x_test,y_test)
    return (machine, r2train, r2test)

################################################

################################################
def learn_Svm_iter(x_train, y_train, x_test, y_test, max_iter):
    machine = svm.SVR(kernel="rbf", max_iter=max_iter)
    machine.fit(x_train, y_train)
    r2train = machine.score(x_train,y_train)
    r2test = machine.score(x_test,y_test)
    return (machine, r2train, r2test)

################################################

################################################
def learn_LinearSvm(x_train, y_train, x_test, y_test):
    machine = svm.LinearSVR(max_iter=10**5)
    machine.fit(x_train, y_train)
    r2train = machine.score(x_train,y_train)
    r2test = machine.score(x_test,y_test)
    return (machine, r2train, r2test)

################################################

################################################
def learn_LinearSvm_iter(x_train, y_train, x_test, y_test, max_iter):
    machine = svm.LinearSVR(max_iter=max_iter)
    machine.fit(x_train, y_train)
    r2train = machine.score(x_train,y_train)
    r2test = machine.score(x_test,y_test)
    return (machine, r2train, r2test)

################################################

######### For evaluate hyper parameters #######
def learn_Tuning(x_train, y_train, x_test, y_test, arch):

    R = []
    reg = MLPRegressor(activation='relu', solver='adam',
                       alpha=1e-5, hidden_layer_sizes=arch,
                       random_state=1, early_stopping=False)
    reg.warm_start = False
    # learn ANN, but stop the learning at itr=t in order to record stats
    for t in range(10, MaxItr+1, 10):
        reg.max_iter = t
        reg.fit(x_train, y_train)
        reg.warm_start = True                

        # calculate the prediction score (R^2)
        r2train = reg.score(x_train,y_train)
        r2test = reg.score(x_test,y_test)
        R.append((t,r2train, r2test))

    return R
################################################

try:
    CIDs, x, y = read_dataset(sys.argv[1], sys.argv[2])
    lmd = float(sys.argv[3])
except:
    sys.stderr.write("usage: {} (input_data.csv)(input_values.txt)(lambda)\n\n".format(sys.argv[0]))
    exit(1)

### experiment1 ###

# print("SVR:")
# print("Lambda\t{}".format(lmd))
# f = open(sys.argv[1])
# arr = f.readline().split(',')
# print("NumDesc\t{}".format(len(arr)-1))
# f.close()

# for split_seed in range(1, Times+1):
#     kf = KFold(n_splits=Fold, shuffle=True, random_state=split_seed)

#     fold = 0
#     Tr = []
#     Ts = []
#     Tim = []
#     for train, test in kf.split(x):
#         fold += 1
#         start_time = time.time()
#         _, r2train, r2test = learn_Svm(x[train], y[train], x[test], y[test])
#         comp_time = time.time() - start_time
#         Tr.append(r2train)
#         Ts.append(r2test)
#         Tim.append(comp_time)
#     print("{}\tTrain".format(split_seed), end="")
#     for v in Tr:
#         print("\t{:.6f}".format(v), end="")
#     print()
#     print(" \tTest", end="")
#     for v in Ts:
#         print("\t{:.6f}".format(v), end="")
#     print()
#     print()
    
### experiment2 ###

# print("LinearSVM")
# print("Lambda\t{}".format(lmd))
# f = open(sys.argv[1])
# arr = f.readline().split(',')
# print("NumDesc\t{}".format(len(arr)-1))
# f.close()

# for split_seed in range(1, Times+1):
#     kf = KFold(n_splits=Fold, shuffle=True, random_state=split_seed)

#     fold = 0
#     Tr = []
#     Ts = []
#     Tim = []
#     for train, test in kf.split(x):
#         fold += 1
#         start_time = time.time()
#         _, r2train, r2test = learn_LinearSvm(x[train], y[train], x[test], y[test])
#         comp_time = time.time() - start_time
#         Tr.append(r2train)
#         Ts.append(r2test)
#         Tim.append(comp_time)
#     print("{}\tTrain".format(split_seed), end="")
#     for v in Tr:
#         print("\t{:.6f}".format(v), end="")
#     print()
#     print(" \tTest", end="")
#     for v in Ts:
#         print("\t{:.6f}".format(v), end="")
#     print()
#     print()

### experiment3 ###

print("SVR tuning:")
print("Lambda\t{}".format(lmd))
f = open(sys.argv[1])
arr = f.readline().split(',')
print("NumDesc\t{}".format(len(arr)-1))
f.close()
for max_iter in [10**1, 10**2, 10**3, 10**4, 10**5, 10**6]:
    print("max_iter = {}".format(max_iter))
    trainMid = []
    testMid = []
    for split_seed in range(100, Times+100):
        kf = KFold(n_splits=Fold, shuffle=True, random_state=split_seed)

        fold = 0
        Tr = []
        Ts = []
        Tim = []
        for train, test in kf.split(x):
            fold += 1
            start_time = time.time()
            _, r2train, r2test = learn_Svm_iter(x[train], y[train], x[test], y[test], max_iter)
            comp_time = time.time() - start_time
            Tr.append(r2train)
            Ts.append(r2test)
            Tim.append(comp_time)
        print("{}\tTrain".format(split_seed), end="")
        for v in Tr:
            print("\t{:.6f}".format(v), end="")
            trainMid.append(v)
        print()
        print(" \tTest", end="")
        for v in Ts:
            print("\t{:.6f}".format(v), end="")
            testMid.append(v)
        print()
        print()
    trainMid.sort()
    testMid.sort()
    print("testMid = {}, trainMid = {}".format((trainMid[24] + trainMid[25])/2, (testMid[24] + testMid[25])/2))
    print()
    
### experiment4 ###

# print("SVR Lineartuning:")
# print("Lambda\t{}".format(lmd))
# f = open(sys.argv[1])
# arr = f.readline().split(',')
# print("NumDesc\t{}".format(len(arr)-1))
# f.close()
# for max_iter in [10**1, 10**2, 10**3, 10**4, 10**5]:
#     print("max_iter = {}".format(max_iter))
#     for split_seed in range(1, Times+1):
#         kf = KFold(n_splits=Fold, shuffle=True, random_state=split_seed)

#         fold = 0
#         Tr = []
#         Ts = []
#         Tim = []
#         for train, test in kf.split(x):
#             fold += 1
#             start_time = time.time()
#             _, r2train, r2test = learn_LinearSvm_iter(x[train], y[train], x[test], y[test], max_iter)
#             comp_time = time.time() - start_time
#             Tr.append(r2train)
#             Ts.append(r2test)
#             Tim.append(comp_time)
#         print("{}\tTrain".format(split_seed), end="")
#         for v in Tr:
#             print("\t{:.6f}".format(v), end="")
#         print()
#         print(" \tTest", end="")
#         for v in Ts:
#             print("\t{:.6f}".format(v), end="")
#         print()
#         print()
    
