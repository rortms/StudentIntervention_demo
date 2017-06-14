"""
=================================
Box plots with custom fill colors
=================================

This plot illustrates how to create two types of box plots
(rectangular and notched), and how to fill them with custom
colors by accessing the properties of the artists of the
box plots. Additionally, the ``labels`` parameter is used to
provide x-tick labels for each sample.

A good general reference on boxplots and their history can be found
here: http://vita.had.co.nz/papers/boxplots.pdf
"""

# Index([u'school',
#        u'sex',
#        u'age',
#        u'address',
#        u'famsize',
#        u'Pstatus',
#        u'Medu',
       
#        u'Fedu',
#        u'Mjob',
#        u'Fjob',
#        u'reason',
#        u'guardian',
#        u'traveltime',
       
#        u'studytime',
#        u'failures',
#        u'schoolsup',
#        u'famsup',
#        u'paid',
       
#        u'activities',
#        u'nursery',
#        u'higher',
#        u'internet',
#        u'romantic',
       
#        u'famrel',
#        u'freetime',
#        u'goout',
#        u'Dalc',
#        u'Walc',
#        u'health',
       
#        u'absences',
#        u'passed'],
#       dtype='object')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

student_data = pd.read_csv("./student-data.csv")
print student_data.columns[-1]

#########################################
# Data dimensions and grad_rate
n_students = student_data.shape[0]
n_features = student_data.shape[1] - 1
n_passed = sum([1 for y in student_data['passed'] if y == 'yes'])
n_failed = sum([1 for n in student_data['passed'] if n == 'no'])
grad_rate = 100.*n_passed/(n_passed + n_failed)

#########################################
# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]           # last column is the target/label

x_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]    # corresponding targets/labels


########################################
# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

X_all = preprocess_features(x_all)
# print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))


##########################################################
# Split to Training and testing datasets

# Split fraction
# num_all = student_data.shape[0]  # same as len(student_data)
# num_train = int(np.floor(num_all * .76 ))
# num_test = num_all - num_train

# Split using Sklearn builtin
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size = .24, random_state = 0)
######
# Note: If you need a validation set, extract it from within training data

# print "Training set: {} samples".format(X_train.shape[0])
# print "Test set: {} samples".format(X_test.shape[0])



	##########################################################
        #               Classifier Exploration
        ##########################################################


##############################################
# Helper Functions
import time
from sklearn.metrics import f1_score

# Return the classifier's training time
def timeTraining(clf, X_train, y_train):
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    return (end - start)

# Return the classifier's predictions and prediction time
def predictAndTime(clf, features):
    start = time.time()
    y_pred = clf.predict(features)
    end = time.time()
    return y_pred, (end - start)

# Return the f1 score for the target values and predictions
def F1(target, prediction):
    return f1_score(target.values, prediction, pos_label='yes')
###########

##############
# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

#Adding Kfold validation to try out the Pro Tip from CodeReview2
from sklearn.cross_validation import KFold  

# Setting up KFold cross_validation object
kf = KFold(X_train.shape[0], 10)

# Array of classifiers
clfs = [KNeighborsClassifier(n_neighbors = 3)]
 
#Gathering Table column and index labels
classifier_names = [clf.__class__.__name__ for clf in clfs]
benchmarks = ["Training time",  "F1 score training set","Prediction time", "F1 score test set"]
table = pd.DataFrame(columns = classifier_names, index = benchmarks)

# Fit Classifiers and average the times and f1 scores resulting from KFold (10 folds)
for clf in clfs: 
    classifier   = clf.__class__.__name__
    t_test  = 0.0 
    t_train = 0.0
    F1_test = 0.0
    F1_train =0.0
    
    #Averaging scores and seconds accross the folds
    for tr_i ,t_i in kf:
        #Train (k-1 buckets)
        t_train += timeTraining(clf, X_train.iloc[tr_i], y_train.iloc[tr_i])
        pred_train_set = predictAndTime(clf, X_train.iloc[tr_i])[0]
        F1_train += F1(y_train.iloc[tr_i], pred_train_set)
        #Test (kth bucket)
        pred_test_set, t_t = predictAndTime(clf,X_train.iloc[t_i])
        t_test += t_t
        F1_test += F1(y_train.iloc[t_i], pred_test_set)
        
    #Filling table 
    table[classifier]['Training time']         = "{:10.4f} s".format(t_train/10)
    table[classifier]['F1 score training set'] = F1_train/10
    table[classifier]['Prediction time']       = "{:10.4f} s".format(t_test/10)
    table[classifier]['F1 score test set']     = F1_test/10


print table



from sklearn import grid_search
from sklearn.metrics import make_scorer

scorer = make_scorer(F1)

tree_param = { 'n_estimators' : range(2,7),'criterion': ["entropy", "gini"], 'max_features':["sqrt", "log2"], 'max_depth': range(2,11),
               'min_samples_split':range(2,9), 'min_samples_leaf':range(1,9) }

neigh_param = {'n_neighbors' : [3,5,10,20,25,30,40], 'weights' : ['uniform', 'distance'], 'p':[1,2,3,5,10],
              'algorithm' : ['ball_tree', 'kd_tree', 'brute']}

#Perform grid Search
def gridIt(clf, params):
    #Grid search folds = 10, for consistency with previous computations
    grid_clf = grid_search.GridSearchCV(clf, params,
                                        scorer, n_jobs=4, cv = 10)
    print clf.__class__.__name__
    print "Grid search time:", timeTraining(grid_clf, X_train, y_train)
    print "Parameters of tuned model: ", grid_clf.best_params_
    y_pred, predict_t = predictAndTime(grid_clf, X_test)
    print "f1_score and prediction time on X_test, y_test: "
    print F1(y_test, y_pred), predict_t
    print '------------------\n'
    
# gridIt(KNeighborsClassifier(), neigh_param)
# gridIt(RandomForestClassifier(), tree_param)


#######################################
# t-SNE cluster projection exploration

All_under = X_all.copy(deep=True)
All_under['passed'] = y_all

All_under = All_under[All_under['age'] < 19]

X_under, y_under = All_under[All_under.columns[:-1]], All_under[All_under.columns[-1]]

print X_all.shape, All_under.shape, y_all.shape
print X_under.shape, y_under.shape
# Scaling features to the same range (0-1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

uX_all = scaler.fit_transform(X_all)
# uX_all = scaler.fit_transform(X_under)

from sklearn.manifold import TSNE
model = TSNE(n_components=2,
             early_exaggeration=4.0,
             learning_rate=1000,
             n_iter=1000,
             init='pca',
             random_state=0,
             method='exact')
# np.set_printoptions(suppress=True)
clusters = model.fit_transform(uX_all)

for age in range(15, 21,2):
    lower, upper = age, age+2
    
    x_fail, y_fail = zip(*[ tuple(pt) for i, pt in enumerate(clusters) if y_all[i] == 'no' and
                        
                            X_all['age'][i] >= lower and X_all['age'][i] < upper ])

    x_pass, y_pass = zip(*[ tuple(pt) for i, pt in enumerate(clusters) if y_all[i] == 'yes' and
                        
                            X_all['age'][i] >= lower and X_all['age'][i] < upper ])
    print len(x_pass)+len(x_fail)
    plt.scatter(x_fail, y_fail, s=50, c='red', alpha=0.5, label="student failed")
    plt.scatter(x_pass, y_pass, s=50, c='blue', alpha=0.5, label="student passed")
    plt.show()

### For a 3D scatter plot
# from mpl_toolkits.mplot3d import Axes3D
# x_fail, y_fail, z_fail = zip(* [ tuple(pt) for i, pt in enumerate(clusters) if y_all[i] == 'no'])
# x_pass, y_pass, z_pass = zip(* [ tuple(pt) for i, pt in enumerate(clusters) if y_all[i] == 'yes'])

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(x_fail, y_fail, z_fail, s=50, c='red', alpha=0.5, label="student failed")
# ax.scatter(x_pass, y_pass, z_pass, s=50, c='blue', alpha=0.5, label="student passed")
# plt.show()



# colors = np.random.rand(N)
# area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radii






# # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))

# # rectangular box plot
# bplot1 = axes.boxplot(all_data,
#                          vert=True,   # vertical box aligmnent
#                          patch_artist=True)   # fill with color

# # # notch shape box plot
# # bplot2 = axes[0].boxplot(all_data,
# #                          notch=True,  # notch shape
# #                          vert=True,   # vertical box aligmnent
# #                          patch_artist=True)   # fill with color

# # fill with colors
# colors = ['pink', 'lightblue', 'lightgreen']
# for bplot in (bplot1,):#(bplot1, bplot2):
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)

# # adding horizontal grid lines
# # for ax in axes:
# #     ax.yaxis.grid(True)
# #     ax.set_xticks([y+1 for y in range(len(all_data))], )
# #     ax.set_xlabel('xlabel')
# #     ax.set_ylabel('ylabel')


# ax.yaxis.grid(True)
# ax.set_xticks([y+1 for y in range(len(all_data))], )
# ax.set_xlabel('xlabel')
# ax.set_ylabel('ylabel')

# # add x-tick labels
# plt.setp(axes, xticks=[y+1 for y in range(len(all_data))],
#          xticklabels=['x1', 'x2', 'x3'])#, 'x4'])

# plt.show()
