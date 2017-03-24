import pandas as pd
import numpy as np
import math
import sys

#read data file, split it into equal training and test sets
#scale the features to the training set
data = pd.read_csv("spambase.data", header=None)
data = data.values
np.random.shuffle(data)
(train,test) = np.array_split(data,2)

#compute prior probability of training set for each class (spam and not spam)
pri_prob_spam = sum(train[:,57])/len(train)
pri_prob_notspam = 1.0 - pri_prob_spam
print("prior probability spam: %.2f" % pri_prob_spam)
print("prior probability not spam: %.2f" % pri_prob_notspam)

#compute mean and std dev of each feature in training set for each class
X_train = train[:,0:57]
y_train = train[:,57]
X_test = test[:,0:57]
y_test = test[:,57]
mean_train_s = np.mean(X_train[y_train==1.0],axis=0 )
mean_train_ns = np.mean(X_train[y_train==0.0],axis=0 )
sd_train_s = np.std(X_train[y_train==1.0],axis=0 )
sd_train_ns = np.std(X_train[y_train==0.0],axis=0 )

#classify test set

#log_N returns the log of the normalized sample
def log_N(subsample, mean, sd):
    #if standard deviation is 0, we get a divide by zero error, so
    #make minimum cap on std dev
    if sd < 0.001:
        sd = 0.001
    Na = 1/(math.sqrt(2 * math.pi) * sd)
    Nb = -(subsample-mean)**2/(2*sd**2)
    return math.log(Na) + Nb

#argmax returns 1 if arg1 > arg 2, 0 otherwise
#this is different than the usual 1 or -1 output because we are using 0 for not spam class
def argmax(arg1, arg2):
    if arg1 > arg2:
        return 1
    else:
        return 0

#classify sums the logs of probabilities for the class and the features given the class
def classify(pri_prob, sample, mean, sd):
    sumlogs = math.log(pri_prob)
    for feature in range(0,57):
        sumlogs += log_N(sample[feature], mean[feature], sd[feature]) 
    return sumlogs

pred = []
for sample in X_test:
    pred.append( argmax(classify(pri_prob_spam, sample, mean_train_s, sd_train_s),
        classify(pri_prob_notspam, sample, mean_train_ns, sd_train_ns)))

#compute accuracy, precision, recall, and confusion matrix    
tp = 0
tn = 0
fp = 0
fn = 0

for index in range(0,len(y_test)):
    if pred[index] == 1:
        if y_test[index] == 1:
            tp += 1
        else:
            fp += 1
    else:
        if y_test[index] == 1:
            fn += 1
        else:
            tn += 1
acc = (tp + tn)/(tp+tn+fp+fn)
prec = tp/(tp+fp)
rec = tp/(tp+fn)

print("accuracy: %.4f precision: %.4f recall: %.4f" % (acc, prec, rec))
print("confusion matrix")
print("\tpredicted")
print("actual\tpos\tneg")
print("pos\t%d\t%d" % (tp, fn))
print("neg\t%d\t%d" % (fp, tn))

