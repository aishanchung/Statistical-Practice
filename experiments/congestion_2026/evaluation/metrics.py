import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix,log_loss,precision_recall_fscore_support,average_precision_score
from ..data.contracts import assert_probabilities

def evaluate(y,p,decision=None):
    y=np.asarray(y,int);p=np.asarray(p,float);assert_probabilities(p)
    pred=p.argmax(1) if decision is None else np.asarray(decision)
    pr,re,f1,support=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
    mat=confusion_matrix(y,pred,labels=[0,1,2])
    metrics={'n':len(y),'accuracy':accuracy_score(y,pred),'balanced_accuracy':re.mean(),'macro_f1':f1.mean(),'log_loss':log_loss(y,p,labels=[0,1,2]),'average_precision_2':average_precision_score(y==2,p[:,2]) if (y==2).any() else np.nan,'multiclass_brier':np.mean(np.sum((p-np.eye(3)[y])**2,axis=1)),'y2_false_positive_rate':np.mean(pred[y!=2]==2) if (y!=2).any() else np.nan}
    for k in range(3):metrics.update({f'precision_{k}':pr[k],f'recall_{k}':re[k],f'f1_{k}':f1[k],f'support_{k}':int(support[k])})
    return metrics,mat
