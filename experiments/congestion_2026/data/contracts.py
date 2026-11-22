import numpy as np
import pandas as pd

def spatial_y(speed,weights,c=.5):
    x=np.asarray(speed,float);w=np.asarray(weights,float)
    good=np.isfinite(x)&(x>=0)
    denom=(good*w).sum(axis=1)
    score=np.divide(((x<40)&good)*w,np.ones_like(x),out=np.zeros_like(x)).sum(axis=1)
    score=np.divide(score,denom,out=np.full(len(x),np.nan),where=denom>0)
    y=np.where(denom>0,np.where(score==0,0,np.where(score>=c,2,1)),np.nan)
    return score,y,good.sum(axis=1)
def assert_available(available,prediction):
    if (pd.to_datetime(available)>pd.to_datetime(prediction)).any():raise ValueError('Future feature detected')
def assert_probabilities(p):
    p=np.asarray(p,float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1,atol=1e-6):raise ValueError('Invalid three-class probabilities')
def assert_same_samples(a,b):
    for key in ['prediction_time','target_time','y','fold']:
        if not np.array_equal(a[key].to_numpy(),b[key].to_numpy(),equal_nan=False):raise ValueError('Non-comparable evaluation samples: '+key)
def train_mask(frame,cutoff,embargo=30,warmup_end='2024-04-01'):
    return frame.y.notna()&(frame.prediction_time>=pd.Timestamp(warmup_end))&(frame.target_time+pd.Timedelta(minutes=embargo)<pd.Timestamp(cutoff))
def assert_train_validation(train,validation,embargo=30):
    if train.target_time.max()+pd.Timedelta(minutes=embargo)>=validation.prediction_time.min():raise ValueError('Training targets overlap validation/embargo')
