from sklearn.metrics import f1_score, precision_score, recall_score
from fastai.text import *
from qwk import quadratic_weighted_kappa
from scipy.stats import pearsonr

def f1(preds,targs):
    targs=targs.cpu().numpy()
    preds=np.argmax(F.softmax(V(preds)).data.cpu().numpy(),axis=1)
    return f1_score(targs,preds,average='weighted')

def prec(preds,targs):
    targs=targs.cpu().numpy()
    preds=np.argmax(F.softmax(V(preds)).data.cpu().numpy(),axis=1)
    return precision_score(targs,preds,average='weighted')

def recall(preds,targs):
    targs=targs.cpu().numpy()
    preds=np.argmax(F.softmax(V(preds)).data.cpu().numpy(),axis=1)
    return recall_score(targs,preds,average='weighted')

def qwk(preds,targs):
    targs=targs.cpu().numpy()
    preds=np.argmax(F.softmax(V(preds)).data.cpu().numpy(),axis=1)
    return quadratic_weighted_kappa(targs,preds)

def pearson(preds,targs):
    targs=targs.cpu().numpy()
    preds=np.argmax(F.softmax(V(preds)).data.cpu().numpy(),axis=1)
    return pearsonr(targs,preds)[0]
