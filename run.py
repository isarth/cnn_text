from lstm_model import AttentionModel
from fastai.text import *
from create_toks import get_texts
from qwk import quadratic_weighted_kappa
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
import argparse
torch.cuda.set_device(1)

parser= argparse.ArgumentParser()
parser.add_argument('--bs',type=int,default=32)
parser.add_argument('--output-size',type=int)
parser.add_argument('--emb-size',default=300, type=int)
parser.add_argument('--hid',default=128, type=int)
parser.add_argument('--data')
parser.add_argument('--model-path')
parser.add_argument('--img')
parser.add_argument('--csv')
args=parser.parse_args()

bs = args.bs
output_size= args.output_size
hidden_size= args.hid
embedding_length = args.emb_size

itos=pickle.load(open(f'{args.data}tmp/itos.pkl','rb'))
stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})
vocab_size=len(itos)

stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})

m=AttentionModel(output_size=output_size,hidden_size=hidden_size,vocab_size=vocab_size,
    embedding_length=embedding_length)
m=m.cuda()
m.load_state_dict(torch.load(args.model_path))


def predict(m,ds_test):
    tok_test, test_labels = get_texts(ds_test, 1)
    test_cls = np.array([[stoi[o] for o in p] for p in tok_test])
    test_labels=np.squeeze(test_labels)
    test_ds = TextDataset(test_cls, test_labels)
    test_dl=DataLoader(test_ds, bs, transpose=True, num_workers=1, pad_idx=1)
    m.eval()
    result=[]
    itrator=iter(test_dl)
    while True:
        try:
            x,y=next(itrator)            
            m.eval()
            out=m(V(x))
            result+=list(out.max(1)[1].cpu().data.numpy())
        except:
            break
    return np.array(result)
    

if args.csv:
    df_test = pd.read_csv(args.csv,header=None)
else:
    df_test=pd.read_csv(f'{args.data}test.csv',header=None)

ds_test=pd.DataFrame({0:0,1:df_test[1]})
ans=predict(m,ds_test)


#df_test.to_csv(f'{args.data}result.csv')

cm = confusion_matrix( df_test[0].values, ans)
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True)
plt.show()
plt.savefig(f'{args.data}{args.img}.png')
#print ( np.unique(ans, return_counts=True))

print ('ACCURACY ', accuracy_score(df_test[0].values,ans) )
print('QWK ', quadratic_weighted_kappa(df_test[0].values,ans))
print ('PEARSONR ',pearsonr(df_test[0].values,ans)[0])
print ('F1 ',f1_score(df_test[0].values, ans, average='weighted'))
print ('PREC ', precision_score(df_test[0].values, ans, average='weighted'))
print ('RECALL ', recall_score(df_test[0].values, ans, average='weighted'))



