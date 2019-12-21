from metrics import *
from fastai.text import *
from cnn_model import 
import argparse
from early import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

def train(bs,cuda_id,dir_path, epochs,lr,save_name=None, w_loss=1, use_sam=1):
    
    dir_path=Path(dir_path)
    tmp_path=dir_path/'tmp/'
    models_path = dir_path/'cnn/models'
    models_path.mkdir( exist_ok=True )
    
    torch.cuda.set_device(cuda_id)

    itos = pickle.load(open(tmp_path/'itos.pkl', 'rb'))
    vs = len(itos)
    stoi = collections.defaultdict(lambda:0, {v:k for k,v in enumerate(itos)})

    trn_sent = np.load(tmp_path/'trn_ids.npy',allow_pickle=True)
    val_sent = np.load(tmp_path/'val_ids.npy',allow_pickle=True)
    trn_labels=np.squeeze(np.load(tmp_path/'lbl_trn.npy',allow_pickle=True))
    val_labels=np.squeeze(np.load(tmp_path/'lbl_val.npy',allow_pickle=True))
    num_classes = len(np.unique(trn_labels))

    print ( f'Number of classes: ', num_classes)
    
    if w_loss==1:
        print ('Using weighted Loss')
        w = compute_class_weight('balanced', sorted(np.unique(trn_labels)), trn_labels)
        #w = w/sum(w)
        w=(torch.FloatTensor(w).cuda())
        loss_func = nn.CrossEntropyLoss(weight=w)
    else:
        loss_func = nn.CrossEntropyLoss()

    trn_ds = TextDataset(trn_sent, trn_labels)
    val_ds = TextDataset(val_sent, val_labels)
    
    trn_samp = SortishSampler(trn_sent, key=lambda x: len(trn_sent[x]), bs=bs)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    
    if use_sam==1:
       print ('Using Sortish Sampler')
       trn_dl = DataLoader(trn_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp, drop_last=True) #sampler
       val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp, drop_last=True)
    else:
       trn_dl = DataLoader(trn_ds, bs, transpose=True, num_workers=1, pad_idx=1, drop_last=True) 
       val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, drop_last=True)
        
    md = ModelData(dir_path, trn_dl, val_dl)
    print ('Building Model')
    
    emb_weights=np.load(tmp_path/'glove_embeddings.npy',allow_pickle=True)
    embedding_length = emb_weights.shape[1]
    
    m= CNN_Text(output_size=num_classes,vocab_size=vs,
                 embedding_length=embedding_length,load_emb=True,emb_weights=emb_weights)
    m = m.cuda()

    lo=LayerOptimizer(optim.Adam,m,lr,1e-5)
    cb=[CosAnneal(lo,len(md.trn_dl),cycle_mult=2),EarlyStopping( m, models_path/f'{save_name}',opt='m',pos=5)]

    fit(m, md, epochs, lo.opt ,loss_func,metrics=[accuracy, prec, recall, f1, qwk, pearson ],callbacks=cb)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--data')
    parser.add_argument('--lr',default=1e-3,type=float)
    parser.add_argument('--cuda_id', type=int)
    parser.add_argument('--epochs',default=30, type=int)
    parser.add_argument('--use-sampler',default=1, type=int)
    parser.add_argument('--save-name',default='best-model')
    parser.add_argument('--w-loss',default=1,type=int)
    args=parser.parse_args()
    train(args.bs, args.cuda_id, dir_path=args.data, epochs=args.epochs, w_loss=args.w_loss, use_sam=args.use_sampler, save_name=args.save_name, lr=args.lr )
