# -*- coding: utf-8 -*-
"""Experiment 4: offline replication of the provided MNIST MLP method.
Data: real sklearn handwritten digits (NOT MNIST), resized 8x8 -> 28x28.
"""
from pathlib import Path
import sys, platform, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader

BASE = Path(__file__).resolve().parent
OUT = BASE / 'images'; OUT.mkdir(exist_ok=True)
random.seed(42); np.random.seed(42); torch.manual_seed(42)
torch.set_num_threads(2)
plt.rcParams['font.family'] = 'DejaVu Sans'

# Real 8x8 handwritten digits: 1797 samples; distinct from MNIST.
digits = load_digits()
raw_images = digits.images.astype(np.float32) / 16.0
labels = digits.target.astype(np.int64)
# Resize via nearest-neighbor, yielding 28x28 feature vectors to match PDF's 784 inputs.
x = torch.from_numpy(raw_images[:,None,:,:])
x = F.interpolate(x, size=(28,28), mode='bilinear', align_corners=False).squeeze(1)
y = torch.from_numpy(labels)
ids_train, ids_test = train_test_split(np.arange(len(y)),test_size=0.2,random_state=42,stratify=labels)
train_loader=DataLoader(TensorDataset(x[ids_train],y[ids_train]),batch_size=64,shuffle=True)
test_loader=DataLoader(TensorDataset(x[ids_test],y[ids_test]),batch_size=64,shuffle=False)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)
    def forward(self,x):
        return self.l2(F.relu(self.l1(x)))

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=MLP().to(device)
optimizer=optim.SGD(model.parameters(),lr=0.01)
epochs=30
hist=[]
logs=[]
info={'dataset':'sklearn.datasets.load_digits (NOT MNIST)','num_samples':len(labels),'train_size':len(ids_train),'test_size':len(ids_test),'image_original':'8x8','image_resized':'28x28','seed':42,'device':str(device),'epochs':epochs,'batch_size':64,'lr':0.01,'optimizer':'SGD','parameters':sum(p.numel() for p in model.parameters()),'python':platform.python_version(),'torch':torch.__version__,'sklearn': __import__('sklearn').__version__}
print('Experiment 4 / offline real handwritten digits',flush=True)
print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
t0=time.time()
for epoch in range(1,epochs+1):
    model.train(); train_loss_sum=0.; total=0
    for imgs,targets in train_loader:
        imgs=imgs.reshape(-1,784).to(device);targets=targets.to(device)
        optimizer.zero_grad()
        output=model(imgs)
        loss=F.cross_entropy(output,targets)
        loss.backward();optimizer.step()
        bs=len(targets);train_loss_sum+=loss.item()*bs;total+=bs
    model.eval();ncorrect=0;test_loss_sum=0.; ntest=0
    with torch.no_grad():
        for imgs,targets in test_loader:
            imgs=imgs.reshape(-1,784).to(device);targets=targets.to(device)
            output=model(imgs);bs=len(targets)
            test_loss_sum+=F.cross_entropy(output,targets).item()*bs
            ncorrect+=(output.argmax(1)==targets).sum().item();ntest+=bs
    tr=train_loss_sum/total;te=test_loss_sum/ntest;acc=ncorrect/ntest
    hist.append({'epoch':epoch,'train_loss':tr,'test_loss':te,'accuracy':acc})
    line=f'epoch:{epoch:02d}, train_loss:{tr:.4f}, test_loss:{te:.4f}, acc:{acc:.4f}'
    print(line,flush=True);logs.append(line)
info.update({'runtime_seconds':round(time.time()-t0,2),'final_accuracy':hist[-1]['accuracy'],'final_test_loss':hist[-1]['test_loss']})
model.eval()
with torch.no_grad():
    out=model(x[ids_test].reshape(-1,784).to(device))
    pred=out.argmax(1).cpu().numpy()
truth=labels[ids_test]
cm=confusion_matrix(truth,pred,labels=np.arange(10))
info['confusion_matrix']=cm.tolist()
(BASE/'run_log.txt').write_text('ENVIRONMENT / DATA\n'+json.dumps(info,indent=2,ensure_ascii=False)+'\n\nEPOCH LOG\n'+'\n'.join(logs)+'\n\nCLASSIFICATION REPORT\n'+classification_report(truth,pred,zero_division=0),encoding='utf-8')
(BASE/'metrics.json').write_text(json.dumps({'info':info,'history':hist},ensure_ascii=False,indent=2),encoding='utf-8')

# Figure 1: authentic samples from dataset, real plotting output.
fig,axes=plt.subplots(2,5,figsize=(11,5.4));
for ax,idx in zip(axes.flat,[int(np.where(labels==i)[0][1]) for i in range(10)]):
    ax.imshow(raw_images[idx],cmap='gray_r',vmin=0,vmax=1,interpolation='nearest');ax.set_title(f'Label: {labels[idx]}');ax.axis('off')
fig.suptitle('Real handwritten digits — sklearn digits dataset (8×8)',fontsize=13)
fig.tight_layout();fig.savefig(OUT/'01_real_digit_samples.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(9,4))
idx=int(ids_train[0]);axes[0].imshow(raw_images[idx],cmap='gray_r',vmin=0,vmax=1);axes[0].set_title('Original 8×8')
axes[1].imshow(x[idx].numpy(),cmap='gray_r',vmin=0,vmax=1);axes[1].set_title('Resized 28×28 for 784 inputs')
for ax in axes:ax.axis('off')
fig.suptitle(f'Actual sample: digit {labels[idx]}');fig.tight_layout();fig.savefig(OUT/'02_resize_comparison.png',dpi=160);plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(12,4.5))
axs[0].plot([h['epoch'] for h in hist],[h['train_loss'] for h in hist],marker='.',label='Training loss')
axs[0].plot([h['epoch'] for h in hist],[h['test_loss'] for h in hist],marker='.',label='Test loss')
axs[0].set(title='Cross-entropy loss — actual training',xlabel='Epoch',ylabel='Loss');axs[0].legend();axs[0].grid(alpha=.3)
axs[1].plot([h['epoch'] for h in hist],[100*h['accuracy'] for h in hist],marker='.')
axs[1].set(title='Test accuracy — actual training',xlabel='Epoch',ylabel='Accuracy (%)',ylim=(0,100));axs[1].grid(alpha=.3)
fig.tight_layout();fig.savefig(OUT/'03_training_metrics.png',dpi=170);plt.close(fig)

fig,ax=plt.subplots(figsize=(7.5,6.5));im=ax.imshow(cm,cmap='Blues')
ax.set(xticks=np.arange(10),yticks=np.arange(10),xlabel='Predicted label',ylabel='True label',title='Confusion matrix — held-out test set')
for i in range(10):
 for j in range(10):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=8,color='white' if cm[i,j]>cm.max()*.5 else 'black')
fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/'04_confusion_matrix.png',dpi=160);plt.close(fig)

fig,axes=plt.subplots(3,5,figsize=(12,7))
for ax,pos in zip(axes.flat,np.linspace(0,len(ids_test)-1,15,dtype=int)):
 ax.imshow(x[ids_test[pos]].numpy(),cmap='gray_r',vmin=0,vmax=1)
 ax.set_title(f'True: {truth[pos]} | Pred: {pred[pos]}',fontsize=10,color='darkred' if truth[pos]!=pred[pos] else 'black');ax.axis('off')
fig.suptitle('Real test-set predictions (not mock screenshots)');fig.tight_layout();fig.savefig(OUT/'05_test_predictions.png',dpi=160);plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5));ax.axis('off')
rows=[[str(h['epoch']),f"{h['train_loss']:.4f}",f"{h['test_loss']:.4f}",f"{h['accuracy']*100:.2f}%"] for h in hist if h['epoch'] in (1,5,10,15,20,25,30)]
tb=ax.table(cellText=rows,colLabels=['Epoch','Train loss','Test loss','Test accuracy'],loc='center',cellLoc='center');tb.auto_set_font_size(False);tb.set_fontsize(11);tb.scale(1,1.7)
ax.set_title('Selected logged epochs — actual measurements',pad=18)
fig.tight_layout();fig.savefig(OUT/'06_epoch_data_table.png',dpi=160);plt.close(fig)
print('DONE:',json.dumps({'accuracy':info['final_accuracy'],'loss':info['final_test_loss'],'seconds':info['runtime_seconds']}),flush=True)
