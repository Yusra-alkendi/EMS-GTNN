


#read dependancies

from __future__ import division
import os.path as osp

import math
import shutil
import os
import io
import time 
import csv
import sys
import random 
import glob
import pandas as pd
import networkx as nx
import numpy
import numpy as np
import matplotlib.pyplot as plt


##------------------t
import torch
from torch import nn
import torch.nn as nn 
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch.nn import Sequential as Seq, Dropout, Linear as Lin, BatchNorm1d as BN, ReLU
import torch_scatter
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import max_pool, max_pool_x, graclus, global_mean_pool, GCNConv,  global_mean_pool, SAGEConv
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.transforms import Cartesian
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, download_url
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.nn.pool import radius_graph

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections.abc import Sequence
from collections import Counter
from sklearn.utils import compute_class_weight
from torch import Tensor
try:
    import torch_cluster
except ImportError:
    torch_cluster = None
import time


import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# from utils import Logger


torch.cuda.empty_cache()
torch.backends.cudnn.benchmark=True

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Function and classess
class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=0.5, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
    
def TRAINING_MODULE(model, number_of_epoch, train_loader, FOLDERTOSAVE): 
    model.train()

    i=0
    epoch_losses = []
    print ("Training will start now")
    for epoch in range(number_of_epoch):
        print("Epoch", epoch)
        epoch_loss = 0
        start=time.time()    
        for i, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
    
            # print ("data.x", data)
   
            end_point = model( data.x, data.pos, data.batch)
            # print("end-point", end_point)
            # print("end-data.y", data.y)

            loss=loss_func(end_point, data.y) 
            # loss=F.nll_loss(end_point, data.y) 

            pred = end_point.max(1)[1]
            acc = (pred.eq(data.y).sum().item())/len(data.y)

            loss.backward()
            optimizer.step() 
            epoch_loss += loss.detach().item()
            i=i+1
        epoch_loss /= (i + 1)
        end = time.time()
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), ' Elapsed time: ', end-start)
        epoch_losses.append(epoch_loss)
        torch.save(model.state_dict(), FOLDERTOSAVE+'model_weights.pth')
        torch.save(model, FOLDERTOSAVE+'model.pkl')
    #     toc()
    return epoch_losses
def TESTING_MODULE(model, test_loader, descrpt):
    model.eval()

    correct = 0
    mov_correct=0
    # T_label=0
    # T_label00=0
    # P_label00=0
    # P_label0=0
    GT=[]
    prediction=[]
    # qqqqqqqq=[]
    id_dataaa=[]
    xd_dataaa=[]
    yd_dataaa=[]
    td_dataaa=[]
    torg=[]
    labeld_dataaa=[]
    for i, data in enumerate(test_loader):
        id_dataaa.append(data.id_data)
        xd_dataaa.append(data.x[:,0])
        yd_dataaa.append(data.x[:,1])
        td_dataaa.append(data.x[:,2])
        labeld_dataaa.append(data.y)
        torg.append(data.t_org)
        
        GT.append(data.y)
        data = data.to(device)
        end_point = model(data.x, data.pos, data.batch)
        loss = loss_func(end_point, data.y)
        pred = end_point.max(1)[1]

        # T_label1=(data.y.eq(1).sum().item())

        # T_label+=T_label1
        # T_label0=(data.y.eq(0).sum().item())
        # T_label00+=T_label0
        # P_label1=(pred.eq(1).sum().item())
        # P_label0+=P_label1
        # P_label0=(pred.eq(0).sum().item())
        # P_label00+=P_label0

        acc = (pred.eq(data.y).sum().item())/len(data.y)
        correct += acc


        prediction.append(pred)
        
    torch.save([numpy.hstack(id_dataaa), numpy.hstack(xd_dataaa), numpy.hstack(yd_dataaa),numpy.hstack(td_dataaa)
                , numpy.hstack(torg),  torch.cat(GT), torch.cat(prediction)], descrpt)
    return [GT, prediction]
def METRICS_MODULE(GT_lbls_,argmax_Y_, csvname ):
    tn, fp, fn, tp=confusion_matrix(torch.cat(GT_lbls_).to('cpu'),torch.cat(argmax_Y_).to('cpu')).ravel()
    Tp_matrix=tp
    Fp_matrix=fp
    Fn_matrix=fn
    Tn_matrix=tn
    print('Tp_matrix_test:',(Tp_matrix))
    print('Fp_matrix_test:' ,(Fp_matrix))
    print('Fn_matrix_test:' ,(Fn_matrix))
    print('Tn_matrix_test:', (Tn_matrix))
    Precision_negative=100*Tn_matrix/(Tn_matrix+Fn_matrix)
    Precision=100*Tp_matrix/(Tp_matrix+Fp_matrix)
    Accuracy=100*((Tp_matrix+Tn_matrix)/(Tp_matrix+Tn_matrix+Fp_matrix+Fn_matrix))
    F1_score=(2*Precision*Accuracy)/(Precision+Accuracy)
    Recall_score=100*Tp_matrix/(Tp_matrix+Fn_matrix)
    Specificity=100*(Tn_matrix)/(Tn_matrix+Fp_matrix)
    print('Precision_negative on the testing set: {:.4f}%'.format(Precision_negative))
    print('Precision on the t0esting set: {:.4f}%'.format(Precision))
    print('Recall on the testing set: {:.4f}%'.format(Recall_score))
    print('F1 score on the testing set: {:.4f}%'.format(F1_score))
    print('Accuracy on the testing set: {:.4f}%'.format(Accuracy))
    print('Specificity on the testing set: {:.4f}%'.format(Specificity))


    with open(csvname, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['Tp_matrix_test:',(Tp_matrix)])
        writer.writerow(['Fp_matrix_test:' ,(Fp_matrix)])
        writer.writerow(['Fn_matrix_test:' ,(Fn_matrix)])
        writer.writerow(['Tn_matrix_test:', (Tn_matrix)])

        writer.writerow(['Precision_negative on the testing set',(Precision_negative)])
        writer.writerow(['Precision on the t0esting set:',(Precision)])
        writer.writerow(['Recall on the testing set: ',(Recall_score)])
        writer.writerow(['F1 score on the testing set:',(F1_score)])
        writer.writerow(['Accuracy on the testing set:',(Accuracy)])
        writer.writerow(['Specificity on the testing set: ',(Specificity)])
    
#-------------------------------------------------------------------------------------------------------------------------------------------------
#deine my own dataset
import random
def files_exist(files):
    return all([osp.exists(f) for f in files])
def Slice_Raw_Events_onTemporal(deltatime, seq):
    seq_data=[]
    t_start=0
    deltatime=deltatime
    # plt.plot(seq[0][: ,2]-seq[0][0 ,2])
    # plt.show()
    tdata1=seq[0][: ,2]-seq[0][0 ,2]
    # print("1",  round(max(tdata1)/deltatime))
    for i in range (0,round(max(tdata1)/deltatime)):
        t_start=i*deltatime
        t_end=t_start+deltatime
        index=np.logical_and(tdata1<t_end, tdata1>=t_start )
        seq_data1=seq[0][index]
        # seq_data.append(seq_data1)

        if  len(seq_data1)>=5000:
            seq_data1=seq_data1[-5000:]
            # randomseq_data1=numpy.random.choice(seq_data1, size=round((0.7)*(len(seq_data1))), replace=False)


        # randomseq_data1=numpy.random.choice(seq_data1, size=round((0.7)*(len(seq_data1))), replace=False)
        # print("randomseq_data1", randomseq_data1)


        seq_data.append((seq_data1))
    return seq_data

def Process_creat_Graphs(seq_data, alpha):

    ind=  (seq_data[:, 3]>=100) & (seq_data[:, 3]<=250) & (seq_data[:, 4]>=50) & (seq_data[:, 4]<=150)

    xdata=seq_data [:, 3]*125/(640*125)
    id_data=(seq_data[:, 0])
    t_org=seq_data [:, 2]
    ydata=seq_data [:, 4]*125/(125*480)
    # if len(seq_data[0])>0:
    tdata=alpha*(seq_data [:, 2]-seq_data [0, 2])*(1/0.01)
    # if len(seq_data[0])==0:
        # tdata=alpha*(seq_data [:, 2])*100
    # plt.plot(xdata)
    # plt.plot(ydata)
    # plt.plot(tdata)
    # plt.show()
    pdata=seq_data [:, 5]
    ldata=seq_data [:, 6]

    num_nodes = 1000
    space = 20
    spt_distance = 300.0
    num_steps = 4
    growths = [1, 3]

    x2 = torch.tensor(pdata).view(-1,1).float()
    y2 = torch.tensor(ldata).long()
    c1 = torch.tensor(xdata).double()#.float()
    c2 = torch.tensor(ydata).double()#.float()
    c3 = torch.tensor(tdata).double()#.float()


    pos2 = torch.stack((c1,  c2 , c3),1)#.view(-1,3)

    edges = radius_graph(pos2, r=spt_distance, max_num_neighbors=20,flow='source_to_target',num_workers= 7)
    seqid=0
    graph_event= Data(x=pos2, y=y2,edge_index= edges, pos=pos2, id_data=id_data, t_org=t_org, seqid=seqid)
    max_value=20

    graph_event_updated=graph_event
    dataset1=(graph_event_updated)
   
    return dataset1
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform )

        self._indices: Optional[Sequence] = None
        
    def indices(self) -> Sequence:
        return range(self.len()) if self._indices is None else self._indices
    @property
    def raw_file_names(self):
        filenames = glob.glob(os.path.join(self.raw_dir, '*.bin'))
        file = [f.split('/')[-1] for f in filenames]
        return file




    
    
    @property

    def processed_file_names(self):
        filenames = glob.glob(os.path.join(self.raw_dir,'../processed/', '*_.pt'))
        file = [f.split('/')[-1] for f in filenames]
        saved_file = [f.replace('.pt','.pt') for f in file]
        return saved_file
    def __len__(self):
        return len(self.processed_file_names)
    def indices(self) -> Sequence:
        return range(self.__len__()) if self._indices is None else self._indices
    
    def download(self):
        if files_exist(self.raw_paths):
            return print('No found data!!!!!!!')


    def process(self):
        data=[]
        Numberofdata=7
        import numpy as np
        seq=[]
        pp=[]
        # print("Yusraaaaaaa HERE")
        for raw_path in self.raw_paths:

            
            sample=raw_path

            pp1=(np.fromfile(sample,  dtype=np.float))
            pp=numpy.reshape(pp1, (Numberofdata, int(len(pp1)/Numberofdata)), order='C')
            pp=numpy.transpose(pp)

            sq1=pp
            sq = numpy.reshape(sq1, (1, int(pp.shape[0]), (pp.shape[1])))
            seq.append(numpy.concatenate(sq))


            deltatime=0.01
            content = Slice_Raw_Events_onTemporal(deltatime, seq)
            print("content", len(content))

            for sample1 in range(0, len(content)-1):
                print(sample1, len(content[sample1]))
                # data=[]
                if len(content[sample1]>0):
                    data=Process_creat_Graphs(content[sample1], 150)
                    # print("1",data)
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    saved_name = raw_path.split('/')[-1].replace('.bin','.pt')
                    torch.save(data, osp.join(self.processed_dir, "sample_"+ str(sample1)+"_"+saved_name  ))

                    print("GRAPH DATA ARE SAVED!!!!!!!!!!!!!!!!!!!!!! in ", self.processed_dir)

        # self.graphs=(data)
    def get(self, idx):

        data = torch.load(osp.join(self.processed_paths[idx]))

        return data


##---------------------------------------------------------------------------------------------------------------------
# Modeling Parameters and Folders to save
# ---------------------------------------------------------------------------------------------------------------------  
print("INITIALIZATION")
seed_val = int(2)
print("Random Seed ID is: ", seed_val)
random.seed(seed_val)
numpy.random.seed(seed_val)
torch.manual_seed(seed_val)
os.environ['PYTHONHASHSEED'] = str(seed_val)

# device = torch.device(  'cpu')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

learningRate=0.001
number_of_epoch=150

#-----------------------
# Path='/home/kucarst3-dlws/Desktop'
# Path='/media/kucarst3-dlws/HDD4/3NEW_MoSeg_Model_5000_dataset_simple/'
# Path='/media/kucarst3-dlws/HDD3/1_evimo2/EVIMO2_Raw_original/GoodLight'
Path="/l/proj/kuin0013/Yusra_Sept/Evimo2_trainingdataset_typeA_10ms/EVIMO2_GL_training/"
train_path_seq6_03 = osp.join(Path+'scene6_dyn_train_03_000000')

train_path_seq9_01 = osp.join(Path+'scene9_dyn_train_01_000000')
train_path_seq9_02 = osp.join(Path+'scene9_dyn_train_02_000000')
train_path_seq9_06 = osp.join(Path+'scene9_dyn_train_06_000000')

train_path_seq10_00 = osp.join(Path+'scene10_dyn_train_00_000000')
train_path_seq10_01 = osp.join(Path+'scene10_dyn_train_01_000000')
train_path_seq10_02 = osp.join(Path+'scene10_dyn_train_02_000000')
train_path_seq10_03 = osp.join(Path+'scene10_dyn_train_03_000000')
train_path_seq10_04 = osp.join(Path+'scene10_dyn_train_04_000000')


train_path_seq12_00 = osp.join(Path+'scene12_dyn_test_00_000000')
train_path_seq12_01 = osp.join(Path+'scene12_dyn_test_01_000000')

train_path_seq13_01 = osp.join(Path+'scene13_dyn_test_01_000000')
train_path_seq13_02 = osp.join(Path+'scene13_dyn_test_02_000000')
train_path_seq13_03 = osp.join(Path+'scene13_dyn_test_03_000000')

train_path_seq14_00 = osp.join(Path+'scene14_dyn_test_00_000000')
train_path_seq14_01 = osp.join(Path+'scene14_dyn_test_01_000000')
train_path_seq14_02 = osp.join(Path+'scene14_dyn_test_02_000000')

train_path_seq15_00 = osp.join(Path+'scene15_dyn_test_00_000000')
train_path_seq15_03 = osp.join(Path+'scene15_dyn_test_03_000000')
train_path_seq15_04 = osp.join(Path+'scene15_dyn_test_04_000000')
train_path_seq15_05 = osp.join(Path+'scene15_dyn_test_06_000000')


# Path2='/media/kucarst3-dlws/HDD3/1_evimo2/EVIMO2_Raw_original/LowLight'
# test_path_seq16_01 = osp.join(Path2+'/eval/seq16_01')
# test_path_seq16_04 = osp.join(Path2+'/eval/seq06_04')



train_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomFlip(axis=0, p=0.3), T.RandomScale([0.95,0.999]) ])
test_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomFlip(axis=0, p=0.3), T.RandomScale([0.95,0.999]) ])


# CREATE FOLDERS TO SAVE RESULTS 
discpt='6PointTransfomer'
FOLDERTOSAVE = '6TrainingResults/1_CASE_PointTransfomer'+discpt+'_LR_'+str(learningRate)+"_EPOCH_"+str(number_of_epoch)+'/'
if not os.path.isdir(FOLDERTOSAVE):
    os.makedirs(FOLDERTOSAVE)

##---------------------------------------------------------------------------------------------------------------------
# STAGE A: CREATING TRAINING AND TESTING DATASET
# ---------------------------------------------------------------------------------------------------------------------  
print("STAGE A: CREATING TRAINING AND TESTING DATASET")
tic()


train_seq6_03 = MyOwnDataset(train_path_seq6_03, transform=train_data_aug)      #### transform=T.Cartesian()

train_seq9_01 = MyOwnDataset(train_path_seq9_01, transform=train_data_aug)      #### transform=T.Cartesian()
train_seq9_06 = MyOwnDataset(train_path_seq9_06, transform=train_data_aug)      #### transform=T.Cartesian()
train_seq9_06 = MyOwnDataset(train_path_seq9_06, transform=train_data_aug)      #### transform=T.Cartesian()


train_seq10_00 = MyOwnDataset(train_path_seq10_00, transform=train_data_aug)      #### transform=T.Cartesian()
train_seq10_01 = MyOwnDataset(train_path_seq10_01, transform=train_data_aug)      #### transform=T.Cartesian()
train_seq10_02 = MyOwnDataset(train_path_seq10_02, transform=train_data_aug)      #### transform=T.Cartesian()
train_seq10_03 = MyOwnDataset(train_path_seq10_03, transform=train_data_aug)      #### transform=T.Cartesian()
train_seq10_04 = MyOwnDataset(train_path_seq10_04, transform=train_data_aug)      #### transform=T.Cartesian()



train_seq12_00 = MyOwnDataset(train_path_seq12_00, transform=train_data_aug)      #### transform=T.Cartesian()
train_seq12_01 = MyOwnDataset(train_path_seq12_01, transform=train_data_aug)      #### transform=T.Cartesian()


train_seq13_01 = MyOwnDataset(train_path_seq13_01, transform=train_data_aug)      #### transform=T.Cartesian()
train_seq13_02 = MyOwnDataset(train_path_seq13_02, transform=train_data_aug)      #### transform=T.Cartesian()
train_seq13_03 = MyOwnDataset(train_path_seq13_03, transform=train_data_aug)      #### transform=T.Cartesian()



train_seq14_00 = MyOwnDataset(train_path_seq14_00, transform=train_data_aug)      #### transform=T.Cartesian()
train_seq14_01 = MyOwnDataset(train_path_seq14_01, transform=train_data_aug)      #### transform=T.Cartesian()
train_seq14_02 = MyOwnDataset(train_path_seq14_02, transform=train_data_aug) 





train_seq15_00 = MyOwnDataset(train_path_seq15_00, transform=train_data_aug)      #### transform=T.Cartesian()
train_seq15_03 = MyOwnDataset(train_path_seq15_03, transform=train_data_aug)      #### transform=T.Cartesian()
train_seq15_04 = MyOwnDataset(train_path_seq15_04, transform=train_data_aug) 
train_seq15_05 = MyOwnDataset(train_path_seq15_05, transform=train_data_aug) 


print('train_seq6_03', train_seq6_03)
print('train_seq9_01', train_seq9_01)
print('train_seq9_06', train_seq9_06)
print('train_seq9_06', train_seq9_06)

print('train_seq10_00', train_seq10_00)
print('train_seq10_01', train_seq10_01)
print('train_seq10_02', train_seq10_02)
print('train_seq10_03', train_seq10_03)
print('train_seq10_04', train_seq10_04)

print('train_seq12_00', train_seq12_00)
print('train_seq12_01', train_seq12_01)

print('train_seq13_01', train_seq13_01)
print('train_seq13_02', train_seq13_02)
print('train_seq13_03', train_seq13_03)
print('train_seq14_00', train_seq14_00)
print('train_seq14_01', train_seq14_01)
print('train_seq14_02', train_seq14_02)



print('train_seq15_00', train_seq15_00)
print('train_seq15_03', train_seq15_03)
print('train_seq15_04', train_seq15_04)
print('train_seq15_05', train_seq15_05)






# testgraphs_wall=test_Wall_seq00_p1+test_Wall_seq00_p2+test_Wall_seq00_p3+test_Wall_seq00_p4+test_Wall_seq00_p5+test_Wall_seq00_p6+test_Wall_seq01_p1+test_Wall_seq01_p2+test_Wall_seq01_p3+test_Wall_seq01_p4+test_Wall_seq01_p5+test_Wall_seq01_p6
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

training_evimo2_all=   train_seq6_03+   train_seq9_01+train_seq9_06+train_seq9_06+train_seq10_00+train_seq10_01+train_seq10_02+ train_seq10_03+ train_seq10_04+train_seq12_00+ train_seq12_01+train_seq13_01+ train_seq13_02+ train_seq13_03+train_seq14_00+train_seq14_01+ train_seq14_02+train_seq15_00+train_seq15_03+train_seq15_04+train_seq15_05

evimo2_seq6_9=train_seq6_03+   train_seq9_01+train_seq9_06+train_seq9_06
evimo2_seq10=train_seq10_00+train_seq10_01+train_seq10_02+ train_seq10_03+ train_seq10_04
evimo2_seq12_13=train_seq12_00+ train_seq12_01+train_seq13_01+ train_seq13_02+ train_seq13_03

evimo2_seq14_15=train_seq14_00+train_seq14_01+ train_seq14_02+train_seq15_00+train_seq15_03+train_seq15_04+train_seq15_05



TrainGraphs_all=training_evimo2_all
#traingraphs_room1_obj1_allseqs+traingraphs_room1_obj2_allseqs+traingraphs_room1_obj3_allseqs+traingraphs_room3_obj1_allseqs+traingraphs_room3_obj2_allseqs+traingraphs_room3_obj3_allseqs
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TrainGraphs_all!!!!!!!!1", len(TrainGraphs_all))



# TestGraphs_all=testgraphs_box+testgraphs_floor+testgraphs_table+testgraphs_fast+testgraphs_wall
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TestGraphs_all!!!!!!!!1", len(TestGraphs_all))
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

# print('END!!!-creating graphs')


# # ##---------------------------------------------------------------------------------------------------------------------
# # # STAGE B: Network classes
# # # ---------------------------------------------------------------------------------------------------------------------  
# # from torch_geometric.nn.conv import PointTransformerConv
# # import os.path as osp

# # import torch
# # import torch.nn.functional as F
# # from torch.nn import BatchNorm1d as BN
# # from torch.nn import Identity
# # from torch.nn import Linear as Lin
# # from torch.nn import ReLU
# # from torch.nn import Sequential as Seq
# # from torch_cluster import fps, knn_graph
# # from torch_scatter import scatter_max

# # import torch_geometric.transforms as T
# # from torch_geometric.datasets import ModelNet
# # from torch_geometric.loader import DataLoader
# # from torch_geometric.nn import global_mean_pool
# # from torch_geometric.nn.conv import PointTransformerConv
# # from torch_geometric.nn.pool import knn
# # from torch_geometric.nn.unpool import knn_interpolate



# # class TransformerBlock(torch.nn.Module):
# #     def __init__(self, in_channels, out_channels):
# #         super().__init__()
# #         self.lin_in = Lin(in_channels, in_channels)
# #         self.lin_out = Lin(out_channels, out_channels)

# #         self.pos_nn = MLP([3, 64, out_channels], batch_norm=False)

# #         self.attn_nn = MLP([out_channels, 64, out_channels], batch_norm=False)

# #         self.transformer = PointTransformerConv(in_channels, out_channels,
# #                                                 pos_nn=self.pos_nn,
# #                                                 attn_nn=self.attn_nn)

# #     def forward(self, x, pos, edge_index):
# #         x = self.lin_in(x).relu()
# #         x = self.transformer(x, pos, edge_index)
# #         x = self.lin_out(x).relu()
# #         return x




# # class TransitionDown(torch.nn.Module):
# #     '''
# #         Samples the input point cloud by a ratio percentage to reduce
# #         cardinality and uses an mlp to augment features dimensionnality
# #     '''
# #     def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
# #         super().__init__()
# #         self.k = k
# #         self.ratio = ratio
# #         self.mlp = MLP([in_channels, out_channels])

# #     def forward(self, x, pos, batch):
# #         # FPS sampling
# #         id_clusters = fps(pos, ratio=self.ratio, batch=batch)

# #         # compute for each cluster the k nearest points
# #         sub_batch = batch[id_clusters] if batch is not None else None

# #         # beware of self loop
# #         id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
# #                             batch_y=sub_batch)

# #         # transformation of features through a simple MLP
# #         x = self.mlp(x)

# #         # Max pool onto each cluster the features from knn in points
# #         x_out, _ = scatter_max(x[id_k_neighbor[1]], id_k_neighbor[0],
# #                                dim_size=id_clusters.size(0), dim=0)

# #         # keep only the clusters and their max-pooled features
# #         sub_pos, out = pos[id_clusters], x_out
# #         return out, sub_pos, sub_batch


# # def MLP(channels, batch_norm=True):
# #     return Seq(*[
# #         Seq(Lin(channels[i - 1], channels[i]),
# #             BN(channels[i]) if batch_norm else Identity(), ReLU())
# #         for i in range(1, len(channels))
# #     ])
# # class TransitionUp(torch.nn.Module):
# #     '''
# #         Reduce features dimensionnality and interpolate back to higher
# #         resolution and cardinality
# #     '''
# #     def __init__(self, in_channels, out_channels):
# #         super().__init__()
# #         self.mlp_sub = MLP([in_channels, out_channels])
# #         self.mlp = MLP([out_channels, out_channels])

# #     def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
# #         # transform low-res features and reduce the number of features
# #         x_sub = self.mlp_sub(x_sub)

# #         # interpolate low-res feats to high-res points
# #         x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=3,
# #                                          batch_x=batch_sub, batch_y=batch)

# #         x = self.mlp(x) + x_interpolated

# #         return x
# # class Net_PointTransformer(torch.nn.Module):
# #     def __init__(self, in_channels, out_channels, dim_model, k=16):
# #         super().__init__()
# #         self.k = k

# #         # dummy feature is created if there is none given
# #         in_channels = max(in_channels, 1)

# #         # first block
# #         self.mlp_input = MLP([in_channels, dim_model[0]])

# #         self.transformer_input = TransformerBlock(
# #             in_channels=dim_model[0],
# #             out_channels=dim_model[0],
# #         )

# #         # backbone layers
# #         self.transformers_up = torch.nn.ModuleList()
# #         self.transformers_down = torch.nn.ModuleList()
# #         self.transition_up = torch.nn.ModuleList()
# #         self.transition_down = torch.nn.ModuleList()

# #         for i in range(0, len(dim_model) - 1):

# #             # Add Transition Down block followed by a Point Transformer block
# #             self.transition_down.append(
# #                 TransitionDown(in_channels=dim_model[i],
# #                                out_channels=dim_model[i + 1], k=self.k))

# #             self.transformers_down.append(
# #                 TransformerBlock(in_channels=dim_model[i + 1],
# #                                  out_channels=dim_model[i + 1]))

# #             # Add Transition Up block followed by Point Transformer block
# #             self.transition_up.append(
# #                 TransitionUp(in_channels=dim_model[i + 1],
# #                              out_channels=dim_model[i]))

# #             self.transformers_up.append(
# #                 TransformerBlock(in_channels=dim_model[i],
# #                                  out_channels=dim_model[i]))

# #         # summit layers
# #         self.mlp_summit = MLP([dim_model[-1], dim_model[-1]], batch_norm=False)

# #         self.transformer_summit = TransformerBlock(
# #             in_channels=dim_model[-1],
# #             out_channels=dim_model[-1],
# #         )

# #         # class score computation
# #         self.mlp_output = Seq(Lin(dim_model[0], 64), ReLU(), Lin(64, 64),
# #                               ReLU(), Lin(64, out_channels))

# #     def forward(self, x, pos, batch=None):

# #         # add dummy features in case there is none
# #         if x is None:
# #             x = torch.ones((pos.shape[0], 1)).to(pos.get_device())

# #         out_x = []
# #         out_pos = []
# #         out_batch = []

# #         # first block
# #         x = self.mlp_input(x)
# #         edge_index = knn_graph(pos, k=self.k, batch=batch)
# #         x = self.transformer_input(x, pos, edge_index)

# #         # save outputs for skipping connections
# #         out_x.append(x)
# #         out_pos.append(pos)
# #         out_batch.append(batch)

# #         # backbone down : #reduce cardinality and augment dimensionnality
# #         for i in range(len(self.transformers_down)):
# #             x, pos, batch = self.transition_down[i](x, pos, batch=batch)
# #             edge_index = knn_graph(pos, k=self.k, batch=batch)
# #             x = self.transformers_down[i](x, pos, edge_index)

# #             out_x.append(x)
# #             out_pos.append(pos)
# #             out_batch.append(batch)

# #         # summit
# #         x = self.mlp_summit(x)
# #         edge_index = knn_graph(pos, k=self.k, batch=batch)
# #         x = self.transformer_summit(x, pos, edge_index)

# #         # backbone up : augment cardinality and reduce dimensionnality
# #         n = len(self.transformers_down)
# #         for i in range(n):
# #             x = self.transition_up[-i - 1](x=out_x[-i - 2], x_sub=x,
# #                                            pos=out_pos[-i - 2],
# #                                            pos_sub=out_pos[-i - 1],
# #                                            batch_sub=out_batch[-i - 1],
# #                                            batch=out_batch[-i - 2])

# #             edge_index = knn_graph(out_pos[-i - 2], k=self.k,
# #                                    batch=out_batch[-i - 2])
# #             x = self.transformers_up[-i - 1](x, out_pos[-i - 2], edge_index)

# #         # Class score
# #         out = self.mlp_output(x)

# #         return F.softmax(out, dim=1)



# # transform = T.Cartesian(cat=False)

# # class Net_TRANSFORMER(torch.nn.Module):
# #     def __init__(self):
# #         super(Net_TRANSFORMER, self).__init__()
# #         # self.bn1 = torch.nn.BatchNorm1d(64)
        
# #         # self.conv2 = SAGEConv(64, 64)
# #         # self.bn2 = torch.nn.BatchNorm1d(64)
        
# #         # self.conv3 = SAGEConv(64, 64)
# #         # self.bn3 = torch.nn.BatchNorm1d(64)
# #         # self.conv4 = SAGEConv(64, 64)
# #         # self.bn4 = torch.nn.BatchNorm1d(64)
        
# #         # self.conv5 = SAGEConv(64, 64)
# #         # self.bn5 = torch.nn.BatchNorm1d(64)
        
# #         # self.conv6 = SAGEConv(64, 2)
# #         # self.bn6 = torch.nn.BatchNorm1d(64)
        
# #         # self.fc1 = torch.nn.Linear(32*512, 1024)
# #         # self.fc2 = torch.nn.Linear(1024, 2)

# #  ########################################################################################

# #         feature_size=3
# #         self.encoder_embedding_size=64
# #         self.encoder_embedding_size1=32
# #         self.encoder_embedding_size2=16

        

# #         self.edge_dim=3
# #         self.conv1 = TransformerConv(feature_size, 
# #                                     self.encoder_embedding_size, 
# #                                     heads=4, 
# #                                     concat=False,
# #                                     beta=True,
# #                                     edge_dim=self.edge_dim)       

# #         self.conv2 = TransformerConv(self.encoder_embedding_size, 
# #                                     self.encoder_embedding_size1, 
# #                                     heads=4, 
# #                                     concat=False,
# #                                     beta=True,
# #                                     edge_dim=self.edge_dim)  

# #         self.conv3 = TransformerConv(self.encoder_embedding_size1, 
# #                                     self.encoder_embedding_size2, 
# #                                     heads=4, 
# #                                     concat=False,
# #                                     beta=True,
# #                                     edge_dim=self.edge_dim)   
# #         # self.conv4 = TransformerConv(self.encoder_embedding_size1, 
# #         #                             self.encoder_embedding_size2, 
# #         #                             heads=4, 
# #         #                             concat=False,
# #         #                             beta=True,
# #         #                             edge_dim=self.edge_dim)    
# #         # self.conv5 = TransformerConv(self.encoder_embedding_size2, 
# #         #                             self.encoder_embedding_size2, 
# #         #                             heads=16, 
# #         #                             concat=False,
# #         #                             beta=True,
# #         #                             edge_dim=self.edge_dim)    
        
# #         self.conv6 = TransformerConv(self.encoder_embedding_size2, 
# #                                     2, 
# #                                     heads=4, 
# #                                     concat=False,
# #                                     beta=True,
# #                                     edge_dim=self.edge_dim)    


# #         self.bn1 = torch.nn.BatchNorm1d(self.encoder_embedding_size)
# #         self.bn2 = torch.nn.BatchNorm1d(self.encoder_embedding_size1)
# #         self.bn3 = torch.nn.BatchNorm1d(self.encoder_embedding_size2)


# #     def forward(self, data):
 
# #         data.x=self.conv1(data.x.double(), data.edge_index, data.edge_attr.double())
# #         data.x = torch.sigmoid(data.x)
# #         data.x = self.bn1(data.x) 

# #         data.x=self.conv2(data.x.double(), data.edge_index, data.edge_attr.double())
# #         data.x = torch.sigmoid(data.x)
# #         data.x = self.bn2(data.x) 

# #         data.x=self.conv3(data.x.double(), data.edge_index, data.edge_attr.double())
# #         data.x = torch.sigmoid(data.x)
# #         data.x = self.bn3(data.x) 

# #         x = torch.sigmoid(self.conv6(data.x.double(), data.edge_index, data.edge_attr.double()))
# #         OUT=F.softmax(x)
# #         # print("x",F.softmax(x))
# #         return OUT

# # class NetConnect_1(torch.nn.Module):
# #     def __init__(self):
# #         super(NetConnect_1, self).__init__()
# #         #self.conv1 = SplineConv(1, 64, dim=3, kernel_size=4)
# #         ##print("Kholous", len(data))
# #         self.conv0 = GCNConv(3, 16)
# #         self.bn0 = torch.nn.BatchNorm1d(16)
        
# #         self.conv1 = GCNConv(16, 64)
# #         self.bn1 = torch.nn.BatchNorm1d(64)
# #         self.conv2 = GCNConv(64, 128)
# #         self.bn2 = torch.nn.BatchNorm1d(128)
        
        
# #         self.conv3 = GCNConv(128, 256)
# #         self.bn3 = torch.nn.BatchNorm1d(256)
# #         #self.bn1 = BatchNorm(64)

# #         self.conv4 = GCNConv(256, 128)
# #         self.bn4 = torch.nn.BatchNorm1d(128)
        
# #         self.conv5 = GCNConv(128*2, 64)
# #         self.bn5 = torch.nn.BatchNorm1d(64)
   

# #         self.conv6 = GCNConv(64*2, 16)
# #         self.bn6 = torch.nn.BatchNorm1d(16)
        
# #         self.conv7 = GCNConv(16*2, 2)
# #         self.bn7 = torch.nn.BatchNorm1d(2)


# # #        self.bn4 = torch.nn.BatchNorm1d(512)
# # #         self.conv8 = GCNConv(16, 2)
# #         #self.bn4 = torch.nn.BatchNorm1d(2)  
# # #         self.fc1 = torch.nn.Linear(32*512, 1024)
# # #         self.fc2 = torch.nn.Linear(1024, 2)

        


# #     def forward(self, data):
# #         # data.x = F.leaky_relu(self.conv0(data.x, data.edge_index))
# #         data.x = torch.sigmoid(self.conv0(data.x, data.edge_index))




# #         # data.x = F.leaky_relu(self.conv0(torch.tensor(data.x.clone().detach(), dtype=torch.float32), data.edge_index))
# #         data.x = self.bn0(data.x)      
# #         part0=data.x
# #         data.x = torch.sigmoid(self.conv1( data.x , data.edge_index))
# #         # data.x = F.leaky_relu(self.conv1( data.x , data.edge_index))

# #         # data.x = F.leaky_relu(self.conv1(torch.tensor(data.x.clone().detach(), dtype=torch.float32), data.edge_index))
# #         data.x = self.bn1(data.x)
# #         part1=data.x
# #         data.x = torch.sigmoid(self.conv2(data.x, data.edge_index))
# #         # data.x = F.leaky_relu(self.conv2(data.x, data.edge_index))

# #         data.x = self.bn2(data.x)
# #         part2=data.x
# #         # data.x = F.leaky_relu(self.conv3(data.x, data.edge_index))
# #         data.x = torch.sigmoid(self.conv3(data.x, data.edge_index))

# #         data.x = self.bn3(data.x)  
        
# #         # data.x = F.leaky_relu(self.conv4(data.x, data.edge_index))
# #         data.x = torch.sigmoid(self.conv4(data.x, data.edge_index))

# #         data.x = self.bn4(data.x) #concat with part2
# #         data.x=torch.cat((data.x, part2), dim=1)
# #         # data.x = F.leaky_relu(self.conv5(data.x, data.edge_index))
# #         data.x = torch.sigmoid(self.conv5(data.x, data.edge_index))

# #         data.x = self.bn5(data.x) #concat with part1
# #         data.x=torch.cat((data.x, part1), dim=1)
# #         # data.x = F.leaky_relu(self.conv6(data.x, data.edge_index))

# #         data.x = torch.sigmoid(self.conv6(data.x, data.edge_index))
# #         data.x = self.bn6(data.x) #concat with part0
# #         data.x=torch.cat((data.x, part0), dim=1)

# #         data.x = torch.sigmoid(self.conv7(data.x, data.edge_index))
# #         data.x = self.bn7(data.x)
# # #         data.x = F.sigmoid(self.conv8(data.x, data.edge_index))

# #         #data.x = F.elu(self.conv4(data.x, data.edge_index))
# #         #data.x = self.bn4(data.x)
# #         #cluster = voxel_grid(data.pos, data.batch, size=[32,32])
# #         #x = max_pool_x(cluster, data.x, data.batch, size=32)
# #         ##print("abdulrahman done",x.size())

# #         #x = x.view(-1, self.fc1.weight.size(1))
# #         ##print("abdulrahman2 done",x.size())

# #         #x = F.elu(self.fc1(x))
# #         ##print("abdulrahman3 done",x.size())

# #         #x = F.dropout(x, training=self.training)
# #         #x = self.fc2(x)
# #         ##print("network done",x.size())
# #         out=F.softmax(data.x, dim=1)
# #         #print("helooooo",(out))
# #         return out#data.x

# # class NetConnect_2(torch.nn.Module):
# #     def __init__(self):
# #         super(NetConnect_2, self).__init__()
# #         #self.conv1 = SplineConv(1, 64, dim=3, kernel_size=4)
# #         ##print("Kholous", len(data))
# #         self.conv0 = GCNConv(3, 16)
# #         self.bn0 = torch.nn.BatchNorm1d(16)
        
# #         self.conv1 = GCNConv(16, 64)
# #         self.bn1 = torch.nn.BatchNorm1d(64)


# #         self.conv2 = GCNConv(64, 128)
# #         self.bn2 = torch.nn.BatchNorm1d(128)
        
# #         self.conv3 = GCNConv(128, 256)
# #         self.bn3 = torch.nn.BatchNorm1d(256)
# #         #self.bn1 = BatchNorm(64)

# #         self.conv4 = GCNConv(256, 512)
# #         self.bn4 = torch.nn.BatchNorm1d(512)


# #         self.conv5 = GCNConv(512, 256)
# #         self.bn5 = torch.nn.BatchNorm1d(256)


# #         self.conv6 = GCNConv(256*2, 128)
# #         self.bn6 = torch.nn.BatchNorm1d(128)
   

# #         self.conv7 = GCNConv(128*2, 64)
# #         self.bn7 = torch.nn.BatchNorm1d(64)
        
# #         self.conv8 = GCNConv(64*2, 16)
# #         self.bn8 = torch.nn.BatchNorm1d(16)


# #         self.conv9 = GCNConv(16*2, 2)
# #         self.bn9 = torch.nn.BatchNorm1d(2)



# #     def forward(self, data):
# #         # data.x = F.leaky_relu(self.conv0(data.x, data.edge_index))
# #         data.x = torch.sigmoid(self.conv0(data.x, data.edge_index))

# #         # data.x = F.leaky_relu(self.conv0(torch.tensor(data.x.clone().detach(), dtype=torch.float32), data.edge_index))
# #         data.x = self.bn0(data.x)      
# #         part0=data.x
# #         data.x = torch.sigmoid(self.conv1( data.x , data.edge_index))
# #         # data.x = F.leaky_relu(self.conv1( data.x , data.edge_index))

# #         # data.x = F.leaky_relu(self.conv1(torch.tensor(data.x.clone().detach(), dtype=torch.float32), data.edge_index))
# #         data.x = self.bn1(data.x)
# #         part1=data.x
# #         data.x = torch.sigmoid(self.conv2(data.x, data.edge_index))
# #         # data.x = F.leaky_relu(self.conv2(data.x, data.edge_index))

# #         data.x = self.bn2(data.x)
# #         part2=data.x
# #         # data.x = F.leaky_relu(self.conv3(data.x, data.edge_index))
# #         data.x = torch.sigmoid(self.conv3(data.x, data.edge_index))
# #         data.x = self.bn3(data.x)  
# #         part3=data.x

# #         # data.x = F.leaky_relu(self.conv4(data.x, data.edge_index))
# #         data.x = torch.sigmoid(self.conv4(data.x, data.edge_index))

# #         data.x = self.bn4(data.x) #concat with part2


# #         data.x = torch.sigmoid(self.conv5(data.x, data.edge_index))
# #         data.x = self.bn5(data.x) #concat with part2


# #         data.x=torch.cat((data.x, part3), dim=1)
# #         # data.x = F.leaky_relu(self.conv5(data.x, data.edge_index))
# #         data.x = torch.sigmoid(self.conv6(data.x, data.edge_index))

# #         data.x = self.bn6(data.x) #concat with part1


# #         data.x=torch.cat((data.x, part2), dim=1)
# #         # data.x = F.leaky_relu(self.conv6(data.x, data.edge_index))

# #         data.x = torch.sigmoid(self.conv7(data.x, data.edge_index))
# #         data.x = self.bn7(data.x) #concat with part0
# #         data.x=torch.cat((data.x, part1), dim=1)


# #         data.x = torch.sigmoid(self.conv8(data.x, data.edge_index))
# #         data.x = self.bn8(data.x) #concat with part0
# #         data.x=torch.cat((data.x, part0), dim=1)

# #         data.x = torch.sigmoid(self.conv9(data.x, data.edge_index))
# #         data.x = self.bn9(data.x)
# # #         data.x = F.sigmoid(self.conv8(data.x, data.edge_index))

# #         #data.x = F.elu(self.conv4(data.x, data.edge_index))
# #         #data.x = self.bn4(data.x)
# #         #cluster = voxel_grid(data.pos, data.batch, size=[32,32])
# #         #x = max_pool_x(cluster, data.x, data.batch, size=32)
# #         ##print("abdulrahman done",x.size())

# #         #x = x.view(-1, self.fc1.weight.size(1))
# #         ##print("abdulrahman2 done",x.size())

# #         #x = F.elu(self.fc1(x))
# #         ##print("abdulrahman3 done",x.size())

# #         #x = F.dropout(x, training=self.training)
# #         #x = self.fc2(x)
# #         ##print("network done",x.size())
# #         out=F.softmax(data.x, dim=1)
# #         #print("helooooo",(out))
# #         return out#data.x


# # class NetConnect_3(torch.nn.Module):
# #     def __init__(self):
# #         super(NetConnect_3, self).__init__()
# #         #self.conv1 = SplineConv(1, 64, dim=3, kernel_size=4)
# #         ##print("Kholous", len(data))
# #         self.conv0 = GCNConv(3, 64)
# #         self.bn0 = torch.nn.BatchNorm1d(64)
        
# #         self.conv1 = GCNConv(64, 64)
# #         self.bn1 = torch.nn.BatchNorm1d(64)
# #         self.conv2 = GCNConv(64, 64)
# #         self.bn2 = torch.nn.BatchNorm1d(64)
        
        
# #         self.conv3 = GCNConv(64, 64)
# #         self.bn3 = torch.nn.BatchNorm1d(64)
# #         #self.bn1 = BatchNorm(64)

# #         self.conv4 = GCNConv(64, 64)
# #         self.bn4 = torch.nn.BatchNorm1d(64)
        


# #         self.conv5 = GCNConv(64*5, 256)
# #         self.bn5 = torch.nn.BatchNorm1d(256)
   

# #         self.conv6 = GCNConv(256, 64)
# #         self.bn6 = torch.nn.BatchNorm1d(64)
        
# #         self.conv7 = GCNConv(64, 16)
# #         self.bn7 = torch.nn.BatchNorm1d(16)

# #         self.conv8 = GCNConv(16, 4)
# #         self.bn8 = torch.nn.BatchNorm1d(4)        
# #         self.conv9 = GCNConv(4, 2)
# #         self.bn9 = torch.nn.BatchNorm1d(2)
# # #        self.bn4 = torch.nn.BatchNorm1d(512)
# # #         self.conv8 = GCNConv(16, 2)
# #         #self.bn4 = torch.nn.BatchNorm1d(2)  
# # #         self.fc1 = torch.nn.Linear(32*512, 1024)
# # #         self.fc2 = torch.nn.Linear(1024, 2)

        


# #     def forward(self, data):
# #         # data.x = F.leaky_relu(self.conv0(data.x, data.edge_index))
# #         data.x = torch.sigmoid(self.conv0(data.x, data.edge_index))
# #         # data.x = F.leaky_relu(self.conv0(torch.tensor(data.x.clone().detach(), dtype=torch.float32), data.edge_index))
# #         data.x = self.bn0(data.x)      
# #         part0=data.x
# #         data.x = torch.sigmoid(self.conv1( data.x , data.edge_index))
# #         # data.x = F.leaky_relu(self.conv1( data.x , data.edge_index))

# #         # data.x = F.leaky_relu(self.conv1(torch.tensor(data.x.clone().detach(), dtype=torch.float32), data.edge_index))
# #         data.x = self.bn1(data.x)
# #         part1=data.x
# #         data.x = torch.sigmoid(self.conv2(data.x, data.edge_index))
# #         # data.x = F.leaky_relu(self.conv2(data.x, data.edge_index))

# #         data.x = self.bn2(data.x)
# #         part2=data.x
# #         # data.x = F.leaky_relu(self.conv3(data.x, data.edge_index))
# #         data.x = torch.sigmoid(self.conv3(data.x, data.edge_index))

# #         data.x = self.bn3(data.x)  
# #         part3=data.x

# #         # data.x = F.leaky_relu(self.conv4(data.x, data.edge_index))
# #         data.x = torch.sigmoid(self.conv4(data.x, data.edge_index))

# #         data.x = self.bn4(data.x) #concat with part2
# #         part4=data.x




# #         data.x=torch.cat((part4, part3, part2, part1 , part0), dim=1)
# #         # data.x = F.leaky_relu(self.conv5(data.x, data.edge_index))
# #         data.x = torch.sigmoid(self.conv5(data.x, data.edge_index))

# #         data.x = self.bn5(data.x) #concat with part1
# #         # data.x = F.leaky_relu(self.conv6(data.x, data.edge_index))

# #         data.x = torch.sigmoid(self.conv6(data.x, data.edge_index))
# #         data.x = self.bn6(data.x) #concat with part0

# #         data.x = torch.sigmoid(self.conv7(data.x, data.edge_index))
# #         data.x = self.bn7(data.x)


# #         data.x = torch.sigmoid(self.conv8(data.x, data.edge_index))
# #         data.x = self.bn8(data.x)
# #         data.x = torch.sigmoid(self.conv9(data.x, data.edge_index))
# #         data.x = self.bn9(data.x)

# # #         data.x = F.sigmoid(self.conv8(data.x, data.edge_index))

# #         #data.x = F.elu(self.conv4(data.x, data.edge_index))
# #         #data.x = self.bn4(data.x)
# #         #cluster = voxel_grid(data.pos, data.batch, size=[32,32])
# #         #x = max_pool_x(cluster, data.x, data.batch, size=32)
# #         ##print("abdulrahman done",x.size())

# #         #x = x.view(-1, self.fc1.weight.size(1))
# #         ##print("abdulrahman2 done",x.size())

# #         #x = F.elu(self.fc1(x))
# #         ##print("abdulrahman3 done",x.size())

# #         #x = F.dropout(x, training=self.training)
# #         #x = self.fc2(x)
# #         ##print("network done",x.size())
# #         out=F.softmax(data.x, dim=1)
# #         #print("helooooo",(out))
# #         return out#data.x



# # ##---------------------------------------------------------------------------------------------------------------------
# # # STAGE C: Network parameters
# # # ---------------------------------------------------------------------------------------------------------------------  

# # print("STAGE C: BUILDING A NETWORK")
# # # model=Net_TRANSFORMER().to(device)
# # # model = Net_PointTransformer(3,2, dim_model=[32, 64, 128, 256, 512], k=16)
# # # model = Net_PointTransformer(3,2, dim_model=[32, 64, 128, 256 ], k=16)
# # model = Net_PointTransformer(3,2, dim_model=[64, 64 ], k=16)

# # # model = NetConnect_3().to(device)
# # model=model.double()
# # print("Model Structure ",model)
# # model=nn.DataParallel(model, device_ids=[0,1])
# # model.to(device)
# # optimizer = torch.optim.Adam(model.parameters(), lr=learningRate) #torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)

# # # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# # ##---------------------------------------------------------------------------------------------------------------------
# # # STAGE D: TRAINING STAGE
# # # ---------------------------------------------------------------------------------------------------------------------  
# # print("STAGE D: TRAINING STAGE - Feedforward")
# # loss_func=FocalLoss()
# # train_dataset= train_Box_seq00_p1+train_Box_seq00_p2+train_Box_seq00_p3+train_Box_seq00_p4+train_Box_seq00_p5+train_Box_seq00_p6+train_Box_seq01_p1+train_Box_seq01_p2+train_Box_seq01_p3+train_Box_seq01_p4+train_Box_seq01_p5+ train_Box_seq01_p6+train_Box_seq02_p1+train_Box_seq02_p2+train_Box_seq02_p3+train_Box_seq02_p4+ train_Box_seq02_p5+train_Box_seq02_p6+train_Box_seq03_p1+train_Box_seq03_p2+train_Box_seq03_p3+train_Box_seq03_p4+train_Box_seq03_p5+train_Box_seq03_p6+train_Box_seq04_p1+train_Box_seq04_p2+train_Box_seq04_p3+train_Box_seq04_p4+train_Box_seq04_p5+train_Box_seq04_p6+train_Box_seq05_p1+train_Box_seq05_p2+train_Box_seq05_p3+train_Box_seq05_p4+train_Box_seq05_p5+train_Box_seq05_p6+train_Box_seq06_p1+train_Box_seq06_p2+train_Box_seq06_p3+train_Box_seq06_p4+train_Box_seq06_p5+train_Box_seq06_p6+train_Box_seq07_p1+ train_Box_seq07_p2+ train_Box_seq07_p3+ train_Box_seq07_p4+train_Box_seq07_p5+train_Box_seq07_p6+train_Box_seq08_p1+train_Box_seq08_p2+  train_Box_seq08_p3+ train_Box_seq08_p4+  train_Box_seq08_p5+  train_Box_seq08_p6+ train_Box_seq09_p1+  train_Box_seq09_p2+  train_Box_seq09_p3+  train_Box_seq09_p4+  train_Box_seq09_p5+  train_Box_seq09_p6+train_Box_seq10_p1+train_Box_seq10_p2+train_Box_seq10_p3+train_Box_seq10_p4+train_Box_seq10_p5+train_Box_seq10_p6+train_Box_seq11_p1+train_Box_seq11_p2+train_Box_seq11_p3+train_Box_seq11_p4+train_Box_seq11_p5+train_Box_seq11_p6+ train_Table_seq00_p1+  train_Table_seq00_p2+ train_Table_seq00_p3+ train_Table_seq00_p4+ train_Table_seq00_p5+train_Table_seq00_p6+train_Table_seq01_p1+ train_Table_seq01_p2+ train_Table_seq01_p3+ train_Table_seq01_p4+ train_Table_seq01_p5+train_Table_seq01_p6+train_Table_seq02_p1+train_Table_seq02_p2+train_Table_seq02_p3+ train_Table_seq02_p4+ train_Table_seq02_p5+  train_Table_seq02_p6+  train_Table_seq03_p1+train_Table_seq03_p2+train_Table_seq03_p3 +train_Table_seq03_p3+ train_Table_seq03_p4+train_Table_seq03_p5+train_Table_seq03_p6+ train_Table_seq04_p1+  train_Table_seq04_p2+  train_Table_seq04_p3+  train_Table_seq04_p4+  train_Table_seq04_p5+ train_Table_seq04_p6+  train_Table_seq05_p1+   train_Table_seq05_p2+  train_Table_seq05_p3+  train_Table_seq05_p4+ train_Table_seq05_p5+ train_Table_seq05_p6+  train_Floor_seq00_p1+  train_Floor_seq00_p2+ train_Floor_seq00_p3+  train_Floor_seq00_p4+  train_Floor_seq00_p5+ train_Floor_seq00_p6+  train_Floor_seq01_p1+  train_Floor_seq01_p2+  train_Floor_seq01_p3+  train_Floor_seq01_p4+  train_Floor_seq01_p5+ train_Floor_seq01_p6+ train_Floor_seq02_p1+ train_Floor_seq02_p2+train_Floor_seq02_p3+train_Floor_seq02_p4+train_Floor_seq02_p5+train_Floor_seq02_p6+  train_wall_seq00_p1+  train_wall_seq00_p2+  train_wall_seq00_p3+  train_wall_seq00_p4+ train_wall_seq00_p5+ train_wall_seq00_p6+  train_wall_seq01_p1+ train_wall_seq01_p2+ train_wall_seq01_p3+ train_wall_seq01_p4+ train_wall_seq01_p5+train_wall_seq01_p6+train_wall_seq02_p1+train_wall_seq02_p2+train_wall_seq02_p3+train_wall_seq02_p4+train_wall_seq02_p5+train_wall_seq02_p6

# # print("rana!!!!!!!!!!!!!!!!!!!!!", len(train_dataset))


# # # train_dataset= train_Box_seq03_p1#train_Box_seq00_p1+train_Box_seq00_p2+train_Box_seq00_p3+train_Box_seq00_p4+train_Box_seq00_p5+train_Box_seq00_p6+train_Box_seq01_p1+train_Box_seq01_p2+train_Box_seq01_p3+train_Box_seq01_p4+train_Box_seq01_p5+ train_Box_seq01_p6+train_Box_seq02_p1+train_Box_seq02_p2+train_Box_seq02_p3+train_Box_seq02_p4+ train_Box_seq02_p5+train_Box_seq02_p6+train_Box_seq03_p1+train_Box_seq03_p2+train_Box_seq03_p3+train_Box_seq03_p4+train_Box_seq03_p5+train_Box_seq03_p6+train_Box_seq04_p1+train_Box_seq04_p2+train_Box_seq04_p3+train_Box_seq04_p4+train_Box_seq04_p5+train_Box_seq04_p6+train_Box_seq05_p1+train_Box_seq05_p2+train_Box_seq05_p3+train_Box_seq05_p4+train_Box_seq05_p5+train_Box_seq05_p6+train_Box_seq06_p1+train_Box_seq06_p2+train_Box_seq06_p3+train_Box_seq06_p4+train_Box_seq06_p5+train_Box_seq06_p6+train_Box_seq07_p1+ train_Box_seq07_p2+ train_Box_seq07_p3+ train_Box_seq07_p4+train_Box_seq07_p5+train_Box_seq07_p6+train_Box_seq08_p1+train_Box_seq08_p2+  train_Box_seq08_p3+ train_Box_seq08_p4+  train_Box_seq08_p5+  train_Box_seq08_p6+ train_Box_seq09_p1+  train_Box_seq09_p2+  train_Box_seq09_p3+  train_Box_seq09_p4+  train_Box_seq09_p5+  train_Box_seq09_p6+train_Box_seq10_p1+train_Box_seq10_p2+train_Box_seq10_p3+train_Box_seq10_p4+train_Box_seq10_p5+train_Box_seq10_p6+train_Box_seq11_p1+train_Box_seq11_p2+train_Box_seq11_p3+train_Box_seq11_p4+train_Box_seq11_p5+train_Box_seq11_p6+ train_Table_seq00_p1+  train_Table_seq00_p2+ train_Table_seq00_p3+ train_Table_seq00_p4+ train_Table_seq00_p5+train_Table_seq00_p6+train_Table_seq01_p1+ train_Table_seq01_p2+ train_Table_seq01_p3+ train_Table_seq01_p4+ train_Table_seq01_p5+train_Table_seq01_p6+train_Table_seq02_p1+train_Table_seq02_p2+train_Table_seq02_p3+ train_Table_seq02_p4+ train_Table_seq02_p5+  train_Table_seq02_p6+  train_Table_seq03_p1+train_Table_seq03_p2+train_Table_seq03_p3 +train_Table_seq03_p3+ train_Table_seq03_p4+train_Table_seq03_p5+train_Table_seq03_p6+ train_Table_seq04_p1+  train_Table_seq04_p2+  train_Table_seq04_p3+  train_Table_seq04_p4+  train_Table_seq04_p5+ train_Table_seq04_p6+  train_Table_seq05_p1+   train_Table_seq05_p2+  train_Table_seq05_p3+  train_Table_seq05_p4+ train_Table_seq05_p5+ train_Table_seq05_p6+  train_Floor_seq00_p1+  train_Floor_seq00_p2+ train_Floor_seq00_p3+  train_Floor_seq00_p4+  train_Floor_seq00_p5+ train_Floor_seq00_p6+  train_Floor_seq01_p1+  train_Floor_seq01_p2+  train_Floor_seq01_p3+  train_Floor_seq01_p4+  train_Floor_seq01_p5+ train_Floor_seq01_p6+ train_Floor_seq02_p1+ train_Floor_seq02_p2+train_Floor_seq02_p3+train_Floor_seq02_p4+train_Floor_seq02_p5+train_Floor_seq02_p6+  train_wall_seq00_p1+  train_wall_seq00_p2+  train_wall_seq00_p3+  train_wall_seq00_p4+ train_wall_seq00_p5+ train_wall_seq00_p6+  train_wall_seq01_p1+ train_wall_seq01_p2+ train_wall_seq01_p3+ train_wall_seq01_p4+ train_wall_seq01_p5+train_wall_seq01_p6+train_wall_seq02_p1+train_wall_seq02_p2+train_wall_seq02_p3+train_wall_seq02_p4+train_wall_seq02_p5+train_wall_seq02_p6
# # # print("rana", len(train_dataset))

# # train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
# # # breakpoint()
# # loss_func=FocalLoss()
# # # model = model.float()
                                                    

# # epoch_losses= TRAINING_MODULE(model, number_of_epoch, train_loader, FOLDERTOSAVE)


# # with open(FOLDERTOSAVE+'losses.csv', 'w') as csvFile:
# #     writer = csv.writer(csvFile)
# #     writer.writerows([[loss] for loss in epoch_losses])
# #     csvFile.close()

# # plt.title('cross entropy '+discpt)
# # plt.plot(epoch_losses)
# # plt.savefig(FOLDERTOSAVE+discpt+str(number_of_epoch)+'epochs.png',dpi=300, bbox_inches='tight')
# # plt.savefig(FOLDERTOSAVE+discpt+str(number_of_epoch)+'epochs.pdf', format='pdf', dpi=1200)
# # # plt.show()



# # # folder='/home/kucarst3-dlws/YusraMoseg/newtorch/HPC_MoSegModel/TrainingResults/FeedForward_CASE_A_NETWORK_IS_GCN_6LAYERS_LR_0.0001_EPOCH_1/'
# # # model=torch.load(folder+'model.pkl')
# # # model.eval()
# # # print("model loaded is done", model)






# # # # ---------------------------------------------------------------------------------------------------------------------
# # # # STAGE E: TESTING STAGE
# # # # ---------------------------------------------------------------------------------------------------------------------  
# # print("STAGE E: TESTING STAGE")

# # # print("A - TESTING -- TRAINING DATASET ALL")
# # # train_loader1 = DataLoader(train_dataset, batch_size=128, shuffle=False)

# # # descrpt=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTraining.pt'
# # # GT_lbls_, argmax_Y_= TESTING_MODULE(model, train_loader1, descrpt)
# # # csvname=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTraining.csv'
# # # METRICS_MODULE(GT_lbls_,argmax_Y_, csvname )

# # # print("A - TESTING -- TRAINING seq00")

# # # # train_loader_seq00 = DataLoader(train_dataset_seq00, batch_size=1, shuffle=False)
# # # # descrpt_seq00=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTrain_seq00.pt'
# # # # GT_lbls_seq00, argmax_Y_seq00= TESTING_MODULE(model, train_loader_seq00, descrpt_seq00)
# # # # csvname_seq00=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTrain_seq00.csv'
# # # # METRICS_MODULE(GT_lbls_seq00,argmax_Y_seq00, csvname_seq00 )

# # # # # print("A - TESTING -- TRAINING seq01")

# # # # train_loader_seq01 = DataLoader(train_dataset_seq01, batch_size=1, shuffle=False)
# # # # descrpt_seq01=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTrain_seq01.pt'
# # # # GT_lbls_seq01, argmax_Y_seq01= TESTING_MODULE(model, train_loader_seq01, descrpt_seq01)
# # # # csvname_seq01=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTrain_seq01.csv'
# # # # METRICS_MODULE(GT_lbls_seq01,argmax_Y_seq01, csvname_seq01 )


# # # # print("A - TESTING -- TRAINING seq02")
# # # # train_loader_seq02 = DataLoader(train_dataset_seq02, batch_size=1, shuffle=False)
# # # # descrpt_seq02=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTrain_seq02.pt'
# # # # GT_lbls_seq02, argmax_Y_seq02= TESTING_MODULE(model, train_loader_seq02, descrpt_seq02)
# # # # csvname_seq02=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTrain_seq02.csv'
# # # # METRICS_MODULE(GT_lbls_seq02,argmax_Y_seq02, csvname_seq02 )

# # # # print("A - TESTING -- TRAINING seq03")

# # # # train_loader_seq03 = DataLoader(train_dataset_seq03, batch_size=1, shuffle=False)
# # # # descrpt_seq03=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTrain_seq03.pt'
# # # # GT_lbls_seq03, argmax_Y_seq03= TESTING_MODULE(model, train_loader_seq03, descrpt_seq03)
# # # # csvname_seq03=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTrain_seq03.csv'
# # # # METRICS_MODULE(GT_lbls_seq03,argmax_Y_seq03, csvname_seq03 )


# # # # print("A - TESTING -- TRAINING seq04")

# # # # train_loader_seq04 = DataLoader(train_dataset_seq04, batch_size=1, shuffle=False)
# # # # descrpt_seq04=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTrain_seq04.pt'
# # # # GT_lbls_seq04, argmax_Y_seq04= TESTING_MODULE(model, train_loader_seq04, descrpt_seq04)
# # # # csvname_seq04=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTrain_seq04.csv'
# # # # METRICS_MODULE(GT_lbls_seq04,argmax_Y_seq04, csvname_seq04 )

# # # # # print("A - TESTING -- TRAINING seq05")
# # # # # train_loader_seq05 = DataLoader(train_dataset_seq05, batch_size=1, shuffle=False)
# # # # # descrpt_seq05=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTrain_seq05.pt'
# # # # # GT_lbls_seq05, argmax_Y_seq05= TESTING_MODULE(model, train_loader_seq05, descrpt_seq05)
# # # # # csvname_seq05=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTrain_seq05.csv'
# # # # # METRICS_MODULE(GT_lbls_seq05,argmax_Y_seq05, csvname_seq05 )

# # # # # print("A - TESTING -- TRAINING seq06")


# # # # # train_loader_seq06 = DataLoader(train_dataset_seq06, batch_size=1, shuffle=False)
# # # # # descrpt_seq06=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTrain_seq06.pt'
# # # # # GT_lbls_seq06, argmax_Y_seq06= TESTING_MODULE(model, train_loader_seq06, descrpt_seq06)
# # # # # csvname_seq06=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTrain_seq06.csv'
# # # # # METRICS_MODULE(GT_lbls_seq06,argmax_Y_seq06, csvname_seq06 )

# # # # # print("A - TESTING -- TRAINING seq07")

# # # # # train_loader_seq07 = DataLoader(train_dataset_seq07, batch_size=1, shuffle=False)
# # # # # descrpt_seq07=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTrain_seq07.pt'
# # # # # GT_lbls_seq07, argmax_Y_seq07= TESTING_MODULE(model, train_loader_seq07, descrpt_seq07)
# # # # # csvname_seq07=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTrain_seq07.csv'
# # # # # METRICS_MODULE(GT_lbls_seq07,argmax_Y_seq07, csvname_seq07 )

# # # # # print("A - TESTING -- TRAINING seq08")
# # # # # train_loader_seq08 = DataLoader(train_dataset_seq08, batch_size=1, shuffle=False)
# # # # # descrpt_seq08=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTrain_seq08.pt'
# # # # # GT_lbls_seq08, argmax_Y_seq08= TESTING_MODULE(model, train_loader_seq08, descrpt_seq08)
# # # # # csvname_seq08=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTrain_seq08.csv'
# # # # # METRICS_MODULE(GT_lbls_seq08,argmax_Y_seq08, csvname_seq08 )

# # # # # print("A - TESTING -- TRAINING seq09")

# # # # # train_loader_seq09 = DataLoader(train_dataset_seq09, batch_size=1, shuffle=False)
# # # # # descrpt_seq09=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTrain_seq09.pt'
# # # # # GT_lbls_seq09, argmax_Y_seq09= TESTING_MODULE(model, train_loader_seq09, descrpt_seq09)
# # # # # csvname_seq09=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTrain_seq09.csv'
# # # # # METRICS_MODULE(GT_lbls_seq09,argmax_Y_seq09, csvname_seq09 )

# # # # # print("A - TESTING -- TRAINING seq10")

# # # # # train_loader_seq10 = DataLoader(train_dataset_seq10, batch_size=1, shuffle=False)
# # # # # descrpt_seq10=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTrain_seq10.pt'
# # # # # GT_lbls_seq10, argmax_Y_seq10= TESTING_MODULE(model, train_loader_seq10, descrpt_seq10)
# # # # # csvname_seq10=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTrain_seq10.csv'
# # # # # METRICS_MODULE(GT_lbls_seq10,argmax_Y_seq10, csvname_seq10 )


# # # # # print("A - TESTING -- TRAINING seq11")

# # # # # train_loader_seq11 = DataLoader(train_dataset_seq11, batch_size=1, shuffle=False)
# # # # # descrpt_seq11=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTrain_seq11.pt'
# # # # # GT_lbls_seq11, argmax_Y_seq11= TESTING_MODULE(model, train_loader_seq11, descrpt_seq11)
# # # # # csvname_seq11=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTrain_seq11.csv'
# # # # # METRICS_MODULE(GT_lbls_seq11,argmax_Y_seq11, csvname_seq11 )




# # # # print("A - TESTING -- TRAINING DONE!!!!!")

# # # # print("B - TESTING -- TESTING START")
# # # # print("B - TESTING -- TESTING DATASET ALL")
# # # # test_loader= DataLoader(train_dataset, batch_size=1, shuffle=False)

# # # # # test_loader= DataLoader(test_dataset, batch_size=1, shuffle=False)
# # # # descrpt_test=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTesting.pt'
# # # # GT_lbls_test, argmax_Y_test= TESTING_MODULE(model, test_loader, descrpt_test)
# # # # csvname_test=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTesting.csv'
# # # # METRICS_MODULE(GT_lbls_test,argmax_Y_test, csvname_test )


# # # # # print("B - TESTING -- TESTING DATASET seq00")

# # # # # test_loader_seq00= DataLoader(test_dataset_seq00, batch_size=1, shuffle=False)
# # # # # descrpt_seq00=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTesting_seq00.pt'
# # # # # GT_lbls_seq00, argmax_Y_seq00= TESTING_MODULE(model, test_loader_seq00, descrpt_seq00)
# # # # # csvname_test_seq00=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTesting_seq00.csv'
# # # # # METRICS_MODULE(GT_lbls_seq00,argmax_Y_seq00, csvname_test_seq00 )


# # # # # print("B - TESTING -- TESTING DATASET seq01")

# # # # # test_loader_seq01= DataLoader(test_dataset_seq01, batch_size=1, shuffle=False)
# # # # # descrpt_seq01=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTesting_seq01.pt'
# # # # # GT_lbls_seq01, argmax_Y_seq01= TESTING_MODULE(model, test_loader_seq01, descrpt_seq01)
# # # # # csvname_test_seq01=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTesting_seq01.csv'
# # # # # METRICS_MODULE(GT_lbls_seq01,argmax_Y_seq01, csvname_test_seq01 )


# # # # # print("B - TESTING -- TESTING DATASET seq02")


# # # # # test_loader_seq02= DataLoader(test_dataset_seq02, batch_size=1, shuffle=False)
# # # # # descrpt_seq02=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTesting_seq02.pt'
# # # # # GT_lbls_seq02, argmax_Y_seq02= TESTING_MODULE(model, test_loader_seq02, descrpt_seq02)
# # # # # csvname_test_seq02=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTesting_seq02.csv'
# # # # # METRICS_MODULE(GT_lbls_seq02,argmax_Y_seq02, csvname_test_seq02 )

# # # # # print("B - TESTING -- TESTING DATASET seq03")


# # # # # test_loader_seq03= DataLoader(test_dataset_seq03, batch_size=1, shuffle=False)
# # # # # descrpt_seq03=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTesting_seq03.pt'
# # # # # GT_lbls_seq03, argmax_Y_seq03= TESTING_MODULE(model, test_loader_seq03, descrpt_seq03)
# # # # # csvname_test_seq03=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTesting_seq03.csv'
# # # # # METRICS_MODULE(GT_lbls_seq03,argmax_Y_seq03, csvname_test_seq03 )

# # # # # print("B - TESTING -- TESTING DATASET seq04")

# # # # # test_loader_seq04= DataLoader(test_dataset_seq04, batch_size=1, shuffle=False)
# # # # # descrpt_seq04=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTesting_seq04.pt'
# # # # # GT_lbls_seq04, argmax_Y_seq04= TESTING_MODULE(model, test_loader_seq04, descrpt_seq04)
# # # # # csvname_test_seq04=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTesting_seq04.csv'
# # # # # METRICS_MODULE(GT_lbls_seq04,argmax_Y_seq04, csvname_test_seq04 )

# # # # # print("B - TESTING -- TESTING DATASET seq05")

# # # # # test_loader_seq05= DataLoader(test_dataset_seq05, batch_size=1, shuffle=False)
# # # # # descrpt_seq05=FOLDERTOSAVE+'Id_x_y_t_torg_GT_Pred_forTesting_seq05.pt'
# # # # # GT_lbls_seq05, argmax_Y_seq05= TESTING_MODULE(model, test_loader_seq05, descrpt_seq05)
# # # # # csvname_test_seq05=FOLDERTOSAVE+'Confusion_Metric_Scores_detailed_forTesting_seq05.csv'
# # # # # METRICS_MODULE(GT_lbls_seq05,argmax_Y_seq05, csvname_test_seq05 )


# # # # ##---------------------------------------------------------------------------------------------------------------------
# # # # # STAGE F: RESULTs 
# # # # # ---------------------------------------------------------------------------------------------------------------------  
# # # print("STAGE F: RESULTs FINISHED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")





