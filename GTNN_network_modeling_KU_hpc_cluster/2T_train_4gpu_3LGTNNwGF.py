


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

# import script_name_to_call


# from T_read_Aug22 import TrainGraphs_all, TestGraphs_all, traingraphs_wall_allseqs, traingraphs_wall_02,traingraphs_wall_01,traingraphs_wall_00,traingraphs_floor_allseqs, traingraphs_floor_02,traingraphs_floor_01,traingraphs_floor_00,traingraphs_table_allseqs,traingraphs_table_05,traingraphs_table_04,traingraphs_table_03,traingraphs_table_02,traingraphs_table_01,traingraphs_table_00,traingraphs_box_allseqs,traingraphs_box_01,traingraphs_box_02,traingraphs_box_03,traingraphs_box_04,traingraphs_box_05,traingraphs_box_06,traingraphs_box_07,traingraphs_box_08,traingraphs_box_09,traingraphs_box_10,traingraphs_box_11,testgraphs_box,test_Box_seq00_p1,test_Box_seq00_p2,test_Box_seq00_p3,test_Box_seq00_p4,test_Box_seq00_p5,test_Box_seq00_p6,test_Box_seq01_p1,test_Box_seq01_p2,test_Box_seq01_p3,test_Box_seq01_p4,test_Box_seq01_p5,test_Box_seq01_p6,test_Box_seq02_p1,test_Box_seq02_p2,test_Box_seq02_p3,test_Box_seq02_p4,test_Box_seq02_p5,test_Box_seq02_p6,test_Box_seq03_p1,test_Box_seq03_p2,test_Box_seq03_p3,test_Box_seq03_p4,test_Box_seq03_p5,test_Box_seq03_p6,test_Box_seq04_p1,test_Box_seq04_p2,test_Box_seq04_p3,test_Box_seq04_p4,test_Box_seq04_p5,test_Box_seq04_p6,test_Box_seq05_p1,test_Box_seq05_p2,test_Box_seq05_p3,test_Box_seq05_p4,test_Box_seq05_p5,test_Box_seq05_p6,testgraphs_floor,test_Floor_seq00_p1,test_Floor_seq00_p2,test_Floor_seq00_p3,test_Floor_seq00_p4,test_Floor_seq00_p5,test_Floor_seq00_p6,test_Floor_seq01_p1,test_Floor_seq01_p2,test_Floor_seq01_p3,test_Floor_seq01_p4,test_Floor_seq01_p5,test_Floor_seq01_p6,testgraphs_table,test_Table_seq00_p1,test_Table_seq00_p2,test_Table_seq00_p3,test_Table_seq00_p4,test_Table_seq00_p5,test_Table_seq00_p6,test_Table_seq01_p1,test_Table_seq01_p2,test_Table_seq01_p3,test_Table_seq01_p4,test_Table_seq01_p5,test_Table_seq01_p6,test_Table_seq02_p1,test_Table_seq02_p2,test_Table_seq02_p3,test_Table_seq02_p4,test_Table_seq02_p5,test_Table_seq02_p6,test_Table_seq03_p1,test_Table_seq03_p2,test_Table_seq03_p3,test_Table_seq03_p4,test_Table_seq03_p5,test_Table_seq03_p6,testgraphs_fast,test_Fast_seq00_p1,test_Fast_seq00_p2,test_Fast_seq00_p3,test_Fast_seq00_p4,test_Fast_seq00_p5,test_Fast_seq00_p6,test_Fast_seq01_p1,test_Fast_seq01_p2,test_Fast_seq01_p3,test_Fast_seq01_p4,test_Fast_seq01_p5,test_Fast_seq01_p6,test_Fast_seq02_p1,test_Fast_seq02_p2,test_Fast_seq02_p3,test_Fast_seq02_p4,test_Fast_seq02_p5,test_Fast_seq02_p6,testgraphs_wall,test_Wall_seq00_p1,test_Wall_seq00_p2,test_Wall_seq00_p3,test_Wall_seq00_p4,test_Wall_seq00_p5,test_Wall_seq00_p6,test_Wall_seq01_p1,test_Wall_seq01_p2,test_Wall_seq01_p3,test_Wall_seq01_p4,test_Wall_seq01_p5,test_Wall_seq01_p6,TrainGraphs_all,traingraphs_box_allseqs,traingraphs_table_allseqs,traingraphs_floor_allseqs,traingraphs_wall_allseqs,TestGraphs_all,testgraphs_box,testgraphs_floor,testgraphs_table,testgraphs_fast,testgraphs_wall
from ReadPythonFiles.T_read_Aug22_part0 import TrainGraphs_all as TrainGraphs_all_part0, traingraphs_wall_allseqs as traingraphs_wall_allseqs_part0, traingraphs_wall_02 as traingraphs_wall_02_part0,traingraphs_wall_01 as traingraphs_wall_01_part0,traingraphs_wall_00 as traingraphs_wall_00_part0,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part0, traingraphs_floor_02 as traingraphs_floor_02_part0,traingraphs_floor_01 as traingraphs_floor_01_part0,traingraphs_floor_00 as traingraphs_floor_00_part0,traingraphs_table_allseqs as traingraphs_table_allseqs_part0,traingraphs_table_05 as traingraphs_table_05_part0,traingraphs_table_04 as traingraphs_table_04_part0,traingraphs_table_03 as traingraphs_table_03_part0 ,traingraphs_table_02 as traingraphs_table_02_part0, traingraphs_table_01 as traingraphs_table_01_part0,traingraphs_table_00 as traingraphs_table_00_part0,traingraphs_box_allseqs as traingraphs_box_allseqs_part0,traingraphs_box_01 as traingraphs_box_01_part0,traingraphs_box_02 as traingraphs_box_02_part0,traingraphs_box_03 as traingraphs_box_03_part0,traingraphs_box_04 as traingraphs_box_04_part0,traingraphs_box_05 as traingraphs_box_05_part0,traingraphs_box_06 as traingraphs_box_06_part0,traingraphs_box_07 as traingraphs_box_07_part0, traingraphs_box_08 as traingraphs_box_08_part0,traingraphs_box_09 as traingraphs_box_09_part0,traingraphs_box_10 as traingraphs_box_10_part0,traingraphs_box_11 as traingraphs_box_11_part0,traingraphs_box_allseqs as traingraphs_box_allseqs_part0,traingraphs_table_allseqs as traingraphs_table_allseqs_part0,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part0,traingraphs_wall_allseqs as traingraphs_wall_allseqs_part0
from ReadPythonFiles.T_read_Aug22_part1 import TrainGraphs_all as TrainGraphs_all_part1, traingraphs_wall_allseqs as traingraphs_wall_allseqs_part1, traingraphs_wall_02 as traingraphs_wall_02_part1,traingraphs_wall_01 as traingraphs_wall_01_part1,traingraphs_wall_00 as traingraphs_wall_00_part1,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part1, traingraphs_floor_02 as traingraphs_floor_02_part1,traingraphs_floor_01 as traingraphs_floor_01_part1,traingraphs_floor_00 as traingraphs_floor_00_part1,traingraphs_table_allseqs as traingraphs_table_allseqs_part1,traingraphs_table_05 as traingraphs_table_05_part1,traingraphs_table_04 as traingraphs_table_04_part1,traingraphs_table_03 as traingraphs_table_03_part1 ,traingraphs_table_02 as traingraphs_table_02_part1, traingraphs_table_01 as traingraphs_table_01_part1,traingraphs_table_00 as traingraphs_table_00_part1,traingraphs_box_allseqs as traingraphs_box_allseqs_part1,traingraphs_box_01 as traingraphs_box_01_part1,traingraphs_box_02 as traingraphs_box_02_part1,traingraphs_box_03 as traingraphs_box_03_part1,traingraphs_box_04 as traingraphs_box_04_part1,traingraphs_box_05 as traingraphs_box_05_part1,traingraphs_box_06 as traingraphs_box_06_part1,traingraphs_box_07 as traingraphs_box_07_part1, traingraphs_box_08 as traingraphs_box_08_part1,traingraphs_box_09 as traingraphs_box_09_part1,traingraphs_box_10 as traingraphs_box_10_part1,traingraphs_box_11 as traingraphs_box_11_part1,traingraphs_box_allseqs as traingraphs_box_allseqs_part1,traingraphs_table_allseqs as traingraphs_table_allseqs_part1,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part1,traingraphs_wall_allseqs as traingraphs_wall_allseqs_part1
from ReadPythonFiles.T_read_Aug22_part2 import TrainGraphs_all as TrainGraphs_all_part2, traingraphs_wall_allseqs as traingraphs_wall_allseqs_part2, traingraphs_wall_02 as traingraphs_wall_02_part2,traingraphs_wall_01 as traingraphs_wall_01_part2,traingraphs_wall_00 as traingraphs_wall_00_part2,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part2, traingraphs_floor_02 as traingraphs_floor_02_part2,traingraphs_floor_01 as traingraphs_floor_01_part2,traingraphs_floor_00 as traingraphs_floor_00_part2,traingraphs_table_allseqs as traingraphs_table_allseqs_part2,traingraphs_table_05 as traingraphs_table_05_part2,traingraphs_table_04 as traingraphs_table_04_part2,traingraphs_table_03 as traingraphs_table_03_part2 ,traingraphs_table_02 as traingraphs_table_02_part2, traingraphs_table_01 as traingraphs_table_01_part2,traingraphs_table_00 as traingraphs_table_00_part2,traingraphs_box_allseqs as traingraphs_box_allseqs_part2,traingraphs_box_01 as traingraphs_box_01_part2,traingraphs_box_02 as traingraphs_box_02_part2,traingraphs_box_03 as traingraphs_box_03_part2,traingraphs_box_04 as traingraphs_box_04_part2,traingraphs_box_05 as traingraphs_box_05_part2,traingraphs_box_06 as traingraphs_box_06_part2,traingraphs_box_07 as traingraphs_box_07_part2, traingraphs_box_08 as traingraphs_box_08_part2,traingraphs_box_09 as traingraphs_box_09_part2,traingraphs_box_10 as traingraphs_box_10_part2,traingraphs_box_11 as traingraphs_box_11_part2,traingraphs_box_allseqs as traingraphs_box_allseqs_part2,traingraphs_table_allseqs as traingraphs_table_allseqs_part2,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part2,traingraphs_wall_allseqs as traingraphs_wall_allseqs_part2
from ReadPythonFiles.T_read_Aug22_part3 import TrainGraphs_all as TrainGraphs_all_part3, traingraphs_wall_allseqs as traingraphs_wall_allseqs_part3, traingraphs_wall_02 as traingraphs_wall_02_part3,traingraphs_wall_01 as traingraphs_wall_01_part3,traingraphs_wall_00 as traingraphs_wall_00_part3,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part3, traingraphs_floor_02 as traingraphs_floor_02_part3,traingraphs_floor_01 as traingraphs_floor_01_part3,traingraphs_floor_00 as traingraphs_floor_00_part3,traingraphs_table_allseqs as traingraphs_table_allseqs_part3,traingraphs_table_05 as traingraphs_table_05_part3,traingraphs_table_04 as traingraphs_table_04_part3,traingraphs_table_03 as traingraphs_table_03_part3 ,traingraphs_table_02 as traingraphs_table_02_part3, traingraphs_table_01 as traingraphs_table_01_part3,traingraphs_table_00 as traingraphs_table_00_part3,traingraphs_box_allseqs as traingraphs_box_allseqs_part3,traingraphs_box_01 as traingraphs_box_01_part3,traingraphs_box_02 as traingraphs_box_02_part3,traingraphs_box_03 as traingraphs_box_03_part3,traingraphs_box_04 as traingraphs_box_04_part3,traingraphs_box_05 as traingraphs_box_05_part3,traingraphs_box_06 as traingraphs_box_06_part3,traingraphs_box_07 as traingraphs_box_07_part3, traingraphs_box_08 as traingraphs_box_08_part3,traingraphs_box_09 as traingraphs_box_09_part3,traingraphs_box_10 as traingraphs_box_10_part3,traingraphs_box_11 as traingraphs_box_11_part3,traingraphs_box_allseqs as traingraphs_box_allseqs_part3,traingraphs_table_allseqs as traingraphs_table_allseqs_part3,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part3,traingraphs_wall_allseqs as traingraphs_wall_allseqs_part3
from ReadPythonFiles.T_read_Aug22_part4 import TrainGraphs_all as TrainGraphs_all_part4, traingraphs_wall_allseqs as traingraphs_wall_allseqs_part4, traingraphs_wall_02 as traingraphs_wall_02_part4,traingraphs_wall_01 as traingraphs_wall_01_part4,traingraphs_wall_00 as traingraphs_wall_00_part4,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part4, traingraphs_floor_02 as traingraphs_floor_02_part4,traingraphs_floor_01 as traingraphs_floor_01_part4,traingraphs_floor_00 as traingraphs_floor_00_part4,traingraphs_table_allseqs as traingraphs_table_allseqs_part4,traingraphs_table_05 as traingraphs_table_05_part4,traingraphs_table_04 as traingraphs_table_04_part4,traingraphs_table_03 as traingraphs_table_03_part4 ,traingraphs_table_02 as traingraphs_table_02_part4, traingraphs_table_01 as traingraphs_table_01_part4,traingraphs_table_00 as traingraphs_table_00_part4,traingraphs_box_allseqs as traingraphs_box_allseqs_part4,traingraphs_box_01 as traingraphs_box_01_part4,traingraphs_box_02 as traingraphs_box_02_part4,traingraphs_box_03 as traingraphs_box_03_part4,traingraphs_box_04 as traingraphs_box_04_part4,traingraphs_box_05 as traingraphs_box_05_part4,traingraphs_box_06 as traingraphs_box_06_part4,traingraphs_box_07 as traingraphs_box_07_part4, traingraphs_box_08 as traingraphs_box_08_part4,traingraphs_box_09 as traingraphs_box_09_part4,traingraphs_box_10 as traingraphs_box_10_part4,traingraphs_box_11 as traingraphs_box_11_part4,traingraphs_box_allseqs as traingraphs_box_allseqs_part4,traingraphs_table_allseqs as traingraphs_table_allseqs_part4,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part4,traingraphs_wall_allseqs as traingraphs_wall_allseqs_part4
from ReadPythonFiles.T_read_Aug22_part5 import TrainGraphs_all as TrainGraphs_all_part5, traingraphs_wall_allseqs as traingraphs_wall_allseqs_part5, traingraphs_wall_02 as traingraphs_wall_02_part5,traingraphs_wall_01 as traingraphs_wall_01_part5,traingraphs_wall_00 as traingraphs_wall_00_part5,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part5, traingraphs_floor_02 as traingraphs_floor_02_part5,traingraphs_floor_01 as traingraphs_floor_01_part5,traingraphs_floor_00 as traingraphs_floor_00_part5,traingraphs_table_allseqs as traingraphs_table_allseqs_part5,traingraphs_table_05 as traingraphs_table_05_part5,traingraphs_table_04 as traingraphs_table_04_part5,traingraphs_table_03 as traingraphs_table_03_part5 ,traingraphs_table_02 as traingraphs_table_02_part5, traingraphs_table_01 as traingraphs_table_01_part5,traingraphs_table_00 as traingraphs_table_00_part5,traingraphs_box_allseqs as traingraphs_box_allseqs_part5,traingraphs_box_01 as traingraphs_box_01_part5,traingraphs_box_02 as traingraphs_box_02_part5,traingraphs_box_03 as traingraphs_box_03_part5,traingraphs_box_04 as traingraphs_box_04_part5,traingraphs_box_05 as traingraphs_box_05_part5,traingraphs_box_06 as traingraphs_box_06_part5,traingraphs_box_07 as traingraphs_box_07_part5, traingraphs_box_08 as traingraphs_box_08_part5,traingraphs_box_09 as traingraphs_box_09_part5,traingraphs_box_10 as traingraphs_box_10_part5,traingraphs_box_11 as traingraphs_box_11_part5,traingraphs_box_allseqs as traingraphs_box_allseqs_part5,traingraphs_table_allseqs as traingraphs_table_allseqs_part5,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part5,traingraphs_wall_allseqs as traingraphs_wall_allseqs_part5
from ReadPythonFiles.T_read_Aug22_part6 import TrainGraphs_all as TrainGraphs_all_part6, traingraphs_wall_allseqs as traingraphs_wall_allseqs_part6, traingraphs_wall_02 as traingraphs_wall_02_part6,traingraphs_wall_01 as traingraphs_wall_01_part6,traingraphs_wall_00 as traingraphs_wall_00_part6,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part6, traingraphs_floor_02 as traingraphs_floor_02_part6,traingraphs_floor_01 as traingraphs_floor_01_part6,traingraphs_floor_00 as traingraphs_floor_00_part6,traingraphs_table_allseqs as traingraphs_table_allseqs_part6,traingraphs_table_05 as traingraphs_table_05_part6,traingraphs_table_04 as traingraphs_table_04_part6,traingraphs_table_03 as traingraphs_table_03_part6 ,traingraphs_table_02 as traingraphs_table_02_part6, traingraphs_table_01 as traingraphs_table_01_part6,traingraphs_table_00 as traingraphs_table_00_part6,traingraphs_box_allseqs as traingraphs_box_allseqs_part6,traingraphs_box_01 as traingraphs_box_01_part6,traingraphs_box_02 as traingraphs_box_02_part6,traingraphs_box_03 as traingraphs_box_03_part6,traingraphs_box_04 as traingraphs_box_04_part6,traingraphs_box_05 as traingraphs_box_05_part6,traingraphs_box_06 as traingraphs_box_06_part6,traingraphs_box_07 as traingraphs_box_07_part6, traingraphs_box_08 as traingraphs_box_08_part6,traingraphs_box_09 as traingraphs_box_09_part6,traingraphs_box_10 as traingraphs_box_10_part6,traingraphs_box_11 as traingraphs_box_11_part6,traingraphs_box_allseqs as traingraphs_box_allseqs_part6,traingraphs_table_allseqs as traingraphs_table_allseqs_part6,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part6,traingraphs_wall_allseqs as traingraphs_wall_allseqs_part6
from ReadPythonFiles.T_read_Aug22_part7 import TrainGraphs_all as TrainGraphs_all_part7, traingraphs_wall_allseqs as traingraphs_wall_allseqs_part7, traingraphs_wall_02 as traingraphs_wall_02_part7,traingraphs_wall_01 as traingraphs_wall_01_part7,traingraphs_wall_00 as traingraphs_wall_00_part7,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part7, traingraphs_floor_02 as traingraphs_floor_02_part7,traingraphs_floor_01 as traingraphs_floor_01_part7,traingraphs_floor_00 as traingraphs_floor_00_part7,traingraphs_table_allseqs as traingraphs_table_allseqs_part7,traingraphs_table_05 as traingraphs_table_05_part7,traingraphs_table_04 as traingraphs_table_04_part7,traingraphs_table_03 as traingraphs_table_03_part7 ,traingraphs_table_02 as traingraphs_table_02_part7, traingraphs_table_01 as traingraphs_table_01_part7,traingraphs_table_00 as traingraphs_table_00_part7,traingraphs_box_allseqs as traingraphs_box_allseqs_part7,traingraphs_box_01 as traingraphs_box_01_part7,traingraphs_box_02 as traingraphs_box_02_part7,traingraphs_box_03 as traingraphs_box_03_part7,traingraphs_box_04 as traingraphs_box_04_part7,traingraphs_box_05 as traingraphs_box_05_part7,traingraphs_box_06 as traingraphs_box_06_part7,traingraphs_box_07 as traingraphs_box_07_part7, traingraphs_box_08 as traingraphs_box_08_part7,traingraphs_box_09 as traingraphs_box_09_part7,traingraphs_box_10 as traingraphs_box_10_part7,traingraphs_box_11 as traingraphs_box_11_part7,traingraphs_box_allseqs as traingraphs_box_allseqs_part7,traingraphs_table_allseqs as traingraphs_table_allseqs_part7,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part7,traingraphs_wall_allseqs as traingraphs_wall_allseqs_part7
from ReadPythonFiles.T_read_Aug22_part8 import TrainGraphs_all as TrainGraphs_all_part8, traingraphs_wall_allseqs as traingraphs_wall_allseqs_part8, traingraphs_wall_02 as traingraphs_wall_02_part8,traingraphs_wall_01 as traingraphs_wall_01_part8,traingraphs_wall_00 as traingraphs_wall_00_part8,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part8, traingraphs_floor_02 as traingraphs_floor_02_part8,traingraphs_floor_01 as traingraphs_floor_01_part8,traingraphs_floor_00 as traingraphs_floor_00_part8,traingraphs_table_allseqs as traingraphs_table_allseqs_part8,traingraphs_table_05 as traingraphs_table_05_part8,traingraphs_table_04 as traingraphs_table_04_part8,traingraphs_table_03 as traingraphs_table_03_part8 ,traingraphs_table_02 as traingraphs_table_02_part8, traingraphs_table_01 as traingraphs_table_01_part8,traingraphs_table_00 as traingraphs_table_00_part8,traingraphs_box_allseqs as traingraphs_box_allseqs_part8,traingraphs_box_01 as traingraphs_box_01_part8,traingraphs_box_02 as traingraphs_box_02_part8,traingraphs_box_03 as traingraphs_box_03_part8,traingraphs_box_04 as traingraphs_box_04_part8,traingraphs_box_05 as traingraphs_box_05_part8,traingraphs_box_06 as traingraphs_box_06_part8,traingraphs_box_07 as traingraphs_box_07_part8, traingraphs_box_08 as traingraphs_box_08_part8,traingraphs_box_09 as traingraphs_box_09_part8,traingraphs_box_10 as traingraphs_box_10_part8,traingraphs_box_11 as traingraphs_box_11_part8,traingraphs_box_allseqs as traingraphs_box_allseqs_part8,traingraphs_table_allseqs as traingraphs_table_allseqs_part8,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part8,traingraphs_wall_allseqs as traingraphs_wall_allseqs_part8
from ReadPythonFiles.T_read_Aug22_part9 import TrainGraphs_all as TrainGraphs_all_part9, traingraphs_wall_allseqs as traingraphs_wall_allseqs_part9, traingraphs_wall_02 as traingraphs_wall_02_part9,traingraphs_wall_01 as traingraphs_wall_01_part9,traingraphs_wall_00 as traingraphs_wall_00_part9,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part9, traingraphs_floor_02 as traingraphs_floor_02_part9,traingraphs_floor_01 as traingraphs_floor_01_part9,traingraphs_floor_00 as traingraphs_floor_00_part9,traingraphs_table_allseqs as traingraphs_table_allseqs_part9,traingraphs_table_05 as traingraphs_table_05_part9,traingraphs_table_04 as traingraphs_table_04_part9,traingraphs_table_03 as traingraphs_table_03_part9 ,traingraphs_table_02 as traingraphs_table_02_part9, traingraphs_table_01 as traingraphs_table_01_part9,traingraphs_table_00 as traingraphs_table_00_part9,traingraphs_box_allseqs as traingraphs_box_allseqs_part9,traingraphs_box_01 as traingraphs_box_01_part9,traingraphs_box_02 as traingraphs_box_02_part9,traingraphs_box_03 as traingraphs_box_03_part9,traingraphs_box_04 as traingraphs_box_04_part9,traingraphs_box_05 as traingraphs_box_05_part9,traingraphs_box_06 as traingraphs_box_06_part9,traingraphs_box_07 as traingraphs_box_07_part9, traingraphs_box_08 as traingraphs_box_08_part9,traingraphs_box_09 as traingraphs_box_09_part9,traingraphs_box_10 as traingraphs_box_10_part9,traingraphs_box_11 as traingraphs_box_11_part9,traingraphs_box_allseqs as traingraphs_box_allseqs_part9,traingraphs_table_allseqs as traingraphs_table_allseqs_part9,traingraphs_floor_allseqs as traingraphs_floor_allseqs_part9,traingraphs_wall_allseqs as traingraphs_wall_allseqs_part9
from ReadPythonFiles.T_read_Aug22_part10 import TrainGraphs_all as TrainGraphs_all_part10, traingraphs_room1_obj1_allseqs, traingraphs_room1_obj2_allseqs,traingraphs_room1_obj3_allseqs,traingraphs_room3_obj1_allseqs,traingraphs_room3_obj2_allseqs,traingraphs_room3_obj3_allseqs


from ReadPythonFiles.T_read_Aug22_part11_EV_TypeA import TrainGraphs_all as TrainGraphs_all_part11A, evimo2_seq6_9,evimo2_seq10,evimo2_seq12_13,evimo2_seq14_15




# from utils import Logger


torch.cuda.empty_cache()
torch.backends.cudnn.benchmark=True

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Functions and classess
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
    
# def TRAINING_MODULE(model, number_of_epoch, train_loader, FOLDERTOSAVE): 

def TRAINING_MODULE(model, number_of_epoch, train_loader_part0, train_loader_part1, train_loader_part2, train_loader_part3, train_loader_part4, train_loader_part5, train_loader_part6, train_loader_part7, train_loader_part8, train_loader_part9,train_loader_part10, train_loader_part11, FOLDERTOSAVE):
    model.train()
    #Effective Training Scheme
    nk=10
    lst0= list(np.arange(0, number_of_epoch+1, nk))
    print(lst0)
    lst1= list(np.arange(1, number_of_epoch+1, nk))
    print(lst1)
    lst2= list(np.arange(2, number_of_epoch+1, nk))
    print(lst2)
    lst3= list(np.arange(3, number_of_epoch+1, nk))
    print(lst3)
    lst4= list(np.arange(4, number_of_epoch+1, nk))
    print(lst4)
    lst5= list(np.arange(5, number_of_epoch+1, nk))
    print(lst5)
    lst6= list(np.arange(6, number_of_epoch+1, nk))
    print(lst6)
    lst7= list(np.arange(7, number_of_epoch+1, nk))
    print(lst7)
    lst8= list(np.arange(8, number_of_epoch+1, nk))
    print(lst8)
    lst9= list(np.arange(9, number_of_epoch+1, nk))
    print(lst9)
    lst10= list(np.arange(10, number_of_epoch+1, nk))
    print(lst10)
    i=0
    acc=[]
    epoch_losses = []
    print ("Training will start now")
    for epoch in range(number_of_epoch):
        print("Epoch", epoch)
        epoch_loss = 0
        acc=0
        start=time.time()

        if epoch in lst0:
            train_loader=train_loader_part0
            print("Epoch", epoch ,"train_loader_part0", len(train_loader_part0))
        if epoch in lst1:
            train_loader=train_loader_part2
            print("Epoch", epoch , "train_loader_part1", len(train_loader_part1))

        if epoch in lst2:
            train_loader=train_loader_part4
            print("Epoch", epoch , "train_loader_part2", len(train_loader_part2))

        if epoch in lst3:
            train_loader=train_loader_part6
            print("Epoch", epoch , "train_loader_part3", len(train_loader_part3))

        if epoch in lst4:
            train_loader=train_loader_part8
            print("Epoch", epoch , "train_loader_part4", len(train_loader_part4))


        if epoch in lst5:
            train_loader=train_loader_part0
            print("Epoch", epoch , "train_loader_part5", len(train_loader_part5))

        if epoch in lst6:
            train_loader=train_loader_part2
            print("Epoch", epoch , "train_loader_part6", len(train_loader_part6))

        if epoch in lst7:
            train_loader=train_loader_part4
            print("Epoch", epoch , "train_loader_part7", len(train_loader_part7))

        if epoch in lst8:
            train_loader=train_loader_part6
            print("Epoch", epoch , "train_loader_part8", len(train_loader_part8))

        if epoch in lst9:
            train_loader=train_loader_part8
            print("Epoch", epoch , "train_loader_part9", len(train_loader_part9))

        if epoch in lst10:
            train_loader=train_loader_part11
            print("Epoch", epoch , "train_loader_part10", len(train_loader_part11))



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
            acc += (pred.eq(data.y).sum().item())/len(data.y)

            loss.backward()
            optimizer.step() 
            epoch_loss += loss.detach().item()
            i=i+1
        acc /=(i+1)
        epoch_loss /= (i + 1)
        end = time.time()
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), ' Elapsed time: ', end-start, 'Acc', acc)
        epoch_losses.append(epoch_loss)
        torch.save(model.state_dict(), FOLDERTOSAVE+'model_weights.pth')
        torch.save(model, FOLDERTOSAVE+'model.pkl')


        with open(FOLDERTOSAVE+'losses.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows([[loss] for loss in epoch_losses])
            csvFile.close()

        plt.title('cross entropy '+discpt)
        plt.plot(epoch_losses)
        plt.savefig(FOLDERTOSAVE+discpt+str(number_of_epoch)+'epochs.png',dpi=300, bbox_inches='tight')
        plt.savefig(FOLDERTOSAVE+discpt+str(number_of_epoch)+'epochs.pdf', format='pdf', dpi=1200)
# plt.show()
    #     toc()
    return epoch_losses
def TESTING_MODULE(model, test_loader, descrpt):
    model.eval()

    correct = 0
    mov_correct=0

    GT=[]
    prediction=[]
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
        seq_data.append(seq_data1)
    return seq_data

def Process_creat_Graphs(seq_data, alpha):

    ind=  (seq_data[:, 3]>=100) & (seq_data[:, 3]<=250) & (seq_data[:, 4]>=50) & (seq_data[:, 4]<=150)

    xdata=seq_data [:, 3]*125/(346*125)
    id_data=(seq_data[:, 0])
    t_org=seq_data [:, 2]
    ydata=seq_data [:, 4]*125/(125*260)
    # if len(seq_data[0])>0:
    tdata=alpha*(seq_data [:, 2]-seq_data [0, 2])*(1/0.005)


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
        for raw_path in self.raw_paths:

            
            sample=raw_path

            pp1=(np.fromfile(sample,  dtype=np.float))
            pp=numpy.reshape(pp1, (Numberofdata, int(len(pp1)/Numberofdata)), order='C')
            pp=numpy.transpose(pp)

            sq1=pp
            sq = numpy.reshape(sq1, (1, int(pp.shape[0]), (pp.shape[1])))
            seq.append(numpy.concatenate(sq))


            deltatime=0.005
            content = Slice_Raw_Events_onTemporal(deltatime, seq)
            print("content", len(content))

            for sample1 in range(0, len(content)-1):
                print(sample1, len(content[sample1]))
                # data=[]
                if len(content[sample1]>0):
                    data=Process_creat_Graphs(content[sample1], 1)
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
number_of_epoch=1000
batch_size=96
# CREATE FOLDERS TO SAVE RESULTS 
discpt=''
FOLDERTOSAVE = 'TrainingResults_wMOD_EVIMO2_3L64wG/ModelC_3L64wG_'+discpt+'_LR_'+str(learningRate)+"_EPOCH_"+str(number_of_epoch)+"batch"+str(batch_size)+'/'
if not os.path.isdir(FOLDERTOSAVE):
    os.makedirs(FOLDERTOSAVE)


##---------------------------------------------------------------------------------------------------------------------
# STAGE B: Network classes
# ---------------------------------------------------------------------------------------------------------------------  
from torch_geometric.nn.conv import PointTransformerConv
import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Identity
from torch.nn import Linear as Lin
from torch.nn import ReLU
from torch.nn import Sequential as Seq
from torch_cluster import fps, knn_graph
from torch_scatter import scatter_max

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import PointTransformerConv
from torch_geometric.nn.pool import knn
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn import global_max_pool



class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], batch_norm=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], batch_norm=False)

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x

class TransitionDown(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels])

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out, _ = scatter_max(x[id_k_neighbor[1]], id_k_neighbor[0],
                               dim_size=id_clusters.size(0), dim=0)

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch




class TransitionDown1(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''
    def __init__(self, in_channels, out_channels, k=16):
        super().__init__()
        print("in_channels", in_channels)
        ratio=0.01
        print("ratio", ratio)

        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels])

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)
        # print("allaa!!!!!!!!!!!!!!!id_clusters", len(id_clusters))

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)
        # print("allaa!!!!!!!!!!!!!!!id_k_neighbor", len(id_k_neighbor))

        # transformation of features through a simple MLP
        x = self.mlp(x)
        # print("allaa!!!!!!!!!!!!!!!x", len(x))

        # Max pool onto each cluster the features from knn in points
        x_out, _ = scatter_max(x[id_k_neighbor[1]], id_k_neighbor[0],
                               dim_size=id_clusters.size(0), dim=0)

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out

        # print("allaa!!!!!!!!!!!!!!!1", (out))
        return out, sub_pos, sub_batch

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]),
            BN(channels[i]) if batch_norm else Identity(), ReLU())
        for i in range(1, len(channels))
    ])
class TransitionUp(torch.nn.Module):
    '''
        Reduce features dimensionnality and interpolate back to higher
        resolution and cardinality
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp_sub = MLP([in_channels, out_channels])
        self.mlp = MLP([out_channels, out_channels])

    def forward(self, x, x_sub, pos, pos_sub, batch=None, batch_sub=None):
        # transform low-res features and reduce the number of features
        x_sub = self.mlp_sub(x_sub)

        # interpolate low-res feats to high-res points
        x_interpolated = knn_interpolate(x_sub, pos_sub, pos, k=3,
                                         batch_x=batch_sub, batch_y=batch)

        x = self.mlp(x) + x_interpolated

        return x
class Net_PointTransformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]])

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
        )

        # backbone layers
        self.transformers_up = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        self.transition_up = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()
        



        for i in range(0, len(dim_model) - 1):

            # Add Transition Down block followed by a Point Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.k))

            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1],
                                 out_channels=dim_model[i + 1]))
        

            # Add Transition Up block followed by Point Transformer block
            self.transition_up.append(
                TransitionUp(in_channels=dim_model[i + 1],
                             out_channels=dim_model[i]))

            self.transformers_up.append(
                TransformerBlock(in_channels=dim_model[i],
                                 out_channels=dim_model[i]))



        self.transition_down_YA = TransitionDown1(in_channels=dim_model[i+1], out_channels=512, k=self.k )
        self.transformers_down_YA = TransformerBlock(in_channels=64, out_channels=512*2)

        # summit layers
        self.mlp_summit = MLP([dim_model[-1], dim_model[-1]], batch_norm=False)
        self.mlp_summitYA = MLP([(512), 32], batch_norm=False)

        self.transformer_summit = TransformerBlock(
            in_channels=dim_model[-1],
            out_channels=dim_model[-1],
        )


        self.mlp_output = Seq(Lin(dim_model[0]+32, 64), ReLU(), Lin(64, 64),
                              ReLU(), Lin(64, out_channels))

    def forward(self, x, pos, batch=None):

        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1)).to(pos.get_device())

        out_x = []
        out_pos = []
        out_batch = []
        



        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)

        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)


        ##YA modified start Global feature extraction
        YA_x=x
        YA_pos=pos
        YA_batch=batch

        YA_x1, YA_pos1, YA_batch1 = self.transition_down_YA(YA_x, YA_pos, batch=YA_batch)

        YA_h = global_max_pool(YA_x1,YA_batch1 )  # [num_examples, hidden_channels]

        YA_h2= self.mlp_summitYA(YA_h[0])#.view(-1))

        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_summit(x, pos, edge_index)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.transformers_down)
        for i in range(n):
            x = self.transition_up[-i - 1](x=out_x[-i - 2], x_sub=x,
                                           pos=out_pos[-i - 2],
                                           pos_sub=out_pos[-i - 1],
                                           batch_sub=out_batch[-i - 1],
                                           batch=out_batch[-i - 2])

            edge_index = knn_graph(out_pos[-i - 2], k=self.k,
                                   batch=out_batch[-i - 2])
            x = self.transformers_up[-i - 1](x, out_pos[-i - 2], edge_index)


        YA_h2=YA_h2.repeat(len(x),1)

        x2=torch.cat((x, YA_h2), dim=1)

        out = self.mlp_output(x2)

        return F.softmax(out, dim=1)



transform = T.Cartesian(cat=False)

class Net_TRANSFORMER(torch.nn.Module):
    def __init__(self):
        super(Net_TRANSFORMER, self).__init__()


        feature_size=3
        self.encoder_embedding_size=64
        self.encoder_embedding_size1=32
        self.encoder_embedding_size2=16

        

        self.edge_dim=3
        self.conv1 = TransformerConv(feature_size, 
                                    self.encoder_embedding_size, 
                                    heads=4, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=self.edge_dim)       

        self.conv2 = TransformerConv(self.encoder_embedding_size, 
                                    self.encoder_embedding_size1, 
                                    heads=4, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=self.edge_dim)  

        self.conv3 = TransformerConv(self.encoder_embedding_size1, 
                                    self.encoder_embedding_size2, 
                                    heads=4, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=self.edge_dim)   
    
        
        self.conv6 = TransformerConv(self.encoder_embedding_size2, 
                                    2, 
                                    heads=4, 
                                    concat=False,
                                    beta=True,
                                    edge_dim=self.edge_dim)    


        self.bn1 = torch.nn.BatchNorm1d(self.encoder_embedding_size)
        self.bn2 = torch.nn.BatchNorm1d(self.encoder_embedding_size1)
        self.bn3 = torch.nn.BatchNorm1d(self.encoder_embedding_size2)


    def forward(self, data):
 
        data.x=self.conv1(data.x.double(), data.edge_index, data.edge_attr.double())
        data.x = torch.sigmoid(data.x)
        data.x = self.bn1(data.x) 

        data.x=self.conv2(data.x.double(), data.edge_index, data.edge_attr.double())
        data.x = torch.sigmoid(data.x)
        data.x = self.bn2(data.x) 

        data.x=self.conv3(data.x.double(), data.edge_index, data.edge_attr.double())
        data.x = torch.sigmoid(data.x)
        data.x = self.bn3(data.x) 

        x = torch.sigmoid(self.conv6(data.x.double(), data.edge_index, data.edge_attr.double()))
        OUT=F.softmax(x)
        # print("x",F.softmax(x))
        return OUT

class NetConnect_1(torch.nn.Module):
    def __init__(self):
        super(NetConnect_1, self).__init__()

        self.conv0 = GCNConv(3, 16)
        self.bn0 = torch.nn.BatchNorm1d(16)
        
        self.conv1 = GCNConv(16, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = GCNConv(64, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        
        
        self.conv3 = GCNConv(128, 256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        #self.bn1 = BatchNorm(64)

        self.conv4 = GCNConv(256, 128)
        self.bn4 = torch.nn.BatchNorm1d(128)
        
        self.conv5 = GCNConv(128*2, 64)
        self.bn5 = torch.nn.BatchNorm1d(64)
   

        self.conv6 = GCNConv(64*2, 16)
        self.bn6 = torch.nn.BatchNorm1d(16)
        
        self.conv7 = GCNConv(16*2, 2)
        self.bn7 = torch.nn.BatchNorm1d(2)


#        self.bn4 = torch.nn.BatchNorm1d(512)
#         self.conv8 = GCNConv(16, 2)
        #self.bn4 = torch.nn.BatchNorm1d(2)  
#         self.fc1 = torch.nn.Linear(32*512, 1024)
#         self.fc2 = torch.nn.Linear(1024, 2)


    def forward(self, data):
        # data.x = F.leaky_relu(self.conv0(data.x, data.edge_index))
        data.x = torch.sigmoid(self.conv0(data.x, data.edge_index))




        data.x = self.bn0(data.x)      
        part0=data.x
        data.x = torch.sigmoid(self.conv1( data.x , data.edge_index))
        data.x = self.bn1(data.x)
        part1=data.x
        data.x = torch.sigmoid(self.conv2(data.x, data.edge_index))

        data.x = self.bn2(data.x)
        part2=data.x
        data.x = torch.sigmoid(self.conv3(data.x, data.edge_index))

        data.x = self.bn3(data.x)  
        
        data.x = torch.sigmoid(self.conv4(data.x, data.edge_index))

        data.x = self.bn4(data.x) #concat with part2
        data.x=torch.cat((data.x, part2), dim=1)
        data.x = torch.sigmoid(self.conv5(data.x, data.edge_index))

        data.x = self.bn5(data.x) #concat with part1
        data.x=torch.cat((data.x, part1), dim=1)

        data.x = torch.sigmoid(self.conv6(data.x, data.edge_index))
        data.x = self.bn6(data.x) #concat with part0
        data.x=torch.cat((data.x, part0), dim=1)

        data.x = torch.sigmoid(self.conv7(data.x, data.edge_index))
        data.x = self.bn7(data.x)
        out=F.softmax(data.x, dim=1)
        #print("helooooo",(out))
        return out#data.x

class NetConnect_2(torch.nn.Module):
    def __init__(self):
        super(NetConnect_2, self).__init__()
        self.conv0 = GCNConv(3, 16)
        self.bn0 = torch.nn.BatchNorm1d(16)
        
        self.conv1 = GCNConv(16, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)


        self.conv2 = GCNConv(64, 128)
        self.bn2 = torch.nn.BatchNorm1d(128)
        
        self.conv3 = GCNConv(128, 256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        #self.bn1 = BatchNorm(64)

        self.conv4 = GCNConv(256, 512)
        self.bn4 = torch.nn.BatchNorm1d(512)


        self.conv5 = GCNConv(512, 256)
        self.bn5 = torch.nn.BatchNorm1d(256)


        self.conv6 = GCNConv(256*2, 128)
        self.bn6 = torch.nn.BatchNorm1d(128)
   

        self.conv7 = GCNConv(128*2, 64)
        self.bn7 = torch.nn.BatchNorm1d(64)
        
        self.conv8 = GCNConv(64*2, 16)
        self.bn8 = torch.nn.BatchNorm1d(16)


        self.conv9 = GCNConv(16*2, 2)
        self.bn9 = torch.nn.BatchNorm1d(2)



    def forward(self, data):
        data.x = torch.sigmoid(self.conv0(data.x, data.edge_index))

        data.x = self.bn0(data.x)      
        part0=data.x
        data.x = torch.sigmoid(self.conv1( data.x , data.edge_index))

        data.x = self.bn1(data.x)
        part1=data.x
        data.x = torch.sigmoid(self.conv2(data.x, data.edge_index))

        data.x = self.bn2(data.x)
        part2=data.x
        data.x = torch.sigmoid(self.conv3(data.x, data.edge_index))
        data.x = self.bn3(data.x)  
        part3=data.x


        data.x = torch.sigmoid(self.conv4(data.x, data.edge_index))

        data.x = self.bn4(data.x) #concat with part2


        data.x = torch.sigmoid(self.conv5(data.x, data.edge_index))
        data.x = self.bn5(data.x) #concat with part2


        data.x=torch.cat((data.x, part3), dim=1)
        data.x = torch.sigmoid(self.conv6(data.x, data.edge_index))

        data.x = self.bn6(data.x) #concat with part1


        data.x=torch.cat((data.x, part2), dim=1)

        data.x = torch.sigmoid(self.conv7(data.x, data.edge_index))
        data.x = self.bn7(data.x) #concat with part0
        data.x=torch.cat((data.x, part1), dim=1)


        data.x = torch.sigmoid(self.conv8(data.x, data.edge_index))
        data.x = self.bn8(data.x) #concat with part0
        data.x=torch.cat((data.x, part0), dim=1)

        data.x = torch.sigmoid(self.conv9(data.x, data.edge_index))
        data.x = self.bn9(data.x)

        out=F.softmax(data.x, dim=1)
        #print("helooooo",(out))
        return out#data.x


class NetConnect_3(torch.nn.Module):
    def __init__(self):
        super(NetConnect_3, self).__init__()
        #self.conv1 = SplineConv(1, 64, dim=3, kernel_size=4)
        ##print("Kholous", len(data))
        self.conv0 = GCNConv(3, 64)
        self.bn0 = torch.nn.BatchNorm1d(64)
        
        self.conv1 = GCNConv(64, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = GCNConv(64, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        
        
        self.conv3 = GCNConv(64, 64)
        self.bn3 = torch.nn.BatchNorm1d(64)
        #self.bn1 = BatchNorm(64)

        self.conv4 = GCNConv(64, 64)
        self.bn4 = torch.nn.BatchNorm1d(64)
        


        self.conv5 = GCNConv(64*5, 256)
        self.bn5 = torch.nn.BatchNorm1d(256)
   

        self.conv6 = GCNConv(256, 64)
        self.bn6 = torch.nn.BatchNorm1d(64)
        
        self.co# folder='/home/kucarst3-dlws/YusraMoseg/newtorch/8_10ms_partialonly1/TrainingResults_partial10ms/1_CASE_PointTransfomerPointTransfomer_LR_0.001_EPOCH_20/'
# model=torch.load(folder+'model.pkl')nv7 = GCNConv(64, 16)
        self.bn7 = torch.nn.BatchNorm1d(16)

        self.conv8 = GCNConv(16, 4)
        self.bn8 = torch.nn.BatchNorm1d(4)        
        self.conv9 = GCNConv(4, 2)
        self.bn9 = torch.nn.BatchNorm1d(2)
       


    def forward(self, data):
        data.x = torch.sigmoid(self.conv0(data.x, data.edge_index))
        data.x = self.bn0(data.x)      
        part0=data.x
        data.x = torch.sigmoid(self.conv1( data.x , data.edge_index))

        data.x = self.bn1(data.x)
        part1=data.x
        data.x = torch.sigmoid(self.conv2(data.x, data.edge_index))

        data.x = self.bn2(data.x)
        part2=data.x
        data.x = torch.sigmoid(self.conv3(data.x, data.edge_index))

        data.x = self.bn3(data.x)  
        part3=data.x

        data.x = torch.sigmoid(self.conv4(data.x, data.edge_index))

        data.x = self.bn4(data.x) #concat with part2
        part4=data.x




        data.x=torch.cat((part4, part3, part2, part1 , part0), dim=1)
        data.x = torch.sigmoid(self.conv5(data.x, data.edge_index))

        data.x = self.bn5(data.x) #concat with part1

        data.x = torch.sigmoid(self.conv6(data.x, data.edge_index))
        data.x = self.bn6(data.x) #concat with part0

        data.x = torch.sigmoid(self.conv7(data.x, data.edge_index))
        data.x = self.bn7(data.x)


        data.x = torch.sigmoid(self.conv8(data.x, data.edge_index))
        data.x = self.bn8(data.x)
        data.x = torch.sigmoid(self.conv9(data.x, data.edge_index))
        data.x = self.bn9(data.x)


        out=F.softmax(data.x, dim=1)
        return out



##---------------------------------------------------------------------------------------------------------------------
# STAGE C: My Network parameters
# ---------------------------------------------------------------------------------------------------------------------  

print("STAGE C: BUILDING A NETWORK")
# model=Net_TRANSFORMER().to(device)

# model = Net_PointTransformer(3,2, dim_model=[32, 64, 128, 256, 512], k=16).to(device)
# model = Net_PointTransformer(3,2, dim_model=[32, 64, 128, 256], k=8)
# model = Net_PointTransformer(3,2, dim_model=[32, 64, 128, 256 ], k=16)
# model = Net_PointTransformer(3,2, dim_model=[64 ], k=2)

model = Net_PointTransformer(3,2, dim_model=[64, 64, 64], k=16)
# model = NetConnect_3().to(device)


model=model.double()
print("Model Structure ",model)
model=nn.DataParallel(model)#, device_ids=[0,1,2,3])
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate) #torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)


##---------------------------------------------------------------------------------------------------------------------
# STAGE D: TRAINING STAGE
# ---------------------------------------------------------------------------------------------------------------------  
print("STAGE D: TRAINING STAGE - Feedforward")
loss_func=FocalLoss()



train_dataset=TrainGraphs_all_part0
# train_dataset=traingraphs_wall_02

print("Tranining_part0!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part0))
print("Tranining_part1!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part1))
print("Tranining_part2!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part2))
print("Tranining_part3!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part3))
print("Tranining_part4!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part4))
print("Tranining_part5!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part5))
print("Tranining_part6!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part6))
print("Tranining_part7!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part7))
print("Tranining_part8!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part8))
print("Tranining_part9!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part9))
print("TrainGraphs_all_part10!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part10))
print("TrainGraphs_all_part11A!!!!!!!!!!!!!!!!!!!!!", len(TrainGraphs_all_part11A))



###################################################################################
##---------------------type of learning graphs-----------------------------------##
###################################################################################
TrainGraphs_all_part11=TrainGraphs_all_part11A

train_loader_part0 = DataLoader(TrainGraphs_all_part0+TrainGraphs_all_part10, batch_size=batch_size, shuffle=True)
train_loader_part1 = DataLoader(TrainGraphs_all_part1+evimo2_seq6_9, batch_size=batch_size, shuffle=True)
train_loader_part2 = DataLoader(TrainGraphs_all_part2+TrainGraphs_all_part10, batch_size=batch_size, shuffle=True)
train_loader_part3 = DataLoader(TrainGraphs_all_part3+evimo2_seq10, batch_size=batch_size, shuffle=True)
train_loader_part4 = DataLoader(TrainGraphs_all_part4+TrainGraphs_all_part10, batch_size=batch_size, shuffle=True)
train_loader_part5 = DataLoader(TrainGraphs_all_part5+evimo2_seq12_13, batch_size=batch_size, shuffle=True)
train_loader_part6 = DataLoader(TrainGraphs_all_part6+TrainGraphs_all_part10, batch_size=batch_size, shuffle=True)
train_loader_part7 = DataLoader(TrainGraphs_all_part7+evimo2_seq14_15, batch_size=batch_size, shuffle=True)
train_loader_part8 = DataLoader(TrainGraphs_all_part8+TrainGraphs_all_part10, batch_size=batch_size, shuffle=True)
train_loader_part9 = DataLoader(TrainGraphs_all_part9, batch_size=batch_size, shuffle=True)
train_loader_part10 = DataLoader(TrainGraphs_all_part10, batch_size=batch_size, shuffle=True)
train_loader_part11 = DataLoader(TrainGraphs_all_part11, batch_size=batch_size, shuffle=True)


# breakpoint()
loss_func=FocalLoss()
# model = model.float()

folder="/home/kunet.ae/100048632/ModelABCD_v2/30_Sept_withevimo2_AtypeV4/TrainingResults_wMOD_EVIMO2_3L64wG/ModelC_3L64wG__LR_0.001_EPOCH_1000batch96/"
PATH=folder+'model_weights.pth'
model.load_state_dict(torch.load(PATH))

model.eval()
print("model loaded is done", model)



#-----------------------------------------------------------------------------------------------------------------------------------------
epoch_losses= TRAINING_MODULE(model, number_of_epoch, train_loader_part0, train_loader_part1, train_loader_part2, train_loader_part3, train_loader_part4, train_loader_part5, train_loader_part6, train_loader_part7, train_loader_part8, train_loader_part9, train_loader_part10,train_loader_part11, FOLDERTOSAVE)
#-----------------------------------------------------------------------------------------------------------------------------------------

with open(FOLDERTOSAVE+'losses.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows([[loss] for loss in epoch_losses])
    csvFile.close()


# # ---------------------------------------------------------------------------------------------------------------------
# # STAGE E: TESTING STAGE
# # ---------------------------------------------------------------------------------------------------------------------  
print("STAGE E: TESTING STAGE")
