import os
import pandas as pd
import pickle
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from os.path import join as pjoin

import tensorflow as tf

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=399)
parser.add_argument('--n_his', type=int, default=96)
parser.add_argument('--n_pred', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--ks', type=int, default=4)
parser.add_argument('--kt', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='merge')

args = parser.parse_args()

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt

# W = weight_matrix(pjoin('./dataset', f'PeMSD7_W_{n}.csv'))
# W=weight_matrix(pjoin('./',f'new2_W.csv'))
# dfW = pd.DataFrame(W);
# dfW[dfW>0]=1;
# dfW.to_csv("./new_W.csv",header=None,index=None);

blocks = [[1, 32, 64], [64, 32, 64]]

def read_pickle(path):
	with open(path,'rb') as f:
		data = pickle.load(f)
	return data

data_file='traffic_data.csv'
n_train, n_val, n_test = 3, 3, 3 # 61 in total
step_train,step_val,step_test=8,8,8

PeMS = data_gen(pjoin('./qtraffic', data_file), (n_train, n_val, n_test),(step_train,step_val,step_test), n, n_his + n_pred,day_slot=96)
print(f'>> Loading dataset with Mean: {PeMS.mean:.2f}, STD: {PeMS.std:.2f}')
train_shape=PeMS.get_data('train').shape
val_shape=PeMS.get_data('val').shape
test_shape=PeMS.get_data('test').shape
print(f'datashape as: train:{train_shape},valid:{val_shape},test:{test_shape}')

subset = read_pickle('qtraffic/road_subset.pk');
W = read_pickle('qtraffic/W.pk');
print(f'W.shape={W.shape}')

# Calculate graph kernel
L = scaled_laplacian(W)
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))


if __name__ == '__main__':
	model_train(PeMS, blocks, args,load=False)
	model_test(PeMS, 100, n_his, n_pred, args.inf_mode)	
	# os.system("shutdown -s -t 0");
	pass