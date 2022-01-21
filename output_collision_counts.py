# Go through every collision run and classify how many underwent head-on collision,
# head-tail collision, or unknown. Also, classify how many trained and in what direction.

import numpy as np
import json
import sys
import os
from numpy.lib import nanfunctions

if len(sys.argv) != 5:
   print('Ensure you have entered: start end length_of_dir path_to_dir')
   sys.exit(1)

s,e = int(sys.argv[1]), int(sys.argv[2])
N = int(sys.argv[3])
sub_path = sys.argv[4]

params_id = np.arange(s,e+1,1)

for i in range(len(params_id)):
   id = params_id[i]
   id_path = 'Simulation Data/{}/Measurements/param{}/'.format(sub_path,id)
   HH_colls, HT_colls, unknown_colls = [], [], []
   train_dir = []
   unresolved_run_ids = []
   noTrain_run_ids = []
   broken_trains = []

   for run in range(N):
      run_path = id_path + 'Contours_{}.json'.format(run)
      if not os.path.exists(run_path):
         print('Check param{} to ensure files exist.'.format(id))
         continue
      
      with open(run_path,'r') as f:
         data = json.load(f)
         
      if data['coll_type'] == -1:
         unknown_colls.append(run)
      elif data['coll_type'] == 0:
         HH_colls.append(run)
      else:
         HT_colls.append(run)

      train_dir.append(data['train_dir'])
      if data['train_dir'] == 'No train':
         noTrain_run_ids.append(run)
      if data['train_dir'] == 'Unresolved':
         unresolved_run_ids.append(run)

      if data['train_broken']:
         broken_trains.append(run)


   n_left, n_right = train_dir.count('Left'), train_dir.count('Right')
   n_mixed, n_unresolved = train_dir.count('Mixed'), train_dir.count('Unresolved')
   n_HH, n_HT, n_unknowncoll = len(HH_colls), len(HT_colls), len(unknown_colls)
   n_NoTrain = train_dir.count('No train')
   n_broken_train = len(broken_trains)
   
   if n_unresolved > 0 or n_unknowncoll > 0 or n_NoTrain > 0 or n_HT > 0:
      print('Collision outcomes for all runs with parameter id: ', id)
      print('\t + Collision status:')
      print('\t\t - Head-Head: ({})'.format(n_HH))
      print('\t\t - Head-Tail: ({})'.format(n_HT),HT_colls)
      print('\t\t - Unknown: ', n_unknowncoll)
      print('\t + Total # of trains with unresolved outcome: ', n_unresolved)
      print('\t\t - The corresponding run ids: ', unresolved_run_ids)
      print('\t + Total # of trains with No Train outcome: ', n_NoTrain)
      print('\t\t - The corresponding run ids: ', noTrain_run_ids)
      print('\t + Train broken: ', broken_trains)
      print('----------------------------------------------------------------------------')
