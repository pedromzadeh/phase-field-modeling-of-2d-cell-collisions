import numpy as np
import json
import sys
import os

# This script reads the raw simulation data from Measurements/, and makes two data tables.
# (1) (v_prot,st,subA,dv,dca,Pwin) with shape MxN,6 -- this one is the extended table
# (2) (v_prot,st,subA,dv,dca,Pwin) with shape M,6    -- this one is the averaged table (average over N)
# M is number of parameter sweeps
# N is the number of runs per parameter (N = 96)

def compute_v_ca(v,ca,t_coll):
   '''Input v and ca time series, consider a selected region in time, report the average over that time.'''
   v1,v2 = zip(*v)
   ca1,ca2 = zip(*ca)
   t_l,t_h = int(5000/500), int((t_coll-2000)/500)
   v1,v2 = np.nanmean(v1[t_l:t_h]), np.nanmean(v2[t_l:t_h])
   ca1,ca2 = np.nanmean(ca1[t_l:t_h]), np.nanmean(ca2[t_l:t_h])
   dv = v2-v1
   dca = ca2-ca1

   return dv,dca

if len(sys.argv) != 4:
   print('Ensure you have entered: num_of_params num_of_runs path_to_dir')
   sys.exit(1)

M,N = int(sys.argv[1]), int(sys.argv[2])
sub_path = sys.argv[3]

params_id = np.arange(0,M,1)
Pwin_expanded_table = np.full(shape=(M*N,6),fill_value=np.nan)

# iterate over parameter space
for id in params_id:
   print('Analyzing simulation outcomes for all runs with parameter id: ', id)
   id_path = 'Simulation Data/{}/Measurements/param{}/'.format(sub_path,id)
   n_skipped_runid = []
   st,pv,subA = -1,-1,-1

   # iterate over runs
   for run in range(N):
      run_path = id_path + 'Contours_{}.json'.format(run)
      pwin = -1
      extended_id = (id*N)+run

      if not os.path.exists(run_path):
         print('Check param{} to ensure files exist.'.format(id))
         continue

      with open(run_path,'r') as f:
         data = json.load(f)

      # extract (st,prot_v,subA) for this parameter set from the first run -- same within a parameter 
      # [0][1] gets the modified cell's attributes
      if run == 0:
         st = data['st'][0][1]
         pv = data['prot_v'][0][1]
         subA = data['subA'][0][1]

      # only consider the statistics of head-head collisions that have trained
      # broken trains are ok to include, as the breakage is a post-collision effect
      if data['n_collision'] > 0 and data['coll_type'] == 0 and data['train_dir'] != 'No train':

         # tally up collision stats for this run
         if data['train_dir'] == 'Left':
            pwin = 1
         if data['train_dir'] == 'Right':
            pwin = 0

         # compute v_cm and contact angle; these are tuples, one for each cell
         # subtraction is (right - left)
         dv,dca = compute_v_ca(data['v'],data['ca'],data['n_collision'])
         Pwin_expanded_table[extended_id] = [pv,st,subA,dv,dca,pwin]
         
      else:
         # Pwin_expanded_table is populated with `nan` if a run is skipped
         n_skipped_runid.append(run)

   print('\t Skipped runs = ', n_skipped_runid)


Pwin_split = np.array(np.split(Pwin_expanded_table,int(M)))
Pwin_averaged_table = np.nanmean(Pwin_split,axis=1)

with open('Simulation Data/{}/tabular_stats.json'.format(sub_path), 'w') as jfile:
   json.dump({'Pwin_expanded_table' : Pwin_expanded_table.tolist(), 'Pwin_averaged_table' : Pwin_averaged_table.tolist()}, jfile)

np.set_printoptions(precision=3,edgeitems=300)
print(Pwin_expanded_table)
print(Pwin_averaged_table)
