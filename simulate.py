import tools as tl
import potentials as ptl
import plots as mplt
import substrates
import numpy as np
import time
import copy
import sys
import json
import multiprocessing
import os
import traceback
from skimage import measure

def run(r,param_id):
   '''Constitues one full simulation run of two cells colliding head-on on an adhesive substrate. 
         1. Wet cells
         2. Actively drive them towards each other
         3. When cells are within a certain distance of each other, begin collecting direction information
            to decide if collision is head-on or head-tail; also monitor for the collision time
         4. Post collision, assess if cells are in contact and have equilibrated; determine the train's direction '''

   # time based seeding
   np.random.seed(int(time.time())+r)

   # read global_params and all local_params
   with open('Params/global_params.json', 'r') as jfile:
      global_params = json.load(jfile)

   N = int(global_params['T'])                    # total simulation time steps
   dt = float(global_params['dt'])                # time step for differentiation
   N_mesh = int(global_params['N_mesh'])          # 1D number of points on which phi is evaluated
   L_box = int(global_params['L_box'])            # physical (real) dimension of the simulation box
   lam = float(global_params['lam'])              # scaling factor, units of L^2
   kappa = float(global_params['kappa'])          # controls strength of repuslive term
   omega = float(global_params['omega'])          # controls strength of adhesive term
   mu = float(global_params['mu'])                # controls strength of area conservation
   eta = float(global_params['eta'])              # friction coefficient
   xi = float(global_params['xi'])                # friction coefficient
   T_WETTING = float(global_params['T_WETTING'])  # time to wet cells on substrate wall
   savepath = global_params['savepath']           # root directory for storage
   dx = L_box/(N_mesh-1)                          # spacing between mesh points

   # pack global parameters
   sim_params = (N,dt,N_mesh,L_box,lam,kappa,omega,mu,eta,xi,T_WETTING,savepath,dx)
   scale = L_box/N_mesh

   # make appropriate directory for storing run images
   savepath = savepath + 'param{}/Contours_{}/'.format(param_id,r)
   if not os.path.exists(savepath):
      os.makedirs(savepath,exist_ok=True)

   # initialize the substrate of choice by grabbing an instance of it
   sub = substrates.Substrate(N_mesh,L_box,'rectangular',xi)

   # initialize all local cells
   grid = [(6,9), (43,9)]
   cells = []
   for i in range(2):
      with open('Params/param{}/local_{}.json'.format(param_id,i), 'r') as jfile:
         local_params = json.load(jfile)

      R_init = float(local_params['R_init'])        # initial radius of cells (real units)
      R_eq = float(local_params['R_eq'])            # controls the area conservation term (desired radius)
      gamma = float(local_params['gamma'])          # surface tension of the cell
      A = float(local_params['A'])                  # cell-substrate adhesive coefficient
      g = float(local_params['g'])                  # cell-substrate repulsive coefficient
      prot_v = float(local_params['prot_v'])        # speed of protrusion across the substrate
      D = float(local_params['D'])                  # diffusion coefficient
      J = float(local_params['J'])                  # strength of velocity-alignment

      # pack parameters
      l_params = (R_init,R_eq,gamma,A,g,prot_v,D,J)

      # initialize the local cell, and store all local parameters in the cell object
      cell = tl.Cell(N_mesh,grid[i],l_params)
      cell = tl.init_phi(cell,dx)
      cells.append(cell)

   # misc. parameters
   plot_num = 0
   motile_force_mode = 'None'
   n_collision = -1
   left_cmx_at_collision = -1
   collision_state = []
   train_dirs = []
   cells_separated = []
   t_hat = np.array([1,0])
   CellAtEdge = False
   eps_edge = 1.0


   # store statistics as a function of simulation time n
   sim_time = []
   ARs_n, cms_n, ps_n, prot_vs_n, STs_n, SA_n, SR_n, vs_n, cas_n = [],[],[],[],[],[],[],[],[]

   # simulate cells for N steps WHILE no cell has reached the edge of the simulation box
   # print run number 
   print('Run # ', r)
   for n in range(N):
      if not CellAtEdge:
         # wet all cells for T_WETTING steps; then push them toward each other; then let free before collision
         if n < T_WETTING:
            cells[0].theta = -np.pi/2
            cells[1].theta = -np.pi/2
            motile_force_mode = 'uniform'
         elif n < T_WETTING + 1000:
            cells[0].theta = 0
            cells[1].theta = -np.pi
            motile_force_mode = 'partosub'
         else:
            motile_force_mode = 'partosub'

         # make a deep copy of cells at this time t prior to updating
         cells_n = copy.deepcopy(cells)

         # update every cell
         for cell in cells:
            k = cells.index(cell)
            ptl.update(cell,cells_n,k,motile_force_mode,sub,sim_params,n,n_collision)

         # first get the contour points
         cntr = [measure.find_contours(cell.phi,0.5) for cell in cells]
         cntr_l,cntr_r = cntr[0], cntr[1]
         cells[0].cntr = cntr_l
         cells[1].cntr = cntr_r
         x_l,x_r = cntr_l[0][:,1], cntr_r[0][:,1]
         x_lmax,x_rmin = np.max(x_l), np.min(x_r)
         x_lmin,x_rmax = np.min(x_l), np.max(x_r)

         # update CellAtEdge
         if (x_lmin*scale) < eps_edge or x_rmax*scale > (L_box-eps_edge):
            CellAtEdge = True

         # when no collision has happened
         if n > T_WETTING and n_collision == -1:
            # check for collision time
            if np.fabs(x_lmax-x_rmin) < lam:
               n_collision = n
               left_cmx_at_collision = tl.cm(cells[0])[0]

            # collect pre collision directions if cells within certain distance
            # do not care about cells turning before collision
            # care about whether the collision is head-on or head-tail
            cms_x = [tl.cm(cell)[0] for cell in cells_n]
            if np.fabs(np.diff(cms_x)[0]) < 5*cells[0].R_eq:
               vs = [cell.v_cm for cell in cells_n]
               vdott_l, vdott_r = np.dot(t_hat,vs[0]), np.dot(t_hat,vs[1])

               if vdott_l > 0 and vdott_r < 0:
                  collision_state.append('HH')
               elif vdott_l > 0 and vdott_r > 0:
                  collision_state.append('HT')
               elif vdott_l < 0 and vdott_r < 0:
                  collision_state.append('HT')
               else:
                  None

         # post collision
         if n_collision != -1:
            # check if train has equilibrated and cells are in good contact
            cells_in_contact = False
            train_equilibrated = False
            if np.max(cells_n[0].phi*cells_n[1].phi) > 0.15:
               cells_in_contact = True
            if np.fabs(tl.cm(cells[0])[0] - left_cmx_at_collision) > cells[0].R_eq:
               train_equilibrated = True

            # assess train direction
            if train_equilibrated and cells_in_contact:
               vs = [cell.v_cm for cell in cells_n]
               if np.dot(t_hat,vs[0]) < 0 and np.dot(-1*t_hat,vs[1]) > 0:
                  train_dirs.append('Left')
               elif np.dot(t_hat,vs[0]) > 0 and np.dot(-1*t_hat,vs[1]) < 0:
                  train_dirs.append('Right')
               else:
                  None

            # does the train brake after having formed?
            if len(train_dirs) > 0 and np.max(cells_n[0].phi*cells_n[1].phi) < 0.15:
               cells_separated.append(1)

         # measure time-dependent quantities, note `cells` is now updated to n+1, use `cells_n`
         if n%500==0 and n>0:
            ARs, cms, ps, prot_vs, STs, SA, SR, vs, cas = tl.make_measurements(cells_n,scale)
            sim_time.append(n)
            ARs_n.append(ARs)
            cms_n.append(cms)
            ps_n.append(ps)
            prot_vs_n.append(prot_vs)
            STs_n.append(STs)
            SA_n.append(SA)
            SR_n.append(SR)
            vs_n.append(vs)
            cas_n.append(cas)

         if n%2000==0 and n>0:
            mplt.plot_int_ca_speed(cells_n,np.array(sim_time),vs_n,cas_n,plot_num,sim_params,savepath)
            plot_num += 1

      else:
         break

   # Simulation is done. Store everything measured
   path = 'Measurements/param{}/'.format(param_id)

   # direction of train...
   train_dirs = np.array(train_dirs)
   if len(train_dirs) > 0:
      vals,counts = np.unique(train_dirs,return_counts=True)
      max = counts.max()
      i_max = list(counts).index(max)
      if vals[i_max] == 'Left':
         train_dir = 'Left'
      elif vals[i_max] == 'Right':
         train_dir = 'Right'
      else:
         train_dir = 'Unresolved'
   # causes: (1) no collision, (2) collision after T-T_BUFFER, (3) cells not in contact at all
   else:
      train_dir = 'No train'

   # state of collision...
   HH_count, HT_count = collision_state.count('HH'), collision_state.count('HT')
   coll_type = -1
   if HH_count > HT_count:
      coll_type = 0
   else:
      coll_type = 1

   # did the cells separate at any point after collision?
   train_broken = False
   if len(cells_separated) > 0:
      train_broken = True

   if not os.path.exists(path):
      os.makedirs(path,exist_ok=True)

   with open(path + 'Contours_{}.json'.format(r),'w') as f:
      json.dump({'AR' : ARs_n, 'cm' : cms_n, 'polDir' : ps_n, 'prot_v' : prot_vs_n, 'st' : STs_n, 'subA' : SA_n,
      'subR' : SR_n, 'v' : vs_n, 'ca' : cas_n, 'n_collision' : n_collision, 'coll_type' : coll_type, 'train_dir' : train_dir, 'train_broken': train_broken},f)