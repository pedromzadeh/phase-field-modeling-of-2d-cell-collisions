import numpy as np
import tools as tl
import matplotlib.pyplot as plt
import substrates
import polarity
import json
import time

# This is the new F_sub implementation, wherein F_sub is taken from the Cao+Camley paper.
# F_CH and F_sub have been tested extensively. The results agree with Surface Evolver when wetting.

def compute_energies(cells,i,sim_params):
   '''Computes the total free energy of the configuration of `cells`. This method is NOT used for proper analysis, but rather
   for debugging and comparing against theoretical calculations. As such, it contains whatever was needed at the last time of
   use. It is not complete and constantly maintained.'''
   _,_,N_mesh,L_box,lam,kappa,omega,mu,_,xi,_,_,dx = sim_params

   all_grad_tups = [tl.compute_gradients(cell.phi,dx) for cell in cells]
   grads_x = [el[0] for el in all_grad_tups]
   grads_y = [el[1] for el in all_grad_tups]
   laps = [el[2] for el in all_grad_tups]

   F_CH, F_area = 0,0
   F_sub = 0
   gamma_sub = 0
   for k in range(1):
      R_eq,A,g = cells[k].R_eq, cells[k].sub_A, cells[k].sub_R
      gamma = cells[k].gamma
      phi = cells[k].phi

      grad_phi_sqrd = grads_x[k]**2 + grads_y[k]**2
      integrand = 4*phi**2*(1-phi)**2 + lam**2*grad_phi_sqrd
      F_CH += np.sum(integrand)*dx**2 * gamma/lam

      area = np.sum(phi**2)*dx**2
      area_target = np.pi * R_eq**2
      F_area += mu * ( 1 - area/area_target)**2

      # considering cells only on the bottom substrate
      sub = substrates.Substrate(N_mesh,L_box,'rectangular',xi)
      arr = np.linspace(0,L_box,N_mesh)
      x,y = np.meshgrid(arr,arr)
      chi = sub.get_sub(x,y)
      chi_yB = sub.get_sub(x,y+lam)
      W = -36*A/xi*chi**2*(1-chi)**2 + 0.5*g*chi_yB
      integrand = phi**2*(2-phi)**2 * W
      F_sub += np.sum(integrand)*dx**2

      # 1D integral, giving gamma_sub
      integrand_1d = integrand[:,int(N_mesh/2)]
      gamma_sub = np.sum(integrand_1d)*dx

      adh_integrand = (cells[0].phi**2*(1-cells[0].phi)**2) * (cells[1].phi**2*(1-cells[1].phi)**2)
      F_adh = np.sum(adh_integrand)*dx**2 * (-omega/lam)

      rep_integrand = cells[0].phi**2 * cells[1].phi**2
      F_rep = np.sum(rep_integrand)*dx**2 * (kappa/lam)

   return F_CH, F_area, F_sub, gamma_sub, area, F_adh, F_rep

def compute_functional_derivative(cells,i,sub,sim_params,n):
   '''Computes the functional derivative of the free energy above. `i` is the cell index with respect to which the functional derivative is taken.
   `cells` is the collection of all cells. Note the functional derivative w.r.t substrate must be added to this. `sub` is a `Substrate` instance,
   from which you can obtain the spacial substrate of whatever predescribed `sub.type` by calling `sub.get_sub(x,y)`.

   *** NOTE, when the substrate configuration changes so must the functional derivative form.***
   '''
   # unpack simulation parameters
   _,_,N_mesh,L_box,lam,kappa,omega,mu,_,xi,_,_,dx = sim_params
   R_eq,A,g, gamma = cells[i].R_eq, cells[i].sub_A, cells[i].sub_R, cells[i].gamma

   # obtain gradients and laplacians
   # grads_x, grads_y, laps: list of 2d arrays of size equal to number of cells
   all_grad_tups = [tl.compute_gradients(cell.phi,dx) for cell in cells]
   grads_x = [el[0] for el in all_grad_tups]
   grads_y = [el[1] for el in all_grad_tups]
   laps = [el[2] for el in all_grad_tups]

   # main cell's phi & laplacian
   phi = cells[i].phi
   grad_x, grad_y = grads_x[i], grads_y[i]
   lap_phi = laps[i]

   sum_over_phi_sqrd = 0
   sum_over_lap_phi = 0
   sum_over_interface = 0
   sum_over_grads_sqrd = 0
   sum_over_gradsx_lap = 0
   sum_over_gradsy_lap = 0
   for cell in cells:
      ind = cells.index(cell)
      if ind != i:
         sum_over_phi_sqrd += cell.phi**2
         sum_over_lap_phi += laps[ind]
         sum_over_interface += cell.phi**2*(1-cell.phi)**2
         sum_over_grads_sqrd += grads_x[ind]**2 + grads_y[ind]**2
         sum_over_gradsx_lap = grads_x[ind] * laps[ind]
         sum_over_gradsy_lap = grads_y[ind] * laps[ind]

   # compute cell-cell interaction forces
   neighbor_term_1 = 4 * kappa * (1/lam) * sum_over_phi_sqrd * phi
   neighbor_term_2 = -2 * omega * (1/lam) * (phi*(1-phi)**2 - phi**2*(1-phi)) * sum_over_interface # O3

   # compute the integral and area conserving term
   area = dx**2 * np.sum(phi**2)
   area_term = (4*mu)/(np.pi*R_eq**2) * (area/(np.pi*R_eq**2) - 1) * phi

   # copmute the Cahn-Hilliard terms
   CH_term = 8 * (gamma/lam) * phi * (1-phi) * (1-2*phi) - 2 * gamma * lam * lap_phi

   dF_dphi_thermo = CH_term + neighbor_term_1 + neighbor_term_2 + area_term

   # compute substrate-based functional derivatives
   # for cell on a floor
   arr = np.linspace(0,L_box,N_mesh)
   x,y = np.meshgrid(arr,arr)
   chi = sub.get_sub(x,y)
   chi_yB = sub.get_sub(x,y+lam)
   W = -36*A/xi*chi**2*(1-chi)**2 + 0.5*g*(chi_yB)
   dF_dphi_sub = 4*(-2+phi)*(-1+phi)*phi*W

   # for cell confined in a box
   # chi_yT = sub.get_sub(x,y-lam)
   # chi_xL = sub.get_sub(x-lam,y)
   # chi_xR = sub.get_sub(x+lam,y)

   return dF_dphi_thermo + dF_dphi_sub

def update(cell,cells_n,i,motile_force_mode,sub,sim_params,n,n_collision):
   '''Update `cell.phi` from `n` to `n+1` timestep according to relevant quantities at the current time step `n`.
      `cell`         -- cell being updated
      `cells_n`      -- list of all cells at timestep `n`
      `i`            -- index of `cell` in `cells_n`
      `motile_force` -- mode of the motility force
      `sub`          -- instance of the Substrate class, from which a substrate can be obtained
      `sim_params`   -- tuple of relevant simulation parameters
      `n`            -- current time step of the simulation
      `n_collision`  -- simulation time at which a collision is detected; used in CG_FR polarity 
      The procedure is as follows:
         (1) use v_n, F_n, theta_n to update phi_n to phi_{n+1},
         (2) use phi_{n+1} and phi_n to compute dr_cm and compute v_cm^{n},
         (3) use v_cm^{n} to update theta_n to theta_{n+1} <-- polarity module does this and returns theta_{n+1},
         (4) use theta_n to set pn and then compute velocity fields v_{n+1}.
   '''
   # unpack simulation parameters
   _,dt,N_mesh,L_box,_,_,_,_,eta,xi,T_WETTING,_,dx = sim_params
   prot_v,D,J = cell.prot_v, cell.D, cell.J
   t_hat = [1,0]

   # obtain relevant variables at time n
   phi_n = cells_n[i].phi
   vx_n = cells_n[i].vx
   vy_n = cells_n[i].vy
   pn = [np.cos(cells_n[i].theta),np.sin(cells_n[i].theta)]
   dF_dphi = compute_functional_derivative(cells_n,i,sub,sim_params,n)
   grad_x, grad_y, lap = tl.compute_gradients(phi_n,dx)
   v_dot_gradphi = vx_n*grad_x + vy_n*grad_y

   # update phi_n --> phi_(n+1)
   cell.phi += -dt * (dF_dphi + v_dot_gradphi)
   cell.cm = tl.cm(cell)

   # using phi_(n+1), compute v_cm(n) and update the class variable
   vn = tl.compute_velocity(cell,cells_n[i],dt)
   cells_n[i].v_cm = vn

   # update cell polarity pn --> p_(n+1) via updating theta
   cell.theta += polarity.CG_FR(cells_n,i,t_hat,6,3,D,dt,dx,n,n_collision)

   # compute active forces and update velocity field v_n --> v_{n+1}
   if motile_force_mode == 'None':
      fx_motil = 0.0
      fy_motil = 0.0
   elif motile_force_mode == 'uniform':
      alpha = 0.1
      fx_motil = alpha * phi_n * pn[0]
      fy_motil = alpha * phi_n * pn[1]
   elif motile_force_mode == 'partosub':
      # cell-substrate force generation, using nth timestep data
      # [all mathematical procedures are applied element-wise]

      arr = np.linspace(0,L_box,N_mesh)
      x,y = np.meshgrid(arr,arr)
      chi = sub.get_sub(x,y)
      grad_chix, grad_chiy, _ = tl.compute_gradients(chi,dx)

      grad_phi = np.array([grad_x,grad_y])
      grad_chi = np.array([grad_chix,grad_chiy])
      p_field = np.array([np.ones(phi_n.shape)*pn[0],np.ones(phi_n.shape)*pn[1]])
      norm_grad_phi = np.sqrt(np.sum(grad_phi*grad_phi,axis=0))
      norm_grad_chi = np.sqrt(np.sum(grad_chi*grad_chi,axis=0))
      n_field = -1 * grad_phi / (norm_grad_phi + 1E-10)
      p_dot_t = np.dot(pn,t_hat)
      p_dot_n = np.sum(p_field*n_field,axis=0)
      H_p_dot_n = np.where(p_dot_n > 0, 1, 0)
      # prior to computing the substrate force, do a smooth low threshold on norm_grad_phi to remove
      # any interface thickness from creeping into the physical force computation
      eps = 0.25
      T = 0.1
      soft_T_norm_grad_phi = 0.5*(np.tanh((norm_grad_phi-T)/eps) + 1)
      magnitude = prot_v * soft_T_norm_grad_phi * norm_grad_chi * H_p_dot_n * p_dot_t
      fx_motil = magnitude * t_hat[0]
      fy_motil = magnitude * t_hat[1]

   else:
      raise ValueError('Motility force not set.')

   # compute thermal forces
   fx_thermo = dF_dphi * grad_x
   fy_thermo = dF_dphi * grad_y

   cell.vx = (fx_thermo + fx_motil) / eta
   cell.vy = (fy_thermo + fy_motil) / eta
