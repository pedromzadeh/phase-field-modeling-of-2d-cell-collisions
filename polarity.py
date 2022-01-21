import numpy as np
import tools as tl
import matplotlib.pyplot as plt

def va_1(pn,vn,D,J,dt):
   '''Defines the velocity aligning cell polarity mechanism of type 1 -- tau ~ 1/Jv.
   Given the polarity and velocity at step n, returns the new polarity p_{n+1} parametrized
   by theta_{n+1}. `D` and `J` are the angular diffusion coefficient and scale for alignment time, respectively.
   `dt` is the timestep in the simulation.'''

   v_norm = np.sqrt(vn[0]**2+vn[1]**2) + 10E-6
   vxp_z = vn[0]*pn[1] - vn[1]*pn[0]
   gamma = np.arcsin(vxp_z/v_norm)
   tau = 1/(J*v_norm)

   return -(1/tau)*gamma*dt + np.sqrt(D*dt)*np.random.randn()

def va_2(pn,vn,D,tau,dt):
   '''Defines the velocity aligning cell polarity mechanism of type 2 -- tau = const.
   Given the polarity and velocity at step n, returns the new polarity p_{n+1} parametrized
   by theta_{n+1}. `D` and `tau` are the angular diffusion coefficient and alignment time, respectively. [`tau`]=PF.T.
   `dt` is the timestep in the simulation.'''

   v_norm = np.sqrt(vn[0]**2+vn[1]**2) + 10E-6
   vxp_z = vn[0]*pn[1] - vn[1]*pn[0]
   gamma = np.arcsin(vxp_z/v_norm)

   return -(1/tau)*gamma*dt + np.sqrt(D*dt)*np.random.randn()

def CG_FR(cells_n,i,t,tau_CG,tau_CIL,D,dt,dx,n,n_collision):
   '''Defines a polarity mechanism based on contact guidance from the substrate, and front
   repolairzation upon head-head contact to represent CIL.
   `cells_n`      -- list of cell objects at timestep n
   `i`            -- index of cell getting updated
   `t`            -- unit vector parallel to substrate
   `tau_CG`       -- timescale for contact guidance
   `tau_CIL`      -- timescale for CIL upon head-head collision
   `D`            -- angular diffusion coefficient
   `dt`, `dx`     -- timestep and meshgrid spacing
   `n_collision`  -- simulation time at which collision is detected
   
   # front repolarization:
   #   - discrete method:
   #        (1) based on thresh, determine the mutual region of contact between the two cells
   #        (2) use the cell's location to form a mask
   #        (3) use mutual contact region and mask to isolate individual contact region (icr)
   #        (4) from all icr points, compute r_CIL as a UNIT vector
   #   - smooth method:
   #        r_CIL = Norm( \int_cell d2r dr * phi_i * phi_i*phi_j )
   #        (1) create 2d fields, dr_x and dr_y and populate them with the x and y components of
   #            vectors from their respective lattice points to CM of cell
   #        (2) compute above equation for each component
   #        (3) normalize r_CIL
   '''

   # unpack values at step n, cell i is getting updated
   j = 1 if (i==0) else 0
   pns = [[np.cos(cells_n[k].theta),np.sin(cells_n[k].theta)] for k in range(2)]
   phis = [cells_n[k].phi for k in range(2)]
   pn_i,pn_j = pns[i], pns[j]
   phi_i,phi_j = phis[i], phis[j]
   cm = cells_n[i].cm
   
   # contact guidance: align parallel to the substrate
   term_CG = np.arcsin(t[0]*pn_i[1]-pn_i[0]*t[1]) * tl.sgn(np.dot(t,pn_i))

   # front repolarization: model CIL only when (1) cells have collided, (2) cells are making
   # head-head contact
   term_CIL = 0
   in_contact = n_collision != -1 and np.max(phi_i*phi_j) > 0.1

   if in_contact:
      # compute the r_CIL for cell i if cells have collided
      N = len(phi_i)
      dr_x = np.zeros(shape=(N,N))
      dr_y = np.zeros(shape=(N,N))
      y,x = np.where(phi_i>0.5)
      dr_x[y,x] = cm[0]-x*dx
      dr_y[y,x] = cm[1]-y*dx

      # NEW ADDITION: in a corner case of (lowest st=0.9, highest subA=0.64), the last 4 highest
      # protrusion strengths were acting weird. I think it's because the field extends considerably 
      # beyond the 1/2 contour and causes r_cil to fluctuate a lot in the control (left) cell. As a 
      # result, the flatter cell would turn most of the time--an unexpected event given the geometry 
      # sensitivity of this polairty model. This line is meant to mediate the extension of the field 
      # beyond the contour. See notes for more details on this.
      # DO WE WANT TO KEEP THIS THRESHOLDING, OR CALL THE CORNER CASE UNPHYSICAL AND EXCLUDE 
      # FROM ANALYSIS?
      phi_j = np.where(phi_j > 0.2, phi_j, 0)

      r_CIL_x = np.sum(dr_x * phi_i * (phi_i*phi_j))*dx**2
      r_CIL_y = np.sum(dr_y * phi_i * (phi_i*phi_j))*dx**2
      r_CIL = np.array([r_CIL_x,r_CIL_y])
      norm = np.linalg.norm(r_CIL)
      r_CIL = r_CIL / norm
   
      # mere cell collision is NOT sufficient for CIL; need to ensure head-head contact
      hh_contact = tl.Heaviside(-pn_i[0]*pn_j[0]) * tl.Heaviside(r_CIL[0]*pn_j[0])

      # if collided && head-head contact, then enable CIL
      if hh_contact:
         cells_n[i].rcil = r_CIL
         term_CIL = np.arcsin(r_CIL[0]*pn_i[1]-pn_i[0]*r_CIL[1])
      
   # update cell polarity by this amount
   cg = -1/tau_CG*term_CG*dt
   fr = -1/tau_CIL*term_CIL*dt
   noise = np.sqrt(D*dt)*np.random.randn()

   # if n%500==0 and n>0:
   #    if i==0:
   #       print(n)
     
   #    print("Are the cells in contact: ", in_contact)
   #    print("r_cil = ", cells_n[i].rcil)
   #    print("term_CIL = ", term_CIL)
   #    print('cg = {:.7f} \t \t fr = {:.7f} \t \t noise = {:.7f}'.format(cg,fr,noise))

   #    if i==1:
   #       print('\n')   

   return cg + fr + noise