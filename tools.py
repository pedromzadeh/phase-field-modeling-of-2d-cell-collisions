import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import time

class Cell:
   '''Defines a cell class, containing the attributes that make up a cell.
   phi    := 2d array of size N_mesh1D x N_mesh1D, holds the phase field for this cell.
   cntr   := (x,y) contour points denoting phi = 1/2.
   R_init := undeformed initial radius.
   R_eq   := equilibrium radius of the cell.
   cm     := tuple of form (x,y), holds the CM of the cell.
   theta  := polarity direction of the cell.
   v_cm   := array denoting the center of mass velocity of the cell from n-1 to n.
   vx, vy := 2d array of size N_mesh1D x N_mesh1D, holds the velocity of each cell.
             In this treatment, each pixel has a velocity, hence the 2d array.
   AR     := aspect ratio of the cell.
   prot_v := speed of protrusion when cell is motile on an adhesive substrate.
   gamma  := surface tension of the cell.
   sub_A  := strength of substrate adhesion.
   sub_R  := strength of substrate repulsion.
   D      := cell's angular diffusion coefficient.
   J      := cell's alignment strength.
   ca     := cell's contact angle -- if not relevant, it will be set to -1.
   '''
   def __init__(self,N_mesh,cm,l_params):
      # unpack all relevant simulation parameters to initialize the cell
      R_init,R_eq,gamma,A,g,prot_v,D,J = l_params

      self.phi = np.zeros(shape=(N_mesh,N_mesh))
      self.cntr = np.zeros(shape=(N_mesh,N_mesh))
      self.cm = cm
      self.R_init = R_init
      self.R_eq = R_eq
      self.gamma = gamma
      self.sub_A = A
      self.sub_R = g
      self.prot_v = prot_v
      self.D = D
      self.J = J
      self.theta = np.random.rand()*np.pi
      self.v_cm = [0,0]
      self.vx = 0*np.ones(shape=(N_mesh,N_mesh))
      self.vy = 0*np.ones(shape=(N_mesh,N_mesh))
      self.rcil = [0,0]

def tanh(r,R,epsilon):
   '''This is used to initialize the phase field for each cell.'''
   return 1/2 + 1/2 * np.tanh(-(r-R)/epsilon)

def init_phi(cell,dx):
   '''Computes the phase field of `cell` using the initializiation tanh function. I've made this
   return `cell` simply because of the way I want to shorthand looping through cells.'''
   # init parameters & using meshgrid to loop (instead of an explicit for loop)
   radius = cell.R_init
   epsilon = 1
   N = cell.phi.shape[0]
   one_dim = np.arange(0,N,1)
   x,y = np.meshgrid(one_dim,one_dim)
   r = np.sqrt((cell.cm[1] - y*dx)**2 + (cell.cm[0] - x*dx)**2)
   cell.phi[y,x] = tanh(r,radius,epsilon)
   return cell

def cm(cell):
   '''Computes the center-of-mass of the `cell` as `\\vec{r}=1/m \\int{dr r phi}, where m=\\int{dr phi}. If the `cell` is wrapping
   through edges via PBCs, this method fails to correctly compute the cm. `cell` must be whole in one unit in space. Returns a list [cmx,cmy] in scaled simulation units.'''
   # way 1: can't deal with edges
   dx = 50/199
   x = np.arange(0,200,1)
   xg,yg = np.meshgrid(x,x)
   a,b = np.sum(xg*cell.phi), np.sum(yg*cell.phi)
   c = np.sum(cell.phi)
   r_cm = [a/c*dx, b/c*dx]
   return r_cm

def compute_area(cell,dx):
   '''Computes the area of the `cell`, according to `cell.phi` and `dx`. Recall that `dx` is
   the spacing between adjacent mesh points and it shows up in the discrete integral.'''
   phi = cell.phi
   area = np.sum(phi**2 * dx**2)
   return area

def compute_velocity(curr,prev,dt):
   '''Computes the center of mass velocity by subtracting the cm positions of cell.phi at
      n and n+1.
      `curr` and `prev` are cell type objects.
   '''
   d_cm = np.array(curr.cm) - np.array(prev.cm)
   dv = d_cm / dt
   return dv

def compute_gradients(phi,dx):
   '''Computes grad(`phi`) and laplace(`phi`). The numerical method is
   finite difference, and we use the 4-point nearest neighbor approach. PBCs are imposed. Note
   that gradients and laplacian are 2d arrays to capture the spatial dependence of the quantities.
   This method has been tested and confirmed to work correctly by solving the 2D heat equation'''
   # compute the gradients, each component is separate
   phi_right = np.roll(phi,1,axis=1)
   phi_left = np.roll(phi,-1,axis=1)
   phi_up = np.roll(phi,-1,axis=0)
   phi_down = np.roll(phi,1,axis=0)

   # compute the gradients, each component is separate
   grad_x = (phi_left - phi_right) / (2*dx)
   grad_y = (phi_up - phi_down) / (2*dx)

   # compute laplacian
   laplacian = (phi_left + phi_right + phi_up + phi_down - 4*phi) / dx**2

   return (grad_x, grad_y, laplacian)

def Heaviside(x):
   '''Returns the Heaviside function evaluated at `x`. H(0)=1 here because I need this in the 
   CG-FR polarity mechanism.'''
   if x >= 0:
      return 1
   else:
      return 0

def sgn(x):
   '''Returns the sign of `x`.'''
   if x == 0:
      return 0
   elif x > 0:
      return 1
   else:
      return -1

def measure_contact_ang(cell,scale):
   '''Measures the contact anlge `cell.phi` contour makes with the adhesive floor. There will be
      two angles measured, one for each side of the contour, and the desired one is manually selected.
      The procedure is to use the scatter points from plt.contour, find the two x-direction extrema, select a
      few of the points and make a line out of them. Find contact anlge from the slope of the lines.'''

   cntr = cell.cntr
   x,y = cntr[0][:,1], cntr[0][:,0]

   xmin,xmax = np.min(x), np.max(x)
   ymin,ymax = np.min(y), np.max(y)
   AR = np.fabs(xmax-xmin)/np.fabs(ymax-ymin)

   # Because the contour points start from the top and go counter clockwise, ind_xmin first
   # matches with y value > ymin. For this reason, we get two indices where xmin occurs, and the second
   # one matches with ymin. Sometimes you get one index, so account for that. ind_xmax is fine as is since
   # the first occurance matches with ymin.

   buffer = 1
   y_on_sub = ymin + buffer
   y_slice = np.where(np.fabs(y-y_on_sub)<1)[0]
   xmin,xmax = np.min(x[y_slice]), np.max(x[y_slice])
   ind_xmin,ind_xmax = np.where(x==xmin)[0], np.where(x==xmax)[0]
   ind_xmin = ind_xmin[1] if len(ind_xmin)>1 else ind_xmin[0]
   ind_xmax = ind_xmax[0]

   # if phi is at the border, return nan
   dict_NAN = {'theta_low' : float('nan'), 'theta_high' : float('nan'), 'x_low' : [], 'y_low' : [], 'x_high' : [], 'y_high' : [], 'AR' : AR}
   if xmin-1 < 0 or xmax+1 > int(50/scale):
      return dict_NAN

   low_pts_x,low_pts_y = x[ind_xmin-10:ind_xmin-4], y[ind_xmin-10:ind_xmin-4]
   high_pts_x,high_pts_y = x[ind_xmax+4:ind_xmax+10], y[ind_xmax+4:ind_xmax+10]

   # compute slopes for the low- and high- bound lines; the smaller one is the contact angle
   x1L,x2L = low_pts_x[0], low_pts_x[-1]
   y1L,y2L = low_pts_y[0], low_pts_y[-1]
   x1H,x2H = high_pts_x[0], high_pts_x[-1]
   y1H,y2H = high_pts_y[0], high_pts_y[-1]

   # don't want to print division by zero warnings; handle internally
   old_settings = np.seterr(all='ignore')

   s_lowbound = (y1L - y2L) / (x1L - x2L)
   s_highbound = (y2H - y1H) / (x2H - x1H)

   if s_lowbound == np.inf or s_highbound == np.inf:
      return dict_NAN

   ang_lowbound = np.arctan(s_lowbound) * 180/np.pi
   ang_highbound = np.arctan(s_highbound) * 180/np.pi

   # compute the lines for visuals
   b_lowbound = y1L-s_lowbound*x1L
   b_highbound = y1H-s_highbound*x1H

   x_low,x_high = np.arange(xmin-5,xmin+5,1), np.arange(xmax-5,xmax+5,1)
   y_low,y_high = s_lowbound*x_low + b_lowbound, s_highbound*x_high+b_highbound

   # scale back to simulation space [0:50,0:50] --- scale = L_box/N_mesh
   x_low *= scale
   y_low *= scale
   x_high *= scale
   y_high *= scale

   dict = {'theta_low' : ang_lowbound, 'theta_high' : ang_highbound, 'x_low' : x_low, 'y_low' : y_low, 'x_high' : x_high, 'y_high' : y_high, 'AR' : AR}
   return dict

def make_measurements(cells_n,scale):
   '''This will compute various statistics and return them as a tuple. `cells_n` is the array of cells at step n.'''

   ARs, cms, ps, prot_vs, gammas, SA, SR, vs, cas = [],[],[],[],[],[],[],[],[]
   for cell in cells_n:
      # speed
      v = cell.v_cm
      v = np.sqrt(v[0]**2+v[1]**2)

      # contact angle
      ca_dict = measure_contact_ang(cell,scale)
      theta_low = ca_dict['theta_low']
      theta_high = ca_dict['theta_high']
      x_low,x_high = ca_dict['x_low'], ca_dict['x_high']
      y_low,y_high = ca_dict['y_low'], ca_dict['y_high']
      tup_lines0 = (x_low,x_high,y_low,y_high)

      if cells_n.index(cell) == 0:
         if theta_high > 0:
            ca = 180 - theta_high
         else:
            ca = np.fabs(theta_high)
      else:
         if theta_low < 0:
            ca = 180 + theta_low
         else:
            ca = theta_low

      ARs.append(ca_dict['AR'])
      cms.append(cell.cm)
      ps.append(cell.theta)
      prot_vs.append(cell.prot_v)
      gammas.append(cell.gamma)
      SA.append(cell.sub_A)
      SR.append(cell.sub_R)
      vs.append(v)
      cas.append(ca)

   return ARs, cms, ps, prot_vs, gammas, SA, SR, vs, cas

def count_flipped_trajectories(arr):
   '''Given the `arr` of locomotion, returns the number of times the cell flips between 0 to 1.'''
   indices = np.where(np.diff(np.array(arr))!=1)[0]+1
   arr_split = np.split(arr,indices)
   counts = [len(segment) for segment in arr_split]
   return np.array(counts)
