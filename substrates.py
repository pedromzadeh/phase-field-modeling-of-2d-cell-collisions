import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Substrate:
   """This function defines the substrate's phase field, which is constant in time. The return
   value is a 2D array of size (N_mesh x N_mesh). Also note that numerical limits here correspond to
   values in the L_box space."""
   def __init__(self,N_mesh,L_box,type,xi):
      self.N_mesh = N_mesh
      self.L_box = L_box
      self.type = type
      self.xi = xi

   def __str__(self):
      return '\t' + " + You are currently using the {} substrate.".format(self.type)

   def get_sub(self,x,y):
      type = self.type
      if type == 'rectangular':
         return self.rect_sub(x,y)
      elif type == 'circular':
         return self.circ_sub(x,y)
      elif type == 'Y':
         return self.Y_sub(x,y)
      elif type == 'cross':
         return self.cross_sub(x,y)
      elif type == 'tri':
         return self.tri_sub(x,y)
      else:
         raise ValueError('Could not obtain a substrate. Check "type".')

   def rect_sub(self,x,y):
      """rectangular confinement"""
      # controls inteface width of wall
      N_mesh, L_box = self.N_mesh, self.L_box
      dx = L_box/(N_mesh-1)
      eps = self.xi
      xL, xR = 5.5,48
      yB, yT = 5,30
      
      # floor substrate
      chi = 0.5*(1-np.tanh((y-yB)/eps))

      # a rectangular sub
      # chi_y = 0.5*((1-np.tanh((y-yB)/eps))+(1+np.tanh((y-yT)/eps)))
      # chi_x = 0.5*((1-np.tanh((x-xL)/eps))+(1+np.tanh((x-xR)/eps)))
      # chi = chi_x + chi_y

      return chi

   def circ_sub(self,x,y):
      """circular confinement"""
      N_mesh, L_box = self.N_mesh, self.L_box
      Rl, Rs = 18, 10
      dx = L_box/(N_mesh-1)
      x_center,y_center = L_box/2, L_box/2
      a,b,c = 1,1,1
      x = x-x_center
      y = y-y_center
      chi_sqrd = (x/a)**2+(y/b)**2
      chi_sqrd *= c**2
      chi_1 = np.sqrt(chi_sqrd) - Rl
      chi_2 = -(np.sqrt(chi_sqrd) - Rs)
      chi_1 = 1/(1+np.exp(-chi_1))
      chi_2 = 1/(1+np.exp(-chi_2))
      chi = chi_1 + chi_2
      return chi

   def Y_sub(self,x,y):
      """Y substrate: very small J is needed to get high persistence so cells can pass through Y"""
      N_mesh, L_box = self.N_mesh, self.L_box
      dx = L_box/(N_mesh-1)
      width = 3
      eps=0.5
      x = x-25
      y = y-25
      chiL = 1/2*((np.tanh((y-x+width+2.5)/eps)) + (-np.tanh((y-x-width)/eps)))
      chiR = 1/2*((np.tanh((y+x+width+2.5)/eps)) + (-np.tanh((y+x-width)/eps)))
      chiC = 1 - 1/2*((np.tanh((x+width)/eps)) + (-np.tanh((x-width)/eps)))
      chiL_trunc = np.zeros(chiL.shape)
      chiL_trunc[0:50,0:50] = chiL[0:50,0:50]
      chiR_trunc = np.zeros(chiR.shape)
      chiR_trunc[0:50,50:100] = chiR[0:50,50:100]
      chiC_trunc = np.ones(chiC.shape)
      chiC_trunc[50:100,:] = chiC[50:100,:]
      chi = chiC_trunc - chiR_trunc - chiL_trunc
      chi = np.where(chi < 0, 0, chi)
      return chi

   def cross_sub(self,x,y):
      """+ substrate"""
      N_mesh, L_box = self.N_mesh, self.L_box
      width = 3
      eps=0.5
      x = x-25
      y = y-25
      chiH = 1/2 - 1/2*((np.tanh((x+width)/eps)) + (-np.tanh((x-width)/eps)))
      chiV = 1/2 - 1/2*((np.tanh((y+width)/eps)) + (-np.tanh((y-width)/eps)))
      chi = chiV + chiH
      chi = np.where(chi<0.001,0,chi)
      return chi

   def tri_sub(self,x,y):
      """semi triangular sub to environmentally induce lamellipodia-like protrusions"""
      N_mesh, L_box = self.N_mesh, self.L_box
      dx = L_box/(N_mesh-1)

      # horizontal cuts
      eps = 1*dx
      y_B, y_T = 6.6,15
      xL, xR = 1,49
      chiH = 1 - 0.25*((1+np.tanh((y-y_B)/eps)) * (1-np.tanh((y-y_T)/eps)))
      # chiH += 1 - 0.25*((1+np.tanh((x-xL)/eps)) * (1-np.tanh((x-xR)/eps)))

      # triangular cuts
      width = 7
      x = x-25
      y = y-25
      chiL = 1/2*((np.tanh((y-300*x+width+2.5)/eps)) + (-np.tanh((y-600*x-width+10)/eps)))
      chiR = 1/2*((np.tanh((y+300*x+width+2.5)/eps)) + (-np.tanh((y+0.6*x-width+10)/eps)))
      chiL_trunc = np.zeros(chiL.shape)
      chiL_trunc[0:50,0:50] = chiL[0:50,0:50]
      chiR_trunc = np.zeros(chiR.shape)
      chiR_trunc[0:50,50:100] = chiR[0:50,50:100]
      chi = chiR_trunc + chiL_trunc
      chi = np.flip(chi)
      chi = np.roll(chi,-40,axis=0)

      # combine
      chi += chiH
      chi = np.where(chi > 1, 1, chi)
      chi = np.where(chi < 0.1, 0, chi)
      return chi

   def visualize(self,x,y,data,limits=[],extent=[]):
      """Plots the 3D and 2D contour of the substrate"""
      if len(extent) == 0:
         raise KeyError('Extent of contour plot not provided.')

      fig = plt.figure(figsize=(13,7))
      ax1 = fig.add_subplot(131,projection='3d')
      ax2 = fig.add_subplot(132)
      ax3 = fig.add_subplot(133)

      ax1.set_title('(L_box space) Substrate')
      ax1.plot_surface(x,y,data)
      if limits:
         ax1.set_xlim(limits[0])
         ax1.set_ylim(limits[1])
      ax2.set_title('(N_mesh space) Heatmap')
      ax2.imshow(data,origin='lower')
      ax3.set_title('(L_box space) Contour')
      ax3.contour(data,levels=[0.5],extent=extent)
      ax3.axis('equal')
      plt.tight_layout()
      plt.show()
