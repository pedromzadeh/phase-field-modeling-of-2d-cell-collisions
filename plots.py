import matplotlib.pyplot as plt
import tools as tl
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter as fmt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import substrates

# matplotlib params:
mpl.rcParams.update({'font.family' : 'serif', 'mathtext.fontset' : 'stix', 'font.size' : '14', 'xtick.labelsize' : 'x-small', 'ytick.labelsize' : 'x-small', 'image.cmap' : 'cividis'})

def plot_int_ca_speed(cells_n,sim_time,speed_tups,ca_tups,n,sim_params,savepath=None,multiLines=None):
   '''Plots the cell interface, as well as cell speed and contact angle as functions of simulation time. This
      is the primary way to visualize the output of simulation runs.
'''
   _,_,N_mesh,L_box,_,_,_,_,_,xi,_,_,dx = sim_params

   xy_tups = [cell.cm for cell in cells_n]
   xcm,ycm = np.array(list(zip(*xy_tups)))

   rcil = [cell.rcil for cell in cells_n]
   rcilx,rcily = np.array(list(zip(*rcil)))

   fig = plt.figure(figsize=(8,7),constrained_layout=True)
   grid = fig.add_gridspec(ncols=2,nrows=2,width_ratios=[1,1],height_ratios=[1,1])
   ax1 = fig.add_subplot(grid[0,:])
   ax2 = fig.add_subplot(grid[1,0])
   ax3 = fig.add_subplot(grid[1,1])

   # setup parameters for plots
   arr = np.linspace(0,L_box,N_mesh)
   x,y = np.meshgrid(arr,arr)
   sub = substrates.Substrate(N_mesh,L_box,'rectangular',xi)
   chi = sub.get_sub(x,y)

   if multiLines != None:
      ca_lines0,ca_lines1 = multiLines
      x_low0,x_high0,y_low0,y_high0 = ca_lines0
      x_low1,x_high1,y_low1,y_high1 = ca_lines1
      # plot tangent lines to contact angle for visualization
      if len(x_low0) > 0:
         ax1.plot(x_low0,y_low0,color='red')
         ax1.plot(x_high0,y_high0,color='red')
      if len(x_low1) > 0:
         ax1.plot(x_low1,y_low1,color='red')
         ax1.plot(x_high1,y_high1,color='red')


   v_tups = [cell.v_cm for cell in cells_n]
   vx,vy = list(zip(*v_tups))
   p_tups = [(np.cos(cell.theta),np.sin(cell.theta)) for cell in cells_n]
   px,py = list(zip(*p_tups))

   # conversion to real units set via considering cells in a circular ring
   # 1 PF.L = 6mu, 1 PF.T = 8min
   PFL_to_MU = 6
   PFT_to_MINS = 8
   V_SCALE = PFL_to_MU/PFT_to_MINS
   v0,v1 = np.array(speed_tups)[:,0], np.array(speed_tups)[:,1]
   ca0,ca1 = np.array(ca_tups)[:,0], np.array(ca_tups)[:,1]

   # plot the contours
   dt = 0.002
   time = sim_time[-1]*dt*PFT_to_MINS
   unit = 'min'
   if time < 1:
      time *= 60
      unit = 's'

   ax1.set_title(r'Interfaces at $t={}$ {}'.format(time,unit))
   ax1.contour(chi,levels=[0.5],colors='black',extent=[0,L_box,0,L_box],linewidths=2)
   [ax1.imshow(cell.phi,origin='lower',extent=[0,50,0,50],cmap='Greys',alpha=0.5) for cell in cells_n]
   [ax1.contour(cell.phi,levels=[0.5],extent=[0,50,0,50],colors=['black'],linewidths=2) for cell in cells_n]
   scalebar = AnchoredSizeBar(ax1.transData,10,r'$60 \mu m$','lower center',
                              pad=0.1,frameon=False,size_vertical=1)

   ax1.add_artist(scalebar)
   ax1.xaxis.set_visible(False)
   ax1.yaxis.set_visible(False)
   ax1.quiver(xcm,ycm,px,py,angles='xy',scale_units='xy',color='blue',label='Polarity',alpha=0.7)
   ax1.quiver(xcm,ycm,vx,vy,angles='xy',scale_units='xy',color='red',label='CM Velocity',alpha=0.7)
   
   ax1.arrow(xcm[0],ycm[0],rcilx[0],rcily[0],color='black',width=0.1,alpha=0.7)
   ax1.arrow(xcm[1],ycm[1],rcilx[1],rcily[1],color='black',width=0.1,label='$\\vec{r}_{CIL}$',alpha=0.7)

   ax1.scatter(xcm,ycm,label='CM',color='black',s=5)
   ax1.legend()
   ax1.set_xlim([0,50])
   ax1.set_ylim([0,25])

   # plot velocity(time)[mu/min] vs [min]
   ax2.plot(sim_time*dt*PFT_to_MINS,v0*V_SCALE,label='Left cell',color='black',ls='dashed')
   ax2.plot(sim_time*dt*PFT_to_MINS,v1*V_SCALE,label='Right cell',color='black',ls='dotted')
   ax2.set_xlabel('Time [min]')
   ax2.set_ylabel(r'$v_{cm}\ [\mu m/min]$')
   ax2.yaxis.set_major_formatter(fmt('%.3f'))
   ax2.legend(loc='upper left')

   # plot contact_angle(time)
   ax3.plot(sim_time*dt*PFT_to_MINS,ca0,label='Left cell',color='black',ls='dashed')
   ax3.plot(sim_time*dt*PFT_to_MINS,ca1,label='Right cell',color='black',ls='dotted')
   ax3.set_xlabel('Time [min]')
   ax3.set_ylabel(r'$\theta$ [$\degree$]')
   ax3.yaxis.set_major_formatter(fmt('%.1f'))
   ax3.legend(loc='upper left')
   plt.savefig(savepath+'img_{}.png'.format(n),dpi=50)
   plt.close()

def plot_int(cells_n,sim_params,savepath,n,q,energies,sim_time):
   '''Plots the cell interface. This is used more for double checking and debugging.'''
   xy_tups = [cell.cm for cell in cells_n]
   xcm,ycm = np.array(list(zip(*xy_tups)))
   v_tups = [cell.v_cm for cell in cells_n]
   vx,vy = list(zip(*v_tups))
   p_tups = [(np.cos(cell.theta),np.sin(cell.theta)) for cell in cells_n]
   px,py = list(zip(*p_tups))

   fig = plt.figure(figsize=(8,7))

   # setup parameters for plots
   _,_,N_mesh,L_box,_,_,_,_,_,xi,_,_,dx = sim_params
   arr = np.linspace(0,L_box,N_mesh)
   x,y = np.meshgrid(arr,arr)
   sub = substrates.Substrate(N_mesh,L_box,'rectangular',xi)
   chi = sub.get_sub(x,y)

   dt = 0.002
   PFL_to_MU = 6
   PFT_to_MINS = 8
   time = n*dt*PFT_to_MINS
   unit = 'min'
   if time < 1:
      time *= 60
      unit = 's'

   ax1 = fig.add_subplot(111)
   ax1.contour(chi,levels=[0.5],colors='black',extent=[0,L_box,0,L_box],linewidths=2)
   # [ax1.imshow(cell.phi,origin='lower',extent=[0,L_box,0,L_box],cmap='Greys',alpha=0.5) for cell in cells_n]
   [ax1.contour(cell.phi,levels=[0.5],extent=[0,L_box,0,L_box],colors=['black']) for cell in cells_n]
   scalebar = AnchoredSizeBar(ax1.transData,10,r'$60 \mu m$','lower center',
                              pad=0.1,frameon=False,size_vertical=1)

   ax1.add_artist(scalebar)
   ax1.xaxis.set_visible(False)
   ax1.yaxis.set_visible(False)
   ax1.quiver(xcm,ycm,px,py,angles='xy',scale_units='xy',color='blue',label='Polarity',alpha=0.7,width=0.005,headwidth=2)
   ax1.quiver(xcm,ycm,vx,vy,angles='xy',scale_units='xy',color='red',label='CM Velocity',alpha=0.7,width=0.005,headwidth=2)
   ax1.scatter(xcm,ycm,label='CM',color='black',s=5)
   ax1.legend()
   ax1.axis('equal')

   ax1.set_title('Time = {:.1f} {}'.format(time,unit))

   plt.savefig(savepath+'img_{}.png'.format(q),dpi=500)
   plt.close()
   # plt.show()
