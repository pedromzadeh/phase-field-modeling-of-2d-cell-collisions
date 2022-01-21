from unittest.util import _MIN_BEGIN_LEN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import json
import sys
import statsmodels.stats.proportion as smp
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random
import seaborn

# mpl.rcParams.update({'font.family' : 'serif', 'mathtext.fontset' : 'stix', 'font.size' : '14', 'xtick.labelsize' : 'x-small', 'ytick.labelsize' : 'x-small', 'image.cmap' : 'cividis'})

def read_json(root):
   '''Reads the json file and returns the collision statistics in DataFrame type.'''
   with open(root+'tabular_stats.json', 'r') as jfile:
      data = json.load(jfile)
   Pwin_expanded_table = pd.DataFrame(data['Pwin_expanded_table'],columns=['ps', 'st', 'subA', 'dv', 'dca', 'Pwin'])
   Pwin_averaged_table = pd.DataFrame(data['Pwin_averaged_table'],columns=['ps', 'st', 'subA', 'dv', 'dca', 'Pwin'])

   return Pwin_expanded_table,Pwin_averaged_table

def plot_one_variable_stats(df,bool_exp,c_name,root):
   '''Given `df`, satisfy the `bool_exp` and return the statistics as a function of variable `c_name`.
   Makes a plot of right cell's velocity and contact angle as a function of `c_name`.'''
   df_rounded = df.round({'ps':3, 'st':3, 'subA':3})
   df_one_var = df_rounded.query(bool_exp)
   x, y1, y2 = df_one_var[c_name], df_one_var['dv'], df_one_var['dca']
   
   if df_one_var.empty:
      print('DataFrame is empty with the given bool_exp. Fix bool_exp. Exiting with code -1.')
      return -1

   print(df_one_var)
   plt.figure(figsize=(8,4))
   plt.subplot(121)
   plt.scatter(x,y1+0.25)
   plt.xlabel(c_name)
   plt.ylabel(r'$v\ [\mu m/min]$')
   plt.subplot(122)
   plt.scatter(x,y2+66)
   plt.xlabel(c_name)
   plt.ylabel(r'$ca\ [\degree]$')
   plt.show()
  
def plot_3d_scatters(df,z,root):
   '''Given `df`, makes a colormap in 3D of `z` as a function of the three independent parameters. `z`
   is a `str`.'''
   ps, st, subA, z_var = df['ps'], df['st'], df['subA'], df[z]
   fig = plt.figure(figsize=(6,5))
   ax = fig.add_subplot(111,projection='3d')
   img = ax.scatter(ps,st,subA,c=z_var,cmap='viridis',s=70,alpha=1)
   ax.set_xlabel('Protrusion strength')
   ax.set_ylabel('Surface tension')
   ax.set_zlabel('Adhesion strength')
   ax.view_init(elev=13,azim=-75)

   cbar = fig.colorbar(img)
   if z == 'Pwin':
      z_lb = 'P_{win}'
   elif z == 'dv':
      z_lb = '\delta v'
   else:
      z_lb = '\delta ca'      
   cbar.set_label(r'${}$'.format(z_lb))
   plt.savefig(root+'Visuals/pwin_3d_scatter.png',dpi=500)
   plt.show()

def translate(var_name):
   '''Given `var_name`, output the descriptive name for the variable.'''
   if var_name == 'ps':
      return 'Protrusion strength'
   elif var_name == 'st':
      return 'Surface tension'
   elif var_name == 'subA':
      return 'Adhesion strength'
   elif var_name == 'dv':
      return '$\delta v [\mu m/min]$'
   elif var_name == 'dca':
      return '$\delta \\theta\ [\degree]$'
   else:
      return '$P_{win}$'      

def plot_2d_viewpoints(df,x,y,z,w,root):
   '''Plots `w` as a function of `x` for a fixed value of `y` at a selected value for `z`. 
   `x`, `y`, `z` can be any of `ps`, `st`, or `subA` of `str` type. Each plot will have `S` subplots, where
   `S = len(z)`, and `M = len(y)` lines in each subplot. `len(x)` will span the x-axis.'''
   df = df.round({'ps':3, 'st':3, 'subA':3})
   Z_unique = df[z].unique()
   Y_unique = df[y].unique()
   S = len(Z_unique)
   M = len(Y_unique)

   fig = plt.figure(figsize=(14,12))
   x_lb, y_lb, z_lb, w_lb = translate(x), translate(y), translate(z), translate(w)
   save_lb = '{}({}) vary {} at fixed {}.png'.format(w,x,y,z)
   for s in range(S):
      ax = fig.add_subplot(3,4,s+1)
      z_s = Z_unique[s]
      ax.set_title('{} = {}'.format(z_lb,z_s))
      y_ind = 0
      
      for y_s in Y_unique:
         df_s = df.query('{}=={} and {}=={}'.format(z,z_s,y,y_s))
         ax.plot(df_s[x],df_s[w],color=plt.cm.tab20(y_ind),label='{}'.format(y_s))
         y_ind += 1

      if s >= 8:
         ax.set_xlabel(x_lb)
      if s%4 == 0:
         ax.set_ylabel(w_lb)   

   plt.legend(loc='lower center', ncol=2, title=y_lb, bbox_to_anchor=(1.5,0.15))
   plt.subplots_adjust(hspace=0.3)
   plt.savefig(root+'Visuals/{}'.format(save_lb),dpi=400)
   plt.show()         

def plot_pwin_dv_dca(df,root):
   '''Plot `Pwin` as a function of `dv` and `dca` in 3D.'''
   fig = plt.figure()
   ax = fig.add_subplot(111,projection='3d')
   ax.scatter(df['dv'],df['dca'],df['Pwin'])
   ax.set_xlabel('$\delta v$')
   ax.set_ylabel('$\delta ca$')
   ax.set_zlabel('$P_{win}$')
   ax.view_init(azim=-70,elev=20)
   plt.savefig(root+'Visuals/Pwin(dv,dca).png',dpi=400)
   plt.show()

def knn_clustering(df):
   '''Use knn clustering to create a 3D class phase diagram based on Pwin values for the three
   independent variables.'''
   # knock out all rows with NaN
   df.dropna(inplace=True)
   knn = KNeighborsClassifier(n_neighbors=50,metric='euclidean')
   knn.fit(df[['ps','st','subA']],df['Pwin'])
   ps,st,subA = df['ps'].unique(), df['st'].unique(), df['subA'].unique()
   
   N = 20
   ps = np.linspace(ps.min(),ps.max(),N)
   st = np.linspace(st.min(),st.max(),N)
   subA = np.linspace(subA.min(),subA.max(),N)
   ps,st,subA = np.meshgrid(ps,st,subA)
   X = list(zip(ps.flatten(),st.flatten(),subA.flatten()))
   X_pred = pd.DataFrame(X,columns=['ps','st','subA'])
   y_pred = knn.predict(X_pred)
   X_pred.insert(3,'class',y_pred)

   fig = plt.figure()
   ax = fig.add_subplot(111,projection='3d')
   ax.set_title('KNN prediction')
   ax.scatter(X_pred['ps'],X_pred['st'],X_pred['subA'],c=X_pred['class'],cmap='Set1')
   # ax.view_init(elev=18,azim=-55)
   ax.set_xlabel('Protrusion strength')
   ax.set_ylabel('Surfacec tension')
   ax.set_zlabel('Adhesion strength')
   plt.show()

def plot_pwin_dv_dca_2d(df,root):
   '''Plots `Pwin` vs `dv`, `Pwin` vs `dca`, and `dca` vs `dv`. Divide the plots within 4 quadrants 
   dictated by the `dca` vs `dv` space.'''
   ind_Q1 = df.query('dv > 0 and dca > 0').index
   ind_Q2 = df.query('dv < 0 and dca > 0').index
   ind_Q3 = df.query('dv < 0 and dca < 0').index
   ind_Q4 = df.query('dv > 0 and dca < 0').index
   quadrants = [ind_Q1,ind_Q2,ind_Q3,ind_Q4]
   colors_dark = plt.cm.get_cmap("Set1")

   fig = plt.figure(figsize=(13,4))
   for i in range(4):
      q = quadrants[i]
            
      plt.subplot(131)
      plt.scatter(df.iloc[q]['dv'],df.iloc[q]['Pwin'],color=colors_dark(i),alpha=0.7)
      plt.xlabel('$\delta v\ [\mu m/min]$')
      plt.ylabel('$P_{win}$')
      print(np.max(df.iloc[q]['dv']))
      print(len(df.iloc[q]['dv']))
      
      plt.subplot(132)
      plt.scatter(df.iloc[q]['dca'],df.iloc[q]['Pwin'],color=colors_dark(i),alpha=0.7)
      plt.xlabel('$\delta \\theta\ [\degree]$')
      plt.ylabel('$P_{win}$')
      plt.xlim((-50,50))

      plt.subplot(133)
      plt.scatter(df.iloc[q]['dv'],df.iloc[q]['dca'],color=colors_dark(i),alpha=0.7)
      plt.xlabel('$\delta v\ [\mu m/min]$')
      plt.ylabel('$\delta \\theta\ [\degree]$')

   plt.subplots_adjust(wspace=0.33,bottom=0.15)
   plt.savefig(root+'Visuals/Pwin_summary_vars_stats.png',dpi=400)
   plt.show()

def logreg_1D(df,df_avg,X,y,root,half_space=False):
   '''Perform a logistic regression with independent variables `X` and dependent variable `y`. `X` and `y`
   are labels for the corresponding variables found in `df`.'''
   df.dropna(inplace=True)
   
   if half_space:
      df = df.query('dv > 0 and dca > 0').append(df.query('dv < 0 and dca < 0'))

   X_train,X_test, y_train,y_test = train_test_split(df[X],df[y],test_size=0.2,random_state=0,shuffle=True)
   lr = LogisticRegression(penalty='l2',solver='liblinear')
   lr.fit(X_train,y_train)
   score = lr.score(X_test,y_test)
   print('Classifier status: ', lr.get_params())
   print('----------------------------------------')
   print('Odds multiplicative change per one unit change in {}: '.format(df[X].columns.values),np.exp(lr.coef_))
   print('Bias: ', np.exp(lr.intercept_))
   print('Score of the LR estimator: ', score)
   print('Classes: ', lr.classes_)

   xmin,xmax = df_avg[X].min(),df_avg[X].max()
   xmin,xmax = xmin,xmax
   x_pred = np.linspace(xmin,xmax,500)
   y_pred = lr.predict_proba(x_pred)[:,1]
   plt.plot(x_pred,y_pred,lw=4,color='black',label='LR fit')
   
   ind_Q1 = df_avg.query('dv > 0 and dca > 0').index
   ind_Q2 = df_avg.query('dv < 0 and dca > 0').index
   ind_Q3 = df_avg.query('dv < 0 and dca < 0').index
   ind_Q4 = df_avg.query('dv > 0 and dca < 0').index
   quadrants = [ind_Q1,ind_Q2,ind_Q3,ind_Q4]

   for i in range(4):
      if half_space and i%2==1:
         continue
      q = quadrants[i]
      x,y = df_avg.iloc[q][X],df_avg.iloc[q]['Pwin']
      plt.scatter(x,y,s=30,color=plt.cm.get_cmap("Set1")(i),alpha=0.7)

   sv_lb = 'halfspace' if half_space else ''
   plt.xlabel(translate(X[0]))
   plt.ylabel("$P_{win}$")
   plt.legend()
   # plt.xlim([-0.3,0.3])
   plt.ylim([-0.1,1.1])
   plt.savefig(root+'Visuals/LR_Pwin_{}_quadrants_{}.png'.format(X[0],sv_lb),dpi=400)
   plt.show()

def logreg_2D(df,df_avg):
   '''Perform a logistic regression with all independent variables.'''
   df.dropna(inplace=True)
   
   # ind_Q1 = df.query('dv > 0 and dca > 0').index
   # ind_Q2 = df.query('dv < 0 and dca > 0').index
   # ind_Q3 = df.query('dv < 0 and dca < 0').index
   # ind_Q4 = df.query('dv > 0 and dca < 0').index
   # quadrants = [ind_Q1,ind_Q2,ind_Q3,ind_Q4]

   # df1 = df.iloc[quadrants[0]]
   # df3 = df.iloc[quadrants[2]]
   # df = pd.concat([df1,df3])

   # ind_Q1 = df_avg.query('dv > 0 and dca > 0').index
   # ind_Q2 = df_avg.query('dv < 0 and dca > 0').index
   # ind_Q3 = df_avg.query('dv < 0 and dca < 0').index
   # ind_Q4 = df_avg.query('dv > 0 and dca < 0').index
   # quadrants = [ind_Q1,ind_Q2,ind_Q3,ind_Q4]

   # df1 = df_avg.iloc[quadrants[0]]
   # df3 = df_avg.iloc[quadrants[2]]
   # df_avg = pd.concat([df1,df3])

   X_train,X_test, y_train,y_test = train_test_split(df[['dv','dca']],df['Pwin'],test_size=0.2,random_state=0,shuffle=True)
   lr = LogisticRegression(penalty='l1',solver='liblinear')
   lr.fit(X_train,y_train)
   score = lr.score(X_test,y_test)
   print('Classifier status: ', lr.get_params())
   print('----------------------------------------')
   print('Odds multiplicative change per one unit change in {}: '.format(df.columns.values),np.exp(lr.coef_))
   print('Bias: ', np.exp(lr.intercept_))
   print('Score of the LR estimator: ', score)
   print('Classes: ', lr.classes_)

   x1_pred = np.linspace(df['dv'].min(),df['dv'].max(),500)
   x2_pred = np.linspace(df['dca'].min(),df['dca'].max(),500)
   x_pred = [[x1,x2] for x1,x2 in zip(x1_pred,x2_pred)]
   x_pred = np.array(x_pred)
   y_pred = lr.predict_proba(x_pred)[:,1]
   
   fig = plt.figure()
   ax = fig.add_subplot(111,projection='3d')
   ax.plot(x_pred[:,0],x_pred[:,1],y_pred,lw=4,color='black',label='LR fit')
   ax.scatter(df_avg['dv'],df_avg['dca'],df_avg['Pwin'],label='Simulation',s=10,color='orange')
   plt.legend()
   # plt.savefig('Visuals/LR_Pwin_{}.png'.format(X[0]),dpi=300)
   plt.show()

def plot_box_plots(df):
   '''Plot box-plots of Pwin as functions of dca and dv. `df` is the extended datafile, with Pwin 
   as 1s and 0s for every individual run.'''
   df.dropna(inplace=True)

   ax = seaborn.boxplot(x="Pwin",y="dca",data=df,orient="v")
   ax.set_xticklabels(['Loser', 'Winner'])
   ax.set_xlabel("")
   ax.set_ylabel(translate('dca')+'[$\degree$]')
   plt.savefig('box_whis_dca.png',dpi=500)
   plt.close()

   ax = seaborn.boxplot(x="Pwin",y="dv",data=df,orient="v")
   ax.set_xticklabels(['Loser', 'Winner'])
   ax.set_xlabel("")
   ax.set_ylabel(translate('dv')+'[$\mu m/min$]')
   plt.savefig('box_whis_dv.png',dpi=500)
   plt.close()

def plot_Pwin_dv_dca_phasediagram(df):
   '''Plots `Pwin` as heatmap in scape of (`dv`,`dca`).'''
   plt.scatter(df['dv'],df['dca'],c=df['Pwin'],cmap='viridis')
   plt.xlabel(translate('dv')+'[$\mu m/min$]')
   plt.ylabel(translate('dca')+'[$\degree$]')
   cbar = plt.colorbar()
   cbar.set_label(translate(''))
   plt.show()

def self_compare(file1,file2,x,n):
    '''Plots Pwin vs. `x` for `n` randomly selected points. The points are chosen from `file1`
    and `file2` with all corresponding to the same simulation paremters (ps,subA,st). This is 
    intended to show the variations in Pwin when the same simulation is run more than once. 
    Two plots are created: (1) self comparison plot with `n` points, (2) same plot with points where
    difference in Pwin is larger than 20%, which is the error from Binomial distribution due to counting. 
    `file1`, `file2` are string paths to the files. `x` is a string denoting the variable against
    which Pwin is plotted, and `n` is an integer.'''

    with open(file1) as jfile:
        data1 = json.load(jfile)
    with open(file2) as jfile:
        data2 = json.load(jfile)
    Pwin_averaged_table1 = pd.DataFrame(data1['Pwin_averaged_table'],columns=['ps', 'st', 'subA', 'dv', 'dca', 'Pwin'])
    Pwin_averaged_table2 = pd.DataFrame(data2['Pwin_averaged_table'],columns=['ps', 'st', 'subA', 'dv', 'dca', 'Pwin'])

    indices = random.sample(range(0,Pwin_averaged_table1.shape[0]),n)
    t1, t2 = Pwin_averaged_table1.iloc[indices], Pwin_averaged_table2.iloc[indices]
    diff = t1['Pwin']-t2['Pwin']
    diff_indx_bool = diff > 0.2
    t1_diff, t2_diff = t1[diff_indx_bool], t2[diff_indx_bool]

    plt.scatter(t1[x],t1['Pwin'],c=range(n),cmap='Set2',label='Run 1')
    plt.scatter(t2[x],t2['Pwin'],c=range(n),cmap='Set2',marker='x',label='Run 2')
    plt.ylabel(translate('Pwin'))
    plt.xlabel(translate(x))
    plt.legend()
    plt.savefig('self_compare_Pwin({}).png'.format(x),dpi=400)
    plt.show()

    plt.title('Difference larger than 20%')
    plt.scatter(t1_diff[x],t1_diff['Pwin'],c=range(t1_diff.shape[0]),cmap='Set2',label='Run 1')
    plt.scatter(t2_diff[x],t2_diff['Pwin'],c=range(t2_diff.shape[0]),cmap='Set2',marker='x',label='Run 2')
    plt.ylabel(translate('Pwin'))
    plt.xlabel(translate(x))
    plt.legend()
    plt.savefig('self_compare_Pwin({})_diffThreshold.png'.format(x),dpi=400)
    plt.show()

def plot_two_variable_stats(df,bool_exps,axes_lbs,root):
   '''Plots `Pwin`, `dv` and `dca` as functions of two variables. The fixed variable is defined in
   `bool_exps`, and the (x,y) components of the plots are determined by the ordering of strings in 
   each element of `axes_lbs`.'''
   df_rounded = df.round({'ps':3, 'st':3, 'subA':3})
   plt.figure(figsize=(15,12))
   
   # determine the widest range of all cases to use as the global range on cbar; Pwin is [0,1]
   min_v, max_v = 0,0
   min_dca, max_dca = 0,0
   for bool_exp, axes_lb in zip(bool_exps,axes_lbs):
      df_two_var = df_rounded.query(bool_exp)
      
      min_vl, max_vl = np.min(df_two_var['dv']), np.max(df_two_var['dv'])
      min_dcal, max_dcal = np.min(df_two_var['dca']), np.max(df_two_var['dca'])

      if min_vl < min_v: min_v = min_vl
      if max_vl > max_v: max_v = max_vl
      if min_dcal < min_dca: min_dca = min_dcal
      if max_dcal > max_dca : max_dca = max_dcal

   # iterate over each condition and make 1x3 plots
   for bool_exp, axes_lb in zip(bool_exps,axes_lbs):
      s = 3 * bool_exps.index(bool_exp) + 1
      df_two_var = df_rounded.query(bool_exp)

      if df_two_var.empty:
         print('DataFrame is empty with the given bool_exp. Fix bool_exp. Exiting with code -1.')
         return -1

      x_lb,y_lb = translate(axes_lb[0]), translate(axes_lb[1])
      
      plt.subplot(3,3,s)
      plt.scatter(df_two_var[axes_lb[0]],df_two_var[axes_lb[1]],c=df_two_var['Pwin'],cmap='viridis',s=150)
      plt.xlabel(x_lb)
      plt.ylabel(y_lb)
      if s == 1:
         cbar = plt.colorbar(location='top', pad = 0.1)
         cbar.set_label('$P_{win}$')
         plt.clim(0,1)
      
      plt.subplot(3,3,s+1)
      plt.scatter(df_two_var[axes_lb[0]],df_two_var[axes_lb[1]],c=df_two_var['dv'],cmap='viridis',s=150)
      plt.xlabel(x_lb)
      plt.ylabel(y_lb)
      cbar = plt.colorbar()
      cbar.set_label('$\delta v\ [\mu m/min]$')
      plt.clim(min_v,max_v)

      plt.subplot(3,3,s+2)
      plt.scatter(df_two_var[axes_lb[0]],df_two_var[axes_lb[1]],c=df_two_var['dca'],cmap='viridis',s=150)
      plt.xlabel(x_lb)
      plt.ylabel(y_lb)
      cbar = plt.colorbar()
      cbar.set_label('$\delta \\theta\ [\degree]$')
      plt.clim(min_dca,max_dca)

   plt.tight_layout()
   plt.subplots_adjust(wspace=0.5,hspace=0.3)
   # plt.savefig(root+'Visuals/Pwin_dv_dca.png',dpi=500)
   plt.show()

# ------------- main ------------- #
root = sys.argv[1]
root = 'Simulation Data/' + root + '/'
Pwin_expanded_table, Pwin_averaged_table = read_json(root)

# plot_one_variable_stats(Pwin_averaged_table,'ps==6.4 and st==1.2','subA',root)
# plot_one_variable_stats(Pwin_averaged_table,'ps==6.4 and subA==0.352','st',root)
# plot_one_variable_stats(Pwin_averaged_table,'subA==0.48 and st==1.2','ps',root)

# plot_3d_scatters(Pwin_averaged_table,'Pwin',root)

# plot_2d_viewpoints(Pwin_averaged_table,'ps','st','subA','Pwin',root)
# plot_2d_viewpoints(Pwin_averaged_table,'ps','subA','st','Pwin',root)
# plot_2d_viewpoints(Pwin_averaged_table,'st','subA','ps','Pwin',root)

# plot_pwin_dv_dca(Pwin_averaged_table,root)

# knn_clustering(Pwin_expanded_table)

# plot_pwin_dv_dca_2d(Pwin_averaged_table,root)

# plot_two_variable_stats(Pwin_averaged_table,'subA==0.48',['ps','st'],root)
# plot_two_variable_stats(Pwin_averaged_table,'st==1.2',['ps','subA'],root)
# plot_two_variable_stats(Pwin_averaged_table,'ps==6.4',['st','subA'],root)

plot_two_variable_stats(Pwin_averaged_table,['st==1.2','subA==0.48','ps==6.4'],
[['ps','subA'],['ps','st'], ['st','subA']],root)

# plot_box_plots(Pwin_expanded_table)

# plot_Pwin_dv_dca_phasediagram(Pwin_averaged_table)

# logreg_1D(Pwin_expanded_table,Pwin_averaged_table,['dv'],'Pwin',root,half_space=False)
# logreg_1D(Pwin_expanded_table,Pwin_averaged_table,['dca'],'Pwin',root,half_space=True)

# file1 = 'Simulation Data/Polarity CG_FR (Run 1)/tabular_stats.json'
# file2 = 'Simulation Data/Polarity CG_FR (Run 2)/tabular_stats.json'
# n = 462
# self_compare(file1,file2,'dv',n)
# self_compare(file1,file2,'dca',n)
