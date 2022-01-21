import substrates as subs
import numpy as np

N_mesh = 100
L_box = 50
dx = L_box/(N_mesh-1)
arr = np.linspace(0,L_box,N_mesh)
x,y = np.meshgrid(arr,arr)

sub = subs.Substrate(N_mesh,L_box,'rectangular',0.5)
chi = sub.get_sub(x,y)
print(sub)
sub.visualize(x,y,chi,[[0,50],[0,50]],[0,50,0,50])

