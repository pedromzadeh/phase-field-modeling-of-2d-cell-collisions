import numpy as np
import multiprocessing
import time
import sys
import simulate
import traceback
import cProfile
import pstats

'''main.py -- Has access to one simulation run, and it modulates how many simulations are run.'''

pid = int(sys.argv[1])
rid = 0
t0 = time.time()

# cProfile.run('simulate.run(rid,pid)', 'stats')
# p = pstats.Stats('stats')
# p.sort_stats('cumtime')
# p.print_stats()
# print('Time taken = ', time.time()-t0)

# simulate.run(rid,pid)
# print('Time taken = ', time.time()-t0)

for q in range(1,3):
   RUNS = 48*q
   start = RUNS - 48
   rs = np.arange(start,RUNS,1)
   try:
      processes = [multiprocessing.Process(target=simulate.run,args=[r,pid]) for r in rs]

      for p in processes:
         p.start()
      for p in processes:
         p.join()

   except Exception:
      print('There was an error with `params_{}.json`:---->'.format(q))
      traceback.print_exc()
      print()

print('Time taken = ', time.time()-t0)