import os
import sys
import pickle
import subprocess
import multiprocessing


log = open("LOG-Main-5K_10K", "w", 1)
sys.stdout = log
sys.stderr = log
sys.stdwar = log

def runCase(case):
	subprocess.call("mkdir -p simulations/"+str(case), shell=True)
	subprocess.call("cp -r testRun/* simulations/"+str(case), shell=True)
	subprocess.call("cp geometries/"+str(case)+"/new_stl_file.stl simulations/"+str(case)+"/constant/triSurface/new_stl_file.stl", shell=True)
	subprocess.call("python3 runSingleSimulation.py "+str(case), shell=True)

# done = set(os.listdir("simulations/"))
cases = set([str(i) for i in range(10000,15000)])
# cases = cases - done
p = multiprocessing.Pool(8)
r = p.map(runCase, cases)
print("Done")