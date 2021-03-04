import os
import sys
import subprocess

sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")
sys.stdwar = open(os.devnull, "w")

case = sys.argv[1]
os.chdir("/tank/car2d/simulations/"+case)
mainLog = open("LOG-Main", "w")
subprocess.call("./runfoam.sh", shell=True, stdout=mainLog, stderr=mainLog)