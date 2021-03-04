import os
import subprocess

cases = os.listdir("simulations/")
times = [str(i) for i in range(100, 2000, 100)]
count = 0
for c in cases:
	# if not os.path.isfile("simulations/"+c+"/runfoam.sh_DONE"):
	# 	subprocess.call("rm -rf simulations/"+c, shell=True)
	# 	continue
	# for t in times:
	if not os.path.isdir("simulations/"+c+"/2000"):
		print(c)
		count += 1
			# print(t, end="\t")
			# print("")
		# subprocess.call("rm -rf simulations/"+c+"/"+t, shell=True)
