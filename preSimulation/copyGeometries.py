import numpy
import os
import sys


geometries = [i for i in range(15000,20000)]
for c, g in enumerate(geometries):
	os.system("cp -r /tank/car2d/geometries/"+str(g)+" /scratch/georgioutk/car2d/geometries/")
	if c % 100 == 0:
		print((c-15000)/5000, end="\r", flush=True)
