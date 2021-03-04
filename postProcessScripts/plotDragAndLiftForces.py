import os
import numpy
from matplotlib import pyplot as plt

cases = os.listdir("simulations/")
drag = []
lift = []
done = []
for c in range(20000):
	if not os.path.isdir("simulations/"+str(c)+"/2000"):
		continue
	done.append(c)
	forces = numpy.loadtxt("simulations/"+str(c)+"/postProcessing/forceCoeffs1/0/forceCoeffs.dat", dtype=numpy.float32)
	drag.append(forces[-1][2])
	lift.append(forces[-1][3])


drag = numpy.array(drag)
lift = numpy.array(lift)
dragInds = numpy.argsort(drag)
liftInds = numpy.argsort(lift)

validDrag = numpy.logical_and(numpy.logical_not(numpy.isinf(drag)), numpy.logical_not(numpy.isnan(drag)))
validDrag = numpy.logical_and(validDrag, drag>0)
validDrag = numpy.logical_and(validDrag, drag < 2)
validDrag = numpy.where(validDrag)

validLift = numpy.logical_and(numpy.logical_not(numpy.isinf(lift)), numpy.logical_not(numpy.isnan(lift)))
validLift = numpy.logical_and(validLift, lift>-2)
validLift = numpy.logical_and(validLift, lift < 3)
validLift = numpy.where(validLift)

validSims = numpy.intersect1d(validLift[0], validDrag[0], assume_unique=True)
