import sys
import os
import numpy

sys.path.append("/tank/georgioutk/pythonTexTbales")
import Table
# types = ["op", "cc"]
types = ["vc"]
# trainDirs = ["trainedFromCheckpointFullModel","trainedFromCheckpointNoBNormStats","trainedFromCheckpointNoKBNormStats",\
# "trainedFromCheckpointNotLast2FC","trainedFromCheckpointNotLastFC","trainedFromCheckpointOnlyConvLayers","trainedFromScratch"]
# strategies = ["trainedFromCheckpointFullModel","trainedFromCheckpointNoBNormStats","trainedFromCheckpointNoKBNormStats",\
# "trainedFromCheckpointNotLast2FC","trainedFromCheckpointNotLastFC","trainedFromCheckpointOnlyConvLayers","trainedFromScratch"]
# strategies = ["trainedFromCheckpointFullModelTrainSetSize", "trainedFromCheckpointNotLastFCTrainSetSize", "trainedFromScratchTrainSetSize"]
# strategies = ["trainedFromScratch", "trainedFromCheckpointOnlyConvLayers", "trainedFromCheckpointFullModel"]
strategies = ["trainedFromCheckpointOnlyConvLayersTrainSetSize"]
# strategies = ["trainedFromScratchTrainSetSize", "trainedFromCheckpointOnlyConvLayersTrainSetSize", "trainedFromCheckpointFullModelTrainSetSize"]
subStrategies = ["/force_flow", "/forceRecon_flow"]
# trainSetSizes = ["500", "1000", "2000", "4000", "8000"]
trainSetSizes = ["500_2", "1000_2"]
# trainSetSize = "/500"
c1 = []
c2 = []
names = []
# re.split("[| |]| - ", a[0])
trainDirs = []
flow = []
flowSTD = []
flowTable = Table.Table(3, justs='lll', caption='Drag', label="tab::perTrainSetSizeFlow")
tD = strategies[0]
for j, trainSetSize in enumerate(trainSetSizes):
	c2.append(list([]))
	for tD in strategies:
		for stD in subStrategies:
			tDir = tD+"/"+trainSetSize+stD+"/vc/"
			# tDir = tD+stD+"/vc/"
			if not os.path.isdir(tDir):
				continue
			trainDirs.append(tDir)
			flow.append(0)
			flowSTD.append(0)
			runs = os.listdir(tDir)
			count = 0
			for r in runs:
				if "Release" not in r:
					continue
				path = tDir+r+"/testPerformance.log"
				try:
					f = open(path, "r")
				except FileNotFoundError:
					# print(path)
					continue
				count += 1
				res = f.read()
				res = res.split(" - ")
				try:
					flow[-1] += float(res[0])
				except ValueError:
					print(path)
					print(r)
					exit()
				# print(float(res[0][1:]), end="\t")
				flowSTD[-1] += float(res[0])**2
			if count == 0:
				continue
			flow[-1] /= count
			flowSTD[-1] /= count
			print("%100s" % tDir, end="\t")
			print(flow[-1])
			if j == 0:
				c1.append(tD+"/"+stD)
			c2[j].append(flow[-1])
			# print(flowSTD[-1], end="\t")
		print("", end="\n\n\n")

# for i, tD in enumerate(trainDirs):
# 	print("%100s" % tD, end="\t")
# 	print(drag[i], end="\t")
# 	print(dragSTD[i], end="\t")
# 	print(lift[i], end="\t")
# 	print(liftSTD[i])

tableFile = open("tableFile", "a")
flowTable.add_data([c1]+c2, sigfigs=4)
flowTable.print_table(tableFile)
tableFile.close()

# dragTable.add_data([c1]+c2, sigfigs=4)
# liftTable.add_data([c1]+c3, sigfigs=4)
# dragTable.print_table(tableFile)
# liftTable.print_table(tableFile)