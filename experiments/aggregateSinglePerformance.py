import sys
import os
import re
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
strategies = ["trainedFromScratchTrainSetSize", "trainedFromCheckpointOnlyConvLayersTrainSetSize", "trainedFromCheckpointFullModelTrainSetSize"]
subStrategies = ["/all_forces", "/flowRecon_forces", "/forceFlow_forces", "/force_forces", "/forceRecon_forces", "/flow_forces", "/forces"]
# trainSetSizes = ["500", "1000", "2000", "4000", "8000"]
trainSetSizes = ["500", "500_2", "1000", "1000_2", "2000", "4000", "8000"]
# trainSetSizes = ["500_2", "1000_2"]
# trainSetSize = "/500"
c1 = []
c2 = []
c3 = []
names = []
# re.split("[| |]| - ", a[0])
trainDirs = []
drag = []
lift = []
dragSTD = []
liftSTD = []
tD = strategies[0]
dragTable = Table.Table(8, justs='llllllll', caption='Drag', label="tab::perTrainSetSizeDrag")
dragTable.add_header_row(['']+trainSetSizes)
for j, trainSetSize in enumerate(trainSetSizes):
	c2.append(list([]))
	for tD in strategies:
		for stD in subStrategies:
			tDir = tD+"/"+trainSetSize+stD+"/vc/"
			# tDir = tD+stD+"/vc/"
			if not os.path.isdir(tDir):
				continue
			trainDirs.append(tDir)
			drag.append(0)
			dragSTD.append(0)
			runs = os.listdir(tDir)
			count = 0
			for r in runs:
				if "Release" not in r:
					continue
				path = tDir+r+"/performance.log"
				try:
					f = open(path, "r")
				except FileNotFoundError:
					# print(path)
					continue
				res = f.read()
				res = re.split("[| |]| - ", res)
				try:
					drag[-1] += float(res[0][1:])
				except ValueError:
					print(path)
					print(r)
					exit()
				# print(float(res[0][1:]), end="\t")
				dragSTD[-1] += float(res[0][1:])**2
				count += 1
				i = 1
				while True:
					if res[i] == '':
						i += 1
					else:
						drag[-1] += float(res[i][:-1])
						count += 1
						dragSTD[-1] += float(res[i][:-1])**2
						# print(float(res[i][:-1]))
						break
			if count == 0:
				continue
			print(count)
			drag[-1] /= count
			dragSTD[-1] /= count
			if j == 0:
				c1.append(tD+"/"+stD)
			c2[j].append(drag[-1])
		# 	print("%100s" % tDir, end="\t")
		# 	print(drag[-1], end="\t")
		# 	print(dragSTD[-1], end="\t")
		# 	print(lift[-1], end="\t")
		# 	print(liftSTD[-1])
		# print("", end="\n\n\n")

# for i, tD in enumerate(trainDirs):
# 	print("%100s" % tD, end="\t")
# 	print(drag[i], end="\t")
# 	print(dragSTD[i], end="\t")
# 	print(lift[i], end="\t")
# 	print(liftSTD[i])

dragTable.add_data([c1]+c2, sigfigs=4)
tableFile = open("tableFile", "a")
dragTable.print_table(tableFile)
tableFile.close()