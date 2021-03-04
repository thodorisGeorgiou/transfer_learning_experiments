import os
import sys
import multiprocessing

# modelTypes = ["op", "cc"]
# modelTypes = ["vc", "ds", "op", "cc"]
modelTypes = ["vc"]
numRuns = 4
# basePath = sys.argv[1]
# mType = sys.argv[2]
mType = "vc"
# if basePath[-1] != "/":
# 	exit("Path must end with a slash")
# gpu = sys.argv[1]
# releaseDirs = ["vc/1/","vc/2/","vc/3/","vc/4/"]

# numTrainExamples = ["500/forces/vc/", "1000/forces/vc/", "2000/forces/vc/", "4000/forces/vc/", "8000/forces/vc/"]
# numTrainExamples = ["500/forceFlow_forces/vc/", "1000/forceFlow_forces/vc/", "2000/forceFlow_forces/vc/", "4000/forceFlow_forces/vc/", "8000/forceFlow_forces/vc/"]
# numTrainExamples += ["500/force_forces/vc/", "1000/force_forces/vc/", "2000/force_forces/vc/", "4000/force_forces/vc/", "8000/force_forces/vc/"]
# numTrainExamples = ["500/flow_forces/vc/", "1000/flow_forces/vc/", "2000/flow_forces/vc/", "4000/flow_forces/vc/", "8000/flow_forces/vc/"]
# numTrainExamples += ["500/forceRecon_forces/vc/", "1000/forceRecon_forces/vc/", "2000/forceRecon_forces/vc/", "4000/forceRecon_forces/vc/", "8000/forceRecon_forces/vc/"]
# numTrainExamples += ["500/flowRecon_forces/vc/", "1000/flowRecon_forces/vc/", "2000/flowRecon_forces/vc/", "4000/flowRecon_forces/vc/", "8000/flowRecon_forces/vc/"]
# numTrainExamples += ["500/all_forces/vc/", "1000/all_forces/vc/", "2000/all_forces/vc/", "4000/all_forces/vc/", "8000/all_forces/vc/"]
# numTrainExamples = ["500_2/forces/vc/", "1000_2/forces/vc/"]
# numTrainExamples = ["500_2/force_forces/vc/", "1000_2/force_forces/vc/", "500_2/all_forces/vc/", "1000_2/all_forces/vc/", "500_2/forceFlow_forces/vc/", \
# "1000_2/forceFlow_forces/vc/", "500_2/forceRecon_forces/vc/", "1000_2/forceRecon_forces/vc/"]
# numTrainExamples = ["500_2/force_forces/vc/", "1000_2/force_forces/vc/", "500_2/all_forces/vc/", "1000_2/all_forces/vc/", "500_2/forceFlow_forces/vc/", \
# "1000_2/forceFlow_forces/vc/", "500_2/forceRecon_forces/vc/", "1000_2/forceRecon_forces/vc/", "500_2/flow_forces/vc/", "1000_2/flow_forces/vc/", \
# "500_2/flowRecon_forces/vc/", "1000_2/flowRecon_forces/vc/"]

# numTrainExamples = ["500/", "1000/", "2000/", "4000/", "8000/"]
numTrainExamples = ["500_2/", "1000_2/"]
# paths = ["trainedFromScratchTrainSetSize/", "trainedFromCheckpointFullModelTrainSetSize/", "trainedFromCheckpointOnlyConvLayersTrainSetSize/"]
paths = ["trainedFromCheckpointOnlyConvLayersTrainSetSize/"]
subPaths = ["force_flow/vc/", "forceRecon_flow/vc/"]
# subPaths = ["flow/vc/", "flow_flow/vc/", "force_flow/vc/", "flowRecon_flow/vc/", "forceRecon_flow/vc/", "forceFlow_flow/vc/", "all_flow/vc/"]
runs = ["1", "2", "3", "4"]

def runTest(relDir):
	# gpu = int(multiprocessing.current_process().name[-1]) - 1
	# run = str(gpu+1)
	# relDir = basePath+run+"Release/"
	if not os.path.isdir(relDir):
		print(relDir+"\nNot there :/")
		return
	# if gpu > 3:
	# 	exit("ID not dependable :(")
	os.system('python3 testNetworksOnFlow.py '+relDir+" "+mType)
	# os.system('CUDA_VISIBLE_DEVICES='+str(gpu)+' python3 testNetworksOnFlow.py '+relDir+" "+mType)

allDirs = [basePath+ntExamples+subPath+run+"Release/" for basePath in paths for ntExamples in numTrainExamples for subPath in subPaths for run in runs if os.path.isdir(basePath+ntExamples+subPath+run+"Release/")]
# allDirs = [basePath+run+"Release/" for run in runs]
p = multiprocessing.Pool(1)
res = p.map(runTest, allDirs)
p.close()
p.join()


# for mType in modelTypes:
# 	for run in range(numRuns):
# 		# relDir = basePath+mType+"/"+str(run+1)+"/"
# 		relDir = basePath+str(run+1)+"Release/"

# 		if not os.path.isdir(relDir):
# 			print(relDir)
# 			continue
# 		os.system('CUDA_VISIBLE_DEVICES='+gpu+' python3 testNetworks.py '+relDir+" "+mType)
# 		# os.system('python3 testNetworks.py '+relDir+" "+mType)