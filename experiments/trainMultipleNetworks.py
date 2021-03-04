import sys
import os
from multiprocessing import Pool


def trainNets(inpt):
	gpu = inpt % 4
	run = inpt
	modelTypes = ["vc"]
	# modelTypes = ['vc', "ds", "op", "cc"]
	# if run > 12:
	# 	modelTypes = ['vc', "op", "cc"]
	# else:
	# 	modelTypes = ["op", "cc"]
	modelType = "vc"
	# numTrainExamplesList = ["500", "1000", "2000", "4000", "8000"]
	numTrainExamplesList = ["500", "1000"]
	# preTrainSupervisionModes = ["Force", "Flow", "FlowRecon", "ForceFlow", "ForceRecon", "All"]
	preTrainSupervisionModes = ["Force", "ForceRecon"]
	# preTrainSupervisionModes = ["Flow", "FlowRecon", "ForceFlow", "All"]
	# os.system("CUDA_VISIBLE_DEVICES="+str(gpu)+" python3 trainOnMultipleGPUsStartFromCheckpointOnlyConv.py "+modelType+" "+str(run)+\
	# 	" /tank/airfoil/experiments/netowrksTrainOnFlow/2x512_3x128_2x64SkipPytorchBNormEpsilonvc/"+str(run)+"Release/")
	# os.system("CUDA_VISIBLE_DEVICES="+str(gpu)+" python3 trainOnFlows.py "+modelType+" "+str(run))
	# for preTrainSupervisionMode in preTrainSupervisionModes:
	# 	argument = preTrainSupervisionMode[0].lower()+preTrainSupervisionMode[1:]
	# 	if gpu==0:
	# 		print("Doing "+argument)
	# 	if preTrainSupervisionMode != "Force":
	# 		os.system("CUDA_VISIBLE_DEVICES="+str(gpu)+" python3 trainOnFlowsStartFromCheckpoint.py "+modelType+" "+str(run)+\
	# 			" /tank/airfoil/experiments/netowrksTrainOn"+preTrainSupervisionMode+"/2x512_3x128_2x64SkipPytorchBNormEpsilonvc/"\
	# 			+str(run)+"Release/ "+argument)
	# 	else:
	# 		os.system("CUDA_VISIBLE_DEVICES="+str(gpu)+" python3 trainOnFlowsStartFromCheckpoint.py "+modelType+" "+str(run)+\
	# 			" /tank/airfoil/experiments/netowrkswithValSet/2x512_3x128_2x64SkipPytorchBNormEpsilonvc/"\
	# 			+str(run)+"Release/ "+argument)
	for numTrainExamples in numTrainExamplesList:
		if gpu==0:
			print("Num Train Examples: "+str(numTrainExamples))
		# os.system("CUDA_VISIBLE_DEVICES="+str(gpu)+" python3 trainOnMultipleGPUs.py "+modelType+" "+str(run)+" "+numTrainExamples)
		# os.system("CUDA_VISIBLE_DEVICES="+str(gpu)+" python3 trainOnFlows.py "+modelType+" "+str(run)+" "+numTrainExamples)
		for preTrainSupervisionMode in preTrainSupervisionModes:
			argument = preTrainSupervisionMode[0].lower()+preTrainSupervisionMode[1:]
			if preTrainSupervisionMode != "Force":
				os.system("CUDA_VISIBLE_DEVICES="+str(gpu)+" python3 trainOnFlowsStartFromCheckpoint.py "+modelType+" "+str(run)+\
					" "+str(numTrainExamples)+" /tank/airfoil/experiments/netowrksTrainOn"+preTrainSupervisionMode+"/2x512_3x128_2x64SkipPytorchBNormEpsilonvc/"\
					+str(run)+"Release/ "+argument)
			else:
				os.system("CUDA_VISIBLE_DEVICES="+str(gpu)+" python3 trainOnFlowsStartFromCheckpoint.py "+modelType+" "+str(run)+\
					" "+str(numTrainExamples)+" /tank/airfoil/experiments/netowrkswithValSet/2x512_3x128_2x64SkipPytorchBNormEpsilonvc/"\
					+str(run)+"Release/ "+argument)

for s in range(1):
	print("Doing s="+str(s))
	runs = [i for i in range(s*4+1,s*4+5)]
	p = Pool(4)
	res = p.map(trainNets, runs)
	p.close()
	p.join()

print("Done!")
