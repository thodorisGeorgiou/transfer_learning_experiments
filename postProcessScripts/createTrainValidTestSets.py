import os
import numpy

validSims = numpy.load("validDone.npy")
numTestExamples = 5000
numValidationExamples = 2000
numTrainExamples = validSims.shape[0] - numTestExamples - numValidationExamples

# randInds = numpy.arange(validSims.shape[0])
# numpy.random.shuffle(randInds)
# numpy.save("shuffleCaseIndices", randInds)

randInds = numpy.load("shuffleCaseIndices.npy")
testCases = validSims[randInds[:numTestExamples]]
validationCases = validSims[randInds[numTestExamples:(numTestExamples+numValidationExamples)]]
trainCases = validSims[randInds[(numTestExamples+numValidationExamples):]]

for i, t in enumerate(testCases):
	os.system("cp /tank/car2d/simulations/"+str(t)+"/postProcessing/sampleDict/2000/allInOne.raw /tank/car2d/sets/testSet/"+str(i)+".raw")

for i, t in enumerate(validationCases):
	os.system("cp /tank/car2d/simulations/"+str(t)+"/postProcessing/sampleDict/2000/allInOne.raw /tank/car2d/sets/validationSet/"+str(i)+".raw")

for i, t in enumerate(trainCases):
	os.system("cp /tank/car2d/simulations/"+str(t)+"/postProcessing/sampleDict/2000/allInOne.raw /tank/car2d/sets/trainSet/"+str(i)+".raw")


testGroundTruth = numpy.zeros([numTestExamples, 2])
for i, c in enumerate(testCases):
	forces = numpy.loadtxt("/tank/car2d/simulations/"+str(c)+"/postProcessing/forceCoeffs1/0/forceCoeffs.dat", dtype=numpy.float32)
	testGroundTruth[i] = forces[-1][2:4]

validationGroundTruth = numpy.zeros([numValidationExamples, 2])
for i, c in enumerate(validationCases):
	forces = numpy.loadtxt("/tank/car2d/simulations/"+str(c)+"/postProcessing/forceCoeffs1/0/forceCoeffs.dat", dtype=numpy.float32)
	validationGroundTruth[i] = forces[-1][2:4]

trainGroundTruth = numpy.zeros([numTrainExamples, 2])
for i, c in enumerate(trainCases):
	forces = numpy.loadtxt("/tank/car2d/simulations/"+str(c)+"/postProcessing/forceCoeffs1/0/forceCoeffs.dat", dtype=numpy.float32)
	trainGroundTruth[i] = forces[-1][2:4]


numpy.save("/tank/car2d/sets/trainGroundTruth", trainGroundTruth)
numpy.save("/tank/car2d/sets/testGroundTruth", testGroundTruth)
numpy.save("/tank/car2d/sets/validationGroundTruth", validationGroundTruth)
