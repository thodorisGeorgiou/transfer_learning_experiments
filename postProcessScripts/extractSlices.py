import os
import numpy
import pickle
import subprocess
from multiprocessing import Pool

def parseFile(filePath, l):
	file = open(filePath)
	cont = file.readlines()
	file.close()
	newCont = numpy.zeros((len(cont)-3, l), dtype=numpy.float32)
	ind = 0
	nLines = len(cont)
	if l == 4:
		for row in cont[3:]:
			t = row.split(" \t")
			try:
				newCont[ind][3] = float(t[3])
				newCont[ind][0] = float(t[0])
				newCont[ind][1] = float(t[1])
				newCont[ind][2] = float(t[2])
			except IndexError:
				if t[0] == 'e\n':
					newCont[ind][0] = 999999
				else:
					print("Something is wrong..")
					sys.exit(0)
			ind += 1
	elif l == 6:
		for row in cont[3:]:
			t = row.split(" \t")
			try:
				newCont[ind][5] = float(t[5])
				newCont[ind][0] = float(t[0])
				newCont[ind][1] = float(t[1])
				newCont[ind][2] = float(t[2])
				newCont[ind][3] = float(t[3])
				newCont[ind][4] = float(t[4])
			except IndexError:
				if t[0] == 'e\n':
					newCont[ind][0] = 999999
				else:
					print("Something is wrong..")
					sys.exit(0)
			ind += 1
	else:
		print("Something is wrong 2...")
		sys.exit(0)
	del cont
	return newCont

def createIndexMap(newCont):
	indMap = [{}, {}]
	end = numpy.where(newCont == 999999)[0][0]
	xs = newCont[:end, 0].tolist()
	ys = newCont[:end, 1].tolist()
	# zs = newCont[:end, 2].tolist()
	xs = list(set(xs))
	ys = list(set(ys))
	# zs = list(set(zs))
	xs.sort()
	ys.sort()
	# zs.sort()
	c = 0
	for x in xs:
		indMap[0][str(x)] = c
		c+= 1
	c = 0
	for y in ys:
		indMap[1][str(y)] = c
		c+= 1
	# c = 0
	# for z in zs:
	# 	indMap[1][str(z)] = c
	# 	c+= 1
	return indMap	

def changeFormat_p_nut_k(oldFormat, indMap, path):
	filePrefix = path
	files=["/p.raw", "/nut.raw", "/k.raw"]
	ind = 0
	p = numpy.zeros((len(indMap[0].keys()), len(indMap[1].keys())), dtype=numpy.float32)
	data = []
	for row in range(len(oldFormat)):
		if oldFormat[row][0] == 999999:
			p.tofile(filePrefix+files[ind])
			data.append(p)
			ind+=1
			p = numpy.zeros((len(indMap[0].keys()), len(indMap[1].keys())), dtype=numpy.float32)
			continue
		try:
			p[indMap[0][str(float(oldFormat[row][0]))]][indMap[1][str(float(oldFormat[row][1]))]] = oldFormat[row][3]
		except KeyError:
			print("row = "+str(row))
			sys.exit(0)
	return numpy.stack(data, axis=-1)

def changeFormat_U(oldFormat, indMap, path):
	fileName = path+"/u.raw"
	u = numpy.zeros((len(indMap[0].keys()), len(indMap[1].keys()), 3), dtype=numpy.float32)
	for row in range(len(oldFormat)):
		if oldFormat[row][0] == 999999:
			continue
		try:
			ind0 = indMap[0][str(float(oldFormat[row][0]))]
			ind1 = indMap[1][str(float(oldFormat[row][1]))]
			u[ind0][ind1][0] = oldFormat[row][3]
			u[ind0][ind1][1] = oldFormat[row][4]
			u[ind0][ind1][2] = oldFormat[row][5]
		except KeyError:
			print("row = "+str(row))
			sys.exit(0)
	u.tofile(fileName)
	return u

def changeFormatForCase():
	postProccessPath = "postProcessing/sampleDict/2000"
	filePath = postProccessPath+"/res_p_nut_k.gplt"
	cont = parseFile(filePath, 4)
	indMap = createIndexMap(cont)
	pNutK = changeFormat_p_nut_k(cont, indMap, postProccessPath)
	filePath = postProccessPath+"/res_U.gplt"
	cont = parseFile(filePath, 6)
	u = changeFormat_U(cont, indMap, postProccessPath)
	allInOne = numpy.concatenate([u,pNutK], axis=-1)
	allInOne.tofile(postProccessPath+"/allInOne.raw")

def runPostProcessing(c):
	os.chdir(c)
	if os.path.isfile("postProcessing/sampleDict/2000/allInOne.raw"):
		return
	if not os.path.isdir("postProcessing/sampleDict/2000"):
		sampleLog = open("LOG-sample", "w")
		subprocess.call("cp /tank/car2d/postProcessScripts/sampleDictCar2d system/sampleDict", shell=True)
		subprocess.call("postProcess -func sampleDict", shell=True, stdout=sampleLog, stderr=sampleLog)
	try:
		changeFormatForCase()
	except FileNotFoundError:
		print(c+" did not run..")
		return
	os.chdir(c)
	doneFile = open("doneSample", "w")
	doneFile.close()

mainDir = "/tank/car2d/simulations/"
validSims = numpy.load("validDone.npy")
# validSims = numpy.load("validSims.npy")
cases = [mainDir+str(sim) for sim in validSims]
print(str(validSims.shape) + " valid simulations")
p = Pool(88)
res = p.map(runPostProcessing, cases)
p.close()
p.join()