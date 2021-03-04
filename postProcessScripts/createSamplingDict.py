import sys

#templatePath = "/tank/georgioutk/latopCleanUp/Documents/dataSetHelpers/extractSlicesNew/sampleDict"
templatePath = "/tank/car2d/postProcessScripts/sampleDict"
fileName = sys.argv[1]

res = 64
file = open(templatePath, "r")
cont = file.readlines()
file.close()

resolution = [128+64, 128, 1]
mmin = [0.5, -0.75, 0.15]
mmax = [3.5, 1.25, 0.15]
step = [float(mmax[0] - mmin[0])/resolution[0], float(mmax[1] - mmin[1])/resolution[1], float(mmax[2] - mmin[2])/resolution[2]]

points = ""
for i in range(resolution[0]):
	x = mmin[0]+step[0]*i
	for j in range(resolution[1]):
		y = mmin[1]+step[1]*j
		for k in range(resolution[2]):
			z = mmin[2]+step[2]*k
			points += "("+str(x)+" "+str(y)+" "+str(z)+")"

newCont = ""
for row in range(len(cont)):
	if "res" in cont[row]:
		newRow = cont[row][:-1]+"Airfoil\n"
		newCont += newRow
	elif "points" in cont[row]:
		newRow = cont[row][:-3]+points+");\n"
		newCont += newRow
	else:
		newCont += cont[row]

file = open(fileName, "w")
file.write(newCont)
file.close()
