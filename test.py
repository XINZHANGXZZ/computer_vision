import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import pickle
from matplotlib.pyplot import text
from skimage.measure import perimeter
from toolFunctions import extractImage, charToInt
import math

from toolFunctions import extractImage, intToChar, getbbimg

def extractFeature(name, showall, showbb, flag):

	(img, regions, ax, rthre, cthre) = extractImage(name, showall, showbb, flag)

	Features = []
	boxes = []

	for props in regions:
		tmp = []
		minr, minc, maxr, maxc = props.bbox
		if maxc - minc < cthre or maxr - minr < rthre or maxc - minc > cthre * 9 or maxr - minr > rthre * 9:
			continue
		tmp.append(minr)
		tmp.append(minc)
		tmp.append(maxr)
		tmp.append(maxc)
		boxes.append(tmp)
		if showbb == 1:
			ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
		# computing hu moments and removing small components
		roi = img[minr:maxr, minc:maxc]
		m = moments(roi)
		cr = m[0, 1] / m[0, 0]
		cc = m[1, 0] / m[0, 0]
		mu = moments_central(roi, cr, cc)
		nu = moments_normalized(mu)
		hu = moments_hu(nu)

		area = (maxr - minr)*(maxc - minc)
		# add convexity
		p = perimeter(img[minr:maxr, minc:maxc])
		con = (area / (p*p)) * 4 * math.pi
		convex = np.array([con])
		hu = np.concatenate((hu,convex))

		# add density
		den = area/float(props.convex_area)
		dense = np.array([den])	
		hu =   np.concatenate((hu,dense))

		Features.append(hu)

	# print boxes

	plt.title('Bounding Boxes') 
	if showbb == 1:
		io.show()

	return Features, boxes, 

# fun_test('test1',1)

def testing(file, mean, std, trainfeatures, labellist, showall, showbb, flag):
	testfeatures = []
	boxes = []

	(testfeatures,boxes) = extractFeature(file, showall, showbb, flag)

	# print len(testfeatures)
	# print mean, var, std

	for i in range(len(testfeatures)):
		for j in range(len(testfeatures[i])):
			testfeatures[i][j] = (testfeatures[i][j] - mean[j]) / float(std[j])

	# mean = np.mean(testfeatures)
	# std = np.std(testfeatures)
	# print mean, std, np.max(testfeatures)


	D = cdist(testfeatures, trainfeatures)
	# print D
	if showbb == 1:
		io.imshow(D) 
		plt.title('Distance Matrix') 
		io.show()

	D_index = np.argsort(D, axis=1)
	# print len(D_index)

	# knn
	if flag == 1:
		k = 5
	else:
		k = 1
	choice = np.zeros((len(D_index),k))
	for i in range(len(D_index)):
		for j in range(0,k):
			tmp = D_index[i][j]
			choice[i,j] = labellist[tmp]

	intresult = []
	
	for i in range(len(choice)):
		intchoice = []
		for j in range(len(choice[i])):
			intchoice.append(choice[i,j])
		counts = np.bincount(intchoice)
		intresult.append(np.argmax(counts))

	# for i in range(len(D_index)):
	# 	j =  D_index[i][0]
	# 	intresult.append(labellist[j])

	charresult = []

	for i in range(len(intresult)):
		charresult.append(intToChar(intresult[i]))

	# print intresult, charresult

	# print len(charresult), len(boxes)

	return charresult, boxes

def computeRate(file, testResult, boxes):

	pkl_file = open(file + '_gt.pkl', 'rb')
	mydict = pickle.load(pkl_file) 
	pkl_file.close()
	classes = mydict['classes'] 
	locations = mydict['locations']

	finaltest = []

	#judge if testresult is in boxes
	for i in range(len(locations)):
		for j in range(len(boxes)):
			if boxes[j][0] > locations[i][1] or boxes[j][1] > locations[i][0] or boxes[j][2] < locations[i][1] or boxes[j][3] < locations[i][0]:
				continue
			finaltest.append(testResult[j])

	right = 0

	for i in range(len(classes)):
		if finaltest[i] == classes[i]:
			right += 1

	rate = right / float(len(finaltest))
	print "recognition correct rate is:"
	print rate

	showDiff(file, rate, locations, finaltest)

def showDiff(file, rate, locations, finaltest):

	getbbimg(file)

	for i in range(len(locations)):
		text(locations[i][0]+20, locations[i][1]+20, finaltest[i], bbox=dict(facecolor='yellow', alpha=0.3))
	text(250, 2, rate, horizontalalignment='center', verticalalignment='center', bbox=dict(facecolor='yellow', alpha=0.5))
	plt.title('Recognition Result')
	plt.show()




