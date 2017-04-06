import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, filters
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import pickle
from skimage.morphology import convex_hull_image
from skimage.measure import perimeter
from toolFunctions import extractImage, charToInt
import math


def extractFeature(name, showall, showbb, flag):

	(img, regions, ax, rthre, cthre) = extractImage(name, showall, showbb, flag)

	Features = []
	lab = []

	for props in regions:
		minr, minc, maxr, maxc = props.bbox
		if maxc - minc < cthre or maxr - minr < rthre or maxc - minc > cthre * 9 or maxr - minr > rthre * 9:
			continue
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
		hu = np.concatenate((hu,dense))
		
		Features.append(hu)

		lab.append(charToInt(name))



	plt.title('Bounding Boxes') 
	if showbb == 1:
		io.show()

	#print len(Features)
	return Features, lab

# extractFeature('z', 1, 1)

def training(showall, showbb, flag):
	files = ['a', 'd', 'f', 'h', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'w', 'x', 'z']
	featureslist = []
	labellist = []

	for f in files:
		(fl, ll) = extractFeature(f, showall, showbb, flag)
		featureslist += fl
		labellist += ll

	# for fe in range(len(featureslist))
	# 	label.append(charToInt(f))

	# print len(featureslist), len(labellist)

	#print featureslist
	mean = []
	std = []

	for i in range(len(featureslist[0])):
		mean.append(np.mean(featureslist[i]))
		std.append(np.std(featureslist[i]))

	# print mean, std

	# mean = np.mean(featureslist)
	# var = np.var(featureslist)
	# std = np.std(featureslist)

	# print mean, var, std, np.max(featureslist)

	for i in range(len(featureslist)):
		for j in range(len(featureslist[i])):
			featureslist[i][j] = (featureslist[i][j] - mean[j]) / float(std[j])

	# mean = np.mean(featureslist)
	# var = np.var(featureslist)
	# std = np.std(featureslist)

	# print mean, var, std, np.max(featureslist)

	# print "-------------------"

	# newmean = []
	# newstd = []
	# for i in range(len(featureslist[0])):
	# 	newmean.append(np.mean(featureslist[i]))
	# 	newstd.append(np.std(featureslist[i]))

	# print newmean, newstd

	D = cdist(featureslist, featureslist)
	# print D
	# io.imshow(D) 
	# plt.title('Distance Matrix') 
	# io.show()

	D_index = np.argsort(D, axis=1)
	# print D_index

	# io.imshow(confM) 
	# plt.title('Confusion Matrix') 
	# io.show()

	return (mean, std, featureslist, labellist)

# training()
# print mean, var, std
