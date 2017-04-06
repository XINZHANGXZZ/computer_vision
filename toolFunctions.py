import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, filters
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import pickle
from skimage.morphology import binary_dilation, binary_erosion, closing

# from skimage.morphology import convex_hull_image

def charToInt(ch):
	if ch == 'a': return 1
	if ch == 'd': return 2
	if ch == 'f': return 3
	if ch == 'h': return 4
	if ch == 'k': return 5
	if ch == 'm': return 6
	if ch == 'n': return 7
	if ch == 'o': return 8
	if ch == 'p': return 9
	if ch == 'q': return 10
	if ch == 'r': return 11
	if ch == 's': return 12
	if ch == 'u': return 13
	if ch == 'w': return 14
	if ch == 'x': return 15
	if ch == 'z': return 16

def intToChar(ch):
	if ch == 1: return 'a'
	if ch == 2: return 'd'
	if ch == 3: return 'f'
	if ch == 4: return 'h'
	if ch == 5: return 'k'
	if ch == 6: return 'm'
	if ch == 7: return 'n'
	if ch == 8: return 'o'
	if ch == 9: return 'p'
	if ch == 10: return 'q'
	if ch == 11: return 'r'
	if ch == 12: return 's'
	if ch == 13: return 'u'
	if ch == 14: return 'w'
	if ch == 15: return 'x'
	if ch == 16: return 'z'

def extractImage(name, showall, showbb, flag):
	img = io.imread(name + '.bmp')
	#print img.shape

	if showall == 1:
		io.imshow(img)
		plt.title('Original Image')	
		io.show()
	
	hist = exposure.histogram(img)
	if showall == 1:
		plt.bar(hist[1], hist[0])
		plt.title('Histogram')
		plt.show()

	if flag == 1:
		th = filters.threshold_otsu(img)
	else:
		th = 200
	# th = filters.threshold_isodata(img)
	# print help(filters.thresholding)

	img_binary = (img < th).astype(np.double)
	if showall == 1:
		io.imshow(img_binary)
		plt.title('Binary Image')	
		io.show()

	dilH = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
	img_dil = binary_dilation(img_binary, dilH)

	if showall == 1:
		io.imshow(img_dil)
		plt.title('dilation Image')
		io.show()

	eroH = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
	img_ero = binary_erosion(img_dil)
	if showall == 1:
		io.imshow(img_ero)
		plt.title('erosion Image')
		io.show()

	img_label = label(img_ero, background = 0)
	if showall == 1:
		io.imshow(img_label)
		plt.title('Labeled Image')
		io.show()

	# display component bounding boxes
	regions = regionprops(img_label) 
	if showbb == 1:
		io.imshow(img_ero)
	ax = plt.gca()
	
	# chull = convex_hull_image(img_binary)
	# io.imshow(chull)
	# io.show()

	row = 0
	col = 0
 
	for props in regions:
		minr, minc, maxr, maxc = props.bbox
		row += maxr - minr
		col += maxc - minc

	#print ndarray.ndim, ndarray.size

	rthre = row / (len(regions) * float(3))
	cthre = col / (len(regions) * float(3))

	return img_binary, regions, ax, rthre, cthre

def normalize(array):
	mean = np.mean(array)
	var = np.var(array)
	std = np.std(array)

	for i in range(len(array)):
		array[i] = (array[i] - mean) / float(std)

	return array

def getbbimg(file):
	# add text
	img = io.imread(file + '.bmp')
	# io.imshow(img)
	th = 200

	img_binary = (img < th).astype(np.double)
	img_label = label(img_binary, background = 0)
	regions = regionprops(img_label) 
	io.imshow(img_binary)
	ax = plt.gca()

	row = 0
	col = 0
 
	for props in regions:
		minr, minc, maxr, maxc = props.bbox
		row += maxr - minr
		col += maxc - minc

	#print ndarray.ndim, ndarray.size

	rthre = row / (len(regions) * float(3))
	cthre = col / (len(regions) * float(3))

	for props in regions:
		minr, minc, maxr, maxc = props.bbox
		if maxc - minc < cthre or maxr - minr < rthre or maxc - minc > cthre * 9 or maxr - minr > rthre * 9:
			continue
		ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
		




