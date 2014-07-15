import numpy
import pyfits
import matplotlib.pyplot as plt
import scipy.integrate as i
import pyregion
import matplotlib.cm as cm
import math

info = 0

def centroid(center_xy, radius,dat):
	center_x, center_y = center_xy
	tempMax = 0
	tempMax2 = 0
	maxX = 0
	maxY = 0
	maxArray = []
	maxCoords = []
	for x in range(int(round(center_x - radius)), int(round(center_x + radius))):
		for y in range(int(round(center_y - radius)), int(round(center_y + radius))):
			if dat[y,x] > tempMax2:
				tempMax2 = dat[y,x]
				maxX = x
				maxY = y
		maxArray.append(tempMax2)
		maxCoords.append([maxX,maxY])
		tempMax2 = 0

	zipped = zip(maxArray, maxCoords)
	sortedZip = sorted(zipped, key=lambda zipped: zipped[0], reverse = True)
	maxArrayS, maxCoordsS = zip(*sortedZip)
	maximum = maxArrayS[0]
	newCenter = maxCoordsS[0]
	
	corners = [dat[center_y-radius,center_x-radius],dat[center_y-radius,center_x+radius],dat[center_y+radius, center_x-radius],dat[center_y+radius, center_x+radius]]
	minBack = min(corners)
	tempback = 0
	backcounter = 0
	for corner in corners:
		if corner < 1.5*sum(corners)/4.0:
			tempback += corner
			backcounter +=1
	background = tempback/backcounter
	
	for x in range(int(round(radius))):
		if (dat[newCenter[1],newCenter[0]+x] - background) > (maximum-background)/math.e**3:
			newRadius = x+1
	return newCenter, newRadius

def main():

	coma = pyfits.open('POSIIF_Coma.fits')
	head = coma[0].header
	dat = coma[0].data

	if info == 1:
		coma.info()
		print repr(head)
	
	comaRegName = "POSIIF_Coma.reg"
	comaReg = pyregion.open(comaRegName).as_imagecoord(head)

	for reg in comaReg:
		center_xy = reg.coord_list[0], reg.coord_list[1]
		radius = reg.coord_list[2]
		newCenter, newRadius = centroid(center_xy,radius,dat)
		reg.coord_list[0], reg.coord_list[1] = newCenter
		reg.coord_list[2] = newRadius
	
	maxval = numpy.amax(dat)
	ax=plt.subplot(111)
	ax.imshow(dat,cmap=cm.gray, vmin=0., vmax=maxval, origin="lower")
	patch_list, text_list = comaReg.get_mpl_patches_texts()
	for p in patch_list:
		ax.add_patch(p)
	for t in text_list:
		ax.add_artist(t)
	plt.show()
	
	coma.close()

main()
