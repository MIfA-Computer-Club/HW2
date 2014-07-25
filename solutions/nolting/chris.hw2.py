import numpy
import pyfits
import matplotlib.pyplot as plt
import scipy.integrate as i
import pyregion
import matplotlib.cm as cm
import math

info = 0

def centroid_bright(center_xy, radius,dat):
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
	
	return newCenter

def getBackground(center_xy, radius, dat):
	center_x, center_y = center_xy
	grid = dat[center_y-radius:center_y+radius,center_x-radius:center_x+radius]
	xCoord, yCoord = numpy.mgrid[center_x - radius:center_x + radius, center_y - radius: center_y+radius]
	radArray = ((xCoord-center_x)**2+(yCoord - center_y)**2)**(0.5)
	bgGrid = dat[center_y-1.3*radius:center_y+1.3*radius,center_x-1.3*radius:center_x+1.3*radius]
	backgroundAnnulusMask = numpy.logical_and(radArray>radius*1.1, radArray<radius*1.2)
	backgroundX, backgroundY = backgroundAnnulusMask.nonzero()
	background = numpy.median(bgGrid[backgroundY,backgroundX])
	
	return background

def getRadius(center_xy, radius,dat):
	center_x, center_y = center_xy
	centerVal = dat[center_y,center_x]
		
	grid = dat[center_y-radius:center_y+radius,center_x-radius:center_x+radius]
	xCoord, yCoord = numpy.mgrid[center_x - radius:center_x + radius, center_y - radius: center_y+radius]
	radArray = ((xCoord-center_x)**2+(yCoord - center_y)**2)**(0.5)
	
	background = getBackground(center_xy, radius, dat)

	for x in range(int(round(radius))):
		mask = numpy.logical_and(radArray<=x, radArray>x-1) ## find all within this annulus
		maskX, maskY = mask.nonzero()
		annulusAverage = 0
		if(len(grid[maskY,maskX])!=0):
			annulusAverage = sum(grid[maskY,maskX])/len(grid[maskY,maskX])
		if((annulusAverage - background) > (centerVal - background)/math.e**3):
			newRadius = x+1

	return newRadius

def makeAp(center_xy, radius, dat):
	center_x, center_y = center_xy
	grid = dat[center_y-radius:center_y+radius,center_x-radius:center_x+radius]
	xCoord, yCoord = numpy.mgrid[center_x - radius:center_x + radius, center_y - radius: center_y+radius]
	radArray = ((xCoord-center_x)**2+(yCoord - center_y)**2)**(0.5)
	mask = radArray <= radius
	maskX, maskY = mask.nonzero()
	aperature = grid[maskY, maskX]
	return aperature

def centroid_weighted(center_xy, radius,dat):
	center_x, center_y = center_xy
	sumWeightedX = 0
	sumWeightedY = 0
	sumWeights = 0
	
	background = getBackground(center_xy, radius, dat)
	
	for x in range(int(round(center_x - radius)), int(round(center_x + radius))):
		for y in range(int(round(center_y - radius)), int(round(center_y + radius))):
			weight = dat[y,x] - background
			sumWeightedX += weight * x
			sumWeightedY += weight * y
			sumWeights += weight
	newCenter = sumWeightedX/sumWeights, sumWeightedY/sumWeights
	return newCenter
	
## calculate the flux per pixel
def getFlux(center, radius, dat):
	center_x, center_y = center
	background = getBackground(center,radius,dat)
	flux = 0
	fluxValues = makeAp(center, radius, dat) - background
	flux = sum(fluxValues)/len(fluxValues)
	return flux
	
def makeProfile(center_xy, radius, dat):
	center_x, center_y = center_xy
	centerVal = dat[center_y,center_x]
	
	grid = dat[center_y-radius:center_y+radius,center_x-radius:center_x+radius]
	xCoord, yCoord = numpy.mgrid[center_x - radius:center_x + radius, center_y - radius: center_y+radius]
	radArray = ((xCoord-center_x)**2+(yCoord - center_y)**2)**(0.5)
	background = getBackground(center_xy, radius, dat)
	
	annulusAverage = numpy.zeros(int(round(radius)))
	error = numpy.zeros(int(round(radius)))
	radiusArray = numpy.zeros(int(round(radius)))
	
	for x in range(int(round(radius))):
		mask = numpy.logical_and(radArray<=x, radArray>x-1) ## find all within this annulus
		maskX, maskY = mask.nonzero()
		if(len(grid[maskY,maskX])!=0):
			annulusAverage[x] = sum(grid[maskY,maskX])/len(grid[maskY,maskX])
		error[x] = numpy.std(grid[maskY,maskX])
		radiusArray[x] = x
	return annulusAverage,error, radiusArray
	
def main():
	
	galNames = numpy.genfromtxt('galaxy_names.txt', dtype='S10')
	coma = pyfits.open('POSIIF_Coma.fits')
	head = coma[0].header
	dat = coma[0].data
	
	scale = head['PLTSCALE'] ##arcsec per mm
	sizeX = head['PLTSIZEX'] ##mm in x
	pixX = head['NAXIS1'] ##number of pixels in x
	sizeY = head['PLTSIZEY'] ##mm in y
	pixY = head['NAXIS2'] ##number of pixels in y
	pix2Permm2 = pixX*pixY/(sizeX*sizeY)
	mm2PerArcsec2 = sizeX*sizeY/(scale*scale)
	
	if info == 1:
		coma.info()
		print repr(head)
	
	comaRegName = "POSIIF_Coma.reg"
	comaReg = pyregion.open(comaRegName)#.as_imagecoord(head)

	for i, reg in enumerate(comaReg):
		center_xy = reg.coord_list[0], reg.coord_list[1]
		radius = reg.coord_list[2]
		name = galNames[i]
		brightCenter = centroid_bright(center_xy,radius,dat)
		newCenter = centroid_weighted(center_xy,radius,dat)
		newRadius = getRadius(newCenter, radius, dat)
		reg.coord_list[0], reg.coord_list[1] = newCenter
		reg.coord_list[2] = newRadius
		flux = getFlux(newCenter, newRadius, dat)*pix2Permm2*mm2PerArcsec2
		print 'flux is ', ("{:.2e}".format(flux)), ' centered on ', center_xy
		profile, err, r = makeProfile(newCenter, newRadius, dat)
		profile = profile * pix2Permm2*mm2PerArcsec2
		err = err * pix2Permm2*mm2PerArcsec2
		r = r*pixX/sizeX*scale
		plt.errorbar(r,profile,err)
		plt.xlabel('radius (in arcsec)')
		plt.ylabel('counts/arcsec^2')
		plt.show()
		plt.savefig(name,format='png')
		for rr, rVal in enumerate(r):
			maxFlux = max(profile)
			if profile[rr] <= 0.5*maxFlux:
				halfLight = rVal
				break
			halfLight = max(r)
		print halfLight
				
	plt.clf
	maxval = numpy.amax(dat)
	ax=plt.subplot(111)
	ax.imshow(dat,cmap=cm.gray, vmin=0., vmax=maxval, origin="lower")
	patch_list, text_list = comaReg.get_mpl_patches_texts()
	for p in patch_list:
		ax.add_patch(p)
	for t in text_list:
		ax.add_artist(t)

	plt.show()
	plt.savefig('ComaRegion',format='png')
	
	coma.close()

main()
