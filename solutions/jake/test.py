import galaxyphot
import matplotlib.path
import matplotlib.patches
import numpy as np

amp = 4.0
xp, yp = 24.3, 37.1
ap, bp = 3.2, 1.8
thetap = 30.0

y, x = np.ogrid[1:51,1:51]  # pixel coords of pixel centers
image = galaxyphot.util.gaussian2d(x, y, amp, xp, yp, ap, bp, thetap)

#dextent = (0.5, data.shape[1]+0.5, 0.5, data.shape[0]+0.5)
#plt.cla()
#plt.imshow(data, extent=dextent, **imshowkw)

x0, y0, a, b, theta = xp-1.1, yp+1.3, 2*ap-0.7, 2*bp+0.7, thetap+15
r = a
v0 = a*np.array([[1, 0], [0.5, 1], [-0.5, 1], [-1, 0], [-0.5, -1], [0.5, -1]]) + np.array([x0, y0])
cap = galaxyphot.CircularAperture((x0, y0), r)
eap = galaxyphot.EllipticalAperture((x0, y0), a, b, theta)
pap = galaxyphot.PolygonAperture(v0)


def testplot(self, arr, pix=True):
    imshowkw = dict(origin='lower', interpolation='nearest', cmap=plt.cm.gist_heat)
    patchkw = dict(ec='g', fc='none')
    extent = self.extent if pix else np.roll(self.ijextent, 2)
    plt.cla()
    plt.imshow(arr, extent=extent, **imshowkw)
    if isinstance(self, galaxyphot.CircularAperture):
        patch = matplotlib.patches.Circle
        xy0 = self.xy0 if pix else self.xy0 - 0.5
        args = (xy0,)
        patchkw['radius'] = self.r
        plt.plot(xy0[0], xy0[1], 'gx')
    elif isinstance(self, galaxyphot.EllipticalAperture):
        patch = matplotlib.patches.Ellipse
        xy0 = self.xy0 if pix else self.xy0 - 0.5
        args = (xy0, 2*self.a, 2*self.b)
        patchkw['angle'] = self.theta
        plt.plot(xy0[0], xy0[1], 'gx')
    elif isinstance(self, galaxyphot.PolygonAperture):
        patch = matplotlib.patches.PathPatch
        vertices = self.path.vertices.copy()
        if not pix: vertices = vertices - 0.5
        path = matplotlib.path.Path(vertices, closed=True)
        args = (self.path,)
    plt.gca().add_patch(patch(*args, **patchkw))
    plt.axis(extent)
    return


plt.figure(1); testplot(cap, cap.extract(image))
plt.figure(2); testplot(cap, cap.weights)

plt.figure(3); testplot(eap, eap.extract(image))
plt.figure(4); testplot(eap, eap.weights)

plt.figure(5); testplot(pap, pap.extract(image))
plt.figure(6); testplot(pap, pap.weights)

points = np.array([[21, 32], [22, 33], [25, 36], [28, 36], [28, 37], [28, 42]])
inside = np.array([False, True, True, False, True, False])
print np.all(pap.path.contains_points(points) == inside)


print galaxyphot.fit_gaussian(image, cap)
print galaxyphot.fit_gaussian(image, eap)
print galaxyphot.fit_gaussian(image, pap)

print galaxyphot.find_centroid(image, cap, mode='2dgauss', adjust=False)
print galaxyphot.find_centroid(image, eap, mode='2dgauss', adjust=False)
print galaxyphot.find_centroid(image, pap, mode='2dgauss', adjust=False)






"""

# a lot of design inspired by photutils; implementations differ
image = astropy.io.getdata(img_filename)
apertures = from_region_file(reg_filename)  # list of CircularAperture instances
annuli = []
for aperture in apertures:
    aperture.xy = find_centroid(image, aperture)
    annuli.append(CircularAnnulus(...))
apphot = measure_photometry(image, apertures, nsub=50)  # label, total (weighted), area (npix), median, std
anphot = measure_photometry(image, annuli, nsub=50)
total = apphot['total']
background = anphot['median'] * apphot['area']
signal = total - background
noise = np.sqrt(total + background)

"""

aperture_list = (cap, eap, pap)

for aperture in aperture_list:
    galaxyphot.find_centroid(image, aperture, mode='2dgauss')

table = galaxyphot.apphot(image, aperture_list)



