import galaxyphot
import matplotlib.path
import matplotlib.patches
import numpy as np

amp = 4.0
xp, yp = 24.3, 37.1
ap, bp = 3.2, 1.8
thetap = 30.0

y, x = np.ogrid[1:51,1:51]  # pixel coords of pixel centers
image = galaxyphot.gaussian2d(x, y, amp, xp, yp, ap, bp, thetap)

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



x = np.arange(10, dtype=float) + 1
w = np.ones(len(x), dtype=float)
plt.plot(np.cumsum(w)-w[0], x, 'ko-')

w[2] = 0.5


f0 = np.array([[10, 15, 10], [15, 20, 15], [10, 15, 10]], dtype=float)
da0 = np.ones(f0.shape, dtype=float)  # Full pixels
I0 = f0 / da0

da = da0.copy()
da[(0, -1, 0, -1), (0, 0, -1, -1)] = 0.5  # Areas of pixels in aperture
f = I0 * da
I = f / da
assert np.all(I == I0)  # I = f0/da0 = f/da

f_tot = np.sum(f)
a_tot = np.sum(da)
f_mean = np.sum(da * f) / a_tot  # weighted
I_mean = f_tot / a_tot  # weighted


z = np.array([
    [22242, 19396, 16674, 14322, 11847],
    [22737, 20634, 17664, 14817, 12342],
    [23480, 21129, 17664, 14570, 12590],
    [23604, 21129, 17416, 14446, 12466],
    [22614, 20139, 16921, 14446, 11971]])



sum    error   area        surf_bri        surf_err
               (arcsec**2) (sum/arcsec**2) (sum/arcsec**2)
437260 661.256 25.3892     17222.3         26.0448

sum    npix mean    median min   max   var      stddev  rms
437260 25   17490.4 17416  11847 23604 1.52e+07 3898.71 17919.7



dec = np.radians(27.956443)

scale = 1.00871  # arcsec / pix
npix = 25
area = 25.3892  # scale**2 * npix?

sum_ = np.sum(z)
err = np.sqrt(sum_)
surf_bri = sum_ / area
surf_err = err / area

mean = np.mean(z)
median = np.median(z)
min_ = np.amin(z)
max_ = np.amax(z)
std = np.std(z)
var = std**2
rms = np.sqrt(np.mean(z**2))


intensity = z
area = np.ones(z.shape)


