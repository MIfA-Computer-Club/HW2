import galaxyphot
import numpy as np

imshowkw = dict(origin='lower', interpolation='nearest', cmap=plt.cm.gist_heat)

amp = 4.0
xp, yp = 24.3, 37.1
ap, bp = 3.2, 1.8
thetap = 30.0

y, x = np.ogrid[1:51,1:51]  # pixel coords of pixel centers
data = galaxyphot.gaussian2d(x, y, amp, xp, yp, ap, bp, thetap)

dextent = (0.5, data.shape[1]+0.5, 0.5, data.shape[0]+0.5)
#plt.cla()
#plt.imshow(data, extent=dextent, **imshowkw)

x0, y0, r = xp-1.1, yp+1.3, ap*2
self = galaxyphot.CircularAperture((x0, y0), r, label='myap')
#x0, y0, a, b, theta = xp-1.1, yp+1.3, ap-0.7, bp+0.7, thetap+15
#self = galaxyphot.EllipticalAperture((x0, y0), a, b, theta, image=data)

plt.cla()
extent = np.roll(self.ijextent, 2)
plt.imshow(self.weights, extent=extent, **imshowkw)
#patch = matplotlib.patches.Ellipse((self.j, self.i), 2*self.a, 2*self.b, angle=self.theta, ec='g', fc='none')
patch = matplotlib.patches.Circle((self.j, self.i), radius=self.r, ec='g', fc='none')
plt.gca().add_patch(patch)
plt.plot(self.j, self.i, 'gx')
plt.axis(extent)

self.centroid(mode='2dgauss', adjust=True)
plt.figure(2)
extent = np.roll(self.ijextent, 2)
plt.imshow(self.section, extent=extent, **imshowkw)
patch2 = matplotlib.patches.Ellipse((self.j, self.i), 2*self.a, 2*self.b, angle=self.theta, ec='g', fc='none')
plt.gca().add_patch(patch2)
#jj = np.linspace(self.jmin, self.jmax, 100)
#ii1 = np.sqrt(self.r**2 - (jj-self.j)**2) + self.i
#ii2 = -np.sqrt(self.r**2 - (jj-self.j)**2) + self.i
#plt.plot(np.hstack((jj, jj[::-1])), np.hstack((ii1, ii2[::-1])), 'g-')
plt.plot(self.j, self.i, 'gx')
plt.axis(extent)

res = self.fit_gaussian()
x1, y1, a1, b1, theta1 = res[1:]
self.xy = x1, y1
self.a, self.b = 2*a1, 2*b1
self.theta = theta1
plt.figure(3)
extent = np.roll(self.ijextent, 2)
plt.imshow(self.section, extent=extent, **imshowkw)
patch3 = matplotlib.patches.Ellipse((self.j, self.i), 2*self.a, 2*self.b, angle=self.theta, ec='g', fc='none')
plt.gca().add_patch(patch3)
plt.plot(self.j, self.i, 'gx')
plt.axis(extent)



class Base(object):

    """Everything in this class is derived from `extent`.

    """

    def __init__(self, _range):
        self._range = _range

    @property
    def extent(self):
        """The extent property."""
        return self._range

    @property
    def xmin(self):
        """The xmin property."""
        return self._range[0]

class Child(Base):

    def __init__(self, xy, a, b):
        self._params = [xy, a, b]
        super(Child, self).__init__(self._calc_range())

    @property
    def params(self):
        """Central container accessed by all parameter properties."""
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

        # Setter instructions to be run when any parameter is updated.
        # Calculate range for this child class and update.
        self._range = self._calc_range()

    @property
    def xy(self):
        return self._params[0]

    @xy.setter
    def xy(self, xy):
        self._params[0] = xy
        self.params = self._params

    @property
    def a(self):
        return self._params[1]

    @a.setter
    def a(self, a):
        self._params[1] = a
        self.params = self._params

    @property
    def b(self):
        return self._params[2]

    @b.setter
    def b(self, b):
        self._params[2] = b
        self.params = self._params

    @property
    def r(self):
        """The r property."""
        return np.sqrt(self.a**2 + self.b**2)

    def _calc_range(self):
        xy, a, b = self.params
        return (xy[0]-a, xy[0]+a, xy[1]-b, xy[1]+b)






def ay(x, x0, y0, a, b, theta):
    theta = theta * np.pi/180
    return np.sin(theta)**2 / a**2 + np.cos(theta)**2 / b**2

def by(x, x0, y0, a, b, theta):
    theta = theta * np.pi/180
    return 2 * (x - x0) * np.sin(theta) * np.cos(theta) * (a**-2 - b**-2)

def cy(x, x0, y0, a, b, theta):
    theta = theta * np.pi/180
    return (x - x0)**2 * (np.cos(theta)**2 / a**2 + np.sin(theta)**2 / b**2) - 1

def ey1(x, x0, y0, a, b, theta):
    Ay = ay(x, x0, y0, a, b, theta)
    By = by(x, x0, y0, a, b, theta)
    Cy = cy(x, x0, y0, a, b, theta)
    term = np.sqrt(By**2 - 4*Ay*Cy)
    y = (-By + term) / 2 / Ay + y0
    return y

def ey2(x, x0, y0, a, b, theta):
    Ay = ay(x, x0, y0, a, b, theta)
    By = by(x, x0, y0, a, b, theta)
    Cy = cy(x, x0, y0, a, b, theta)
    term = np.sqrt(By**2 - 4*Ay*Cy)
    y = (-By - term) / 2 / Ay + y0
    return y

def xrange(x0, y0, a, b, theta):
    theta = theta * np.pi/180
    xmin = -np.sqrt(a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2) + x0
    xmax = np.sqrt(a**2 * np.cos(theta)**2 + b**2 * np.sin(theta)**2) + x0
    return xmin, xmax

def yrange(x0, y0, a, b, theta):
    theta = theta * np.pi/180
    ymin = -np.sqrt(a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2) + y0
    ymax = np.sqrt(a**2 * np.sin(theta)**2 + b**2 * np.cos(theta)**2) + y0
    return ymin, ymax

#x = np.linspace(0, 8, 801)
#y1, y2 = ey1(x, *args), ey2(x, *args)
#idx = -np.isnan(y1)  # same as y2
#x, y1, y2 = x[idx], y1[idx], y2[idx]
#xmin, xmax = xrange(*args)
#ymin, ymax = yrange(*args)
#plt.plot(np.hstack((x, x[::-1])), np.hstack((y1, y2[::-1])), 'k-')







import galaxyphot
import matplotlib.patches


data = np.zeros((50, 50))
data[0::2,0::2], data[1::2,1::2] = 1, 1

x0, y0 = 24.3, 37.1
a, b = 10.0, 8.0
theta = 30.0

self = galaxyphot.EllipticalAperture((x0, y0), a, b, theta, image=data)
weights = self.weights()

extent = np.roll(self.ijextent, 2)
plt.cla()
#arr = self.section
arr = weights
plt.imshow(arr, origin='lower', interpolation='nearest', extent=extent, cmap=plt.cm.gray)#, alpha=0.1, )
patch = matplotlib.patches.Ellipse((self.j, self.i), 2*a, 2*b, angle=theta, ec='k', fc='none')
plt.gca().add_patch(patch)
plt.plot(self.j, self.i, 'gx')
plt.axvline(self.jmin, color='g')
plt.axvline(self.jmax, color='g')
plt.axhline(self.imin, color='g')
plt.axhline(self.imax, color='g')
plt.axis(extent)






import matplotlib.patches

e = matplotlib.patches.Ellipse((x0, y0), 2*a, 2*b, angle=theta)
plt.gca().add_patch(e); plt.draw()
v = e.get_verts()
plt.plot(v[:,0], v[:,1], 'k-')

e.center
e.width
e.height
e.angle

bbox = e.get_extents()
e.get_verts

e.contains_point
e.get_path
    contains_path
    contains_point
    contains_points
    intersects_bbox
    intersects_path

    codes
    get_extents
    vertices
    to_polygons

    iter_segments


e.contains_point((3,88, 1.66))  # False
e.contains_point((3.88, 1.69))  # True; outside curve, inside polygon!
e.contains_point((3.88, 1.72))  # True; outside curve, inside polygon!
e.contains_point((3.88, 1.74))  # True


