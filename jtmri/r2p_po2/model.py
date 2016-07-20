from collections import namedtuple
import itertools
import logging
import time

from matplotlib import pyplot as pl
from numpy.random import uniform
from numpy.linalg import norm
import numpy as np
import scipy.stats

import jtmri.fit
import jtmri.np


log = logging.getLogger(__name__)


def unit_vec(a):
    """Return the unit vector"""
    a = np.array(a)
    a_norm = norm(a)
    if a_norm == 0:
        return a
    return a / a_norm


def proj(a, b):
    """Project a onto b"""
    b_hat = unit_vec(b)
    return a.dot(b_hat) * b_hat


def rej(a, b):
    """Return the rejection of a onto b
    Equivalent to the projection of a onto a plane orthogonal to b
    """
    a = np.array(a)
    b = np.array(b)
    return a - proj(a, b)


def angle(a, b):
    """Return the angle between vector a and b"""
    a_hat = unit_vec(a)
    b_hat = unit_vec(b)
    return np.arccos(np.clip(a_hat.dot(b_hat), -1.0, 1.0))


RBC = namedtuple('RBC', 'radius,position')


def rbc_magnetic_field_shift(pos, rbcs, Bo, delta_chi, theta=None):
    """Magnetic field shift due to a list of red blood cells
    
    Parameters
    ----------
    pos : (3x1) vector
        Position to evalute magnetic field at
    rbcs : list(RBC)
        Red Blood Cell to simulate magnetic field of
    Bo : (3x1) vector or scalar
        Main magnetic field vector
    delta_chi : scalar
        Suscetibility of red blood cell
    theta : scalar (default: None)
        Angle of magnetic field, `Bo` must be a scalar if this is specified

    Returns
    -------
    scalar
        Shift in magnetic field due to `rbc` at point `pos`.
    """
    if theta is not None:
        assert np.isscalar(Bo)
        theta_fn = lambda r: theta
        Bo_mag = Bo
    else:
        Bo_mag = norm(Bo)
        theta_fn = lambda r: angle(r, Bo)
    fs = 0
    for rbc in jtmri.utils.as_iterable(rbcs):
        r = pos - rbc.position
        r_mag = norm(r)
        if r_mag >= rbc.radius:
            fs += Bo_mag \
            * (delta_chi / 3.) \
            * (rbc.radius / r_mag)**3 \
            * (3 * np.cos(theta_fn(r))**2 - 1)
    return fs


def linear_spherical_overlap(r1, r2, d):
    rs = r1 + r2
    rd = np.abs(r1 - r2)
    overlap = (rs - d) / (rs - rd)
    return np.clip(overlap, 0, 1)


def rbc_intersects(rbc, rbcs, max_overlap):
    """Check if `rbc` intersects any rbc in `rbcs`"""
    if len(rbcs) == 0:
        return False
    rs = np.array([x.radius for x in rbcs])
    pos = np.array([x.position for x in rbcs])

    dists = (((rbc.position - pos))**2).sum(axis=1) ** 0.5
    overlaps = linear_spherical_overlap(rbc.radius, rs, dists)
    return np.any(overlaps > max_overlap)


def fill_cylinder_with_rbcs(cylinder_radius, cylinder_len, hct, rbc_radius_fn, max_iter=1e5,
                            epsilon=1e-2, max_overlap=0.5):
    """Add red blood cells to a cylinder until the specified hematocrit has been reached
    Cylinder axis lies on z axis
    """
    total_sphere_volume = 0
    last_err = float('inf')
    total_cylinder_volume = np.pi * cylinder_radius**2 * cylinder_len
    rbcs = []
    intersected = 0
    for i in itertools.count(start=1):
        if i >= max_iter:
            raise Exception('Max iterations reached: {}'.format(max_iter))
        rc = rbc_radius_fn()
        r = (cylinder_radius - rc) * np.sqrt(uniform(0, 1))
        theta = uniform(0, 2 * np.pi)
        z = uniform(-cylinder_len / 2. + rc, cylinder_len / 2. - rc)
        position = [
            r * np.cos(theta),
            r * np.sin(theta),
            z]
        rbc = RBC(rc, np.array(position))
        # Check for intersecting spheres
        if rbc_intersects(rbc, rbcs, max_overlap):
            intersected += 1
            continue
        # Check if hematocrit has been reached
        rbc_volume = (4./3) * np.pi * rbc.radius**3
        curr_hct = (total_sphere_volume + rbc_volume) / total_cylinder_volume
        err = np.abs(curr_hct - hct)
        if err > last_err:
            raise Exception('error has increased, error: {} last_error: {} iteration: {}'
                            .format(err, last_err, i))
            break
        last_err = err
        total_sphere_volume += rbc_volume
        rbcs.append(rbc)
        if err < epsilon:
            break
    log.debug('finished generating RBCs, i: %d error: %f hct: %f num_rbcs: %d dropped: %d efficiency: %f',
              i, err, curr_hct, len(rbcs), intersected, len(rbcs) / float(i))
    return rbcs


def extra_vascular_magnetic_field_shift(r, cylinder, Bo, delta_chi):
    """Returns the magnetic field at position `r` due to an infinite cylinder
    Parameters
    ----------
    r : 3-vector
        Point to estimate field at
    cylinder : Cylinder
        Parameters describing the infinite cylinder
    Bo : 3-vector
        Magnetic field vector
    delta_chi : float
        Susceptibility difference between inside and outside of cylinder
        
    Returns
    -------
    float
        Magnetic field offset in units of input parameter `Bo`
    """
    r = np.array(r)
    Bo = np.array(Bo)
    phi = angle(rej(r, cylinder.axis), rej(Bo, cylinder.axis))
    theta = angle(Bo, cylinder.axis)
    r_mag = cylinder.distance_to_point(r)
    
    assert r_mag > cylinder.radius
    # Extra vascular
    return norm(Bo) \
              * (delta_chi / 2.) \
              * (cylinder.radius / r_mag)**2 \
              * np.cos(2 * phi) \
              * np.sin(theta)**2


class Cylinder(object):
    @classmethod
    def from_axis_offset(cls, axis, offset, radius):
        axis = np.array(axis)
        offset = np.array(offset)
        x0 = axis + offset
        x1 = offset
        return Cylinder(x0, x1, radius)

    def __init__(self, x0, x1, radius):
        assert len(x0) == 3, 'x0 should be a 3-vector'
        assert len(x1) == 3, 'x1 should be a 3-vector'
        self._x0 = np.array(x0)
        self._x1 = np.array(x1)
        assert not np.array_equal(self._x0, self._x1), 'x0 must be unique from x1'
        self._radius = radius
        self._axis = self._x1 - self._x0
        
    @property
    def x0(self):
        return self._x0
    
    @property
    def x1(self):
        return self._x1
    
    @property
    def radius(self):
        return self._radius
    
    @property
    def axis(self):
        return self._axis
    
    def distance_to_point(self, p):
        """Returns the minimum distance between cylinder axis and point `p`"""
        p = np.array(p)
        return norm(np.cross(self.axis, self.x1 - p)) / norm(self.axis)


def sample_voxel(voxel, N_samples):
    """Create a 3 x N_sample vector over the voxel"""
    N_samples = int(N_samples)
    x_samples = uniform(low=-voxel[0]/2., high=voxel[0]/2., size=N_samples)
    y_samples = uniform(low=-voxel[1]/2., high=voxel[1]/2., size=N_samples)
    z_samples = uniform(low=-voxel[2]/2., high=voxel[2]/2., size=N_samples)
    return np.array(zip(x_samples, y_samples, z_samples)).T


def sample_voxel_ev(voxel, N_samples, cylinders):
    """Sample a voxel's extra-vascular space"""
    samples = []
    while len(samples) < N_samples:
        sample = sample_voxel(voxel, 1)
        if not is_extravascular(sample, cylinders):
            continue
        samples.append(sample)
    return np.array(np.squeeze(samples)).T


def create_cylinders(target_volume_fraction, voxel_shape, cylinder_func,
                     epsilon=1e-2, max_iterations=None, N_samples=1e5):
    """Create a volume of cylinders
    Parameters
    ----------
    target_volume_fraction : float
        Target fraction of the cuboid volume to be occupied by cylinders
    voxel_shape : 3-vector
        Dimensions of the voxel being filled, voxel is assumed to be centered
        at the origin
    cylinder_func : function
        A cylinder generating function, encapsulates the distrubtion over the
        cylinder parameters.
    epsilon : float
        Allowed error in reaching `target_volume_fraction`
    max_iterations : int
        Maximum number of iterations to perform when searching for target volume fraction
    N_samples : int
        Number of sample points to use when estimating cylinder volume

    Returns
    -------
    iterable of Cylinder
        An iterable of cylinder objects is returned, such that the RMS distance between
        the `target_volume_fraction` and the total volume fraction of `voxel_shape` occupied by 
        the cylinders is less than `epsilon`.
    """
    assert 0 <= target_volume_fraction <= 1
    target_volume_fraction = float(target_volume_fraction)
    cylinder_func = cylinder_func or random_cylinder
    epsilon_sqd = epsilon**2
    N_samples = int(N_samples)
    samples = sample_voxel(voxel_shape, N_samples)
    mask = np.zeros(N_samples, dtype=bool)
    cylinders = []
    loop = itertools.count() if max_iterations is None else xrange(1, max_iterations + 1)
    last_err = float('inf')
    for i in loop:
        cylinder = cylinder_func()
        mask |= cylinder_mask(cylinder, samples)
        volume_fraction = mask.sum() / float(N_samples)
        err = (1 - (volume_fraction / target_volume_fraction))**2
        if err > last_err:
            log.debug('current iteration error (%f) greater than previous (%f), terminating',
                      err, last_err)
            break
        cylinders.append(cylinder)
        if err <= epsilon_sqd:
            log.debug('target error (%f) reached. err: %f iteration: %d '\
                      'target_volume_fraction: %f actual_volume_fractions: %f',
                      epsilon_sqd, err, i, target_volume_fraction, volume_fraction)
            break
        last_err = err
    if i == max_iterations:
        log.debug('max iterations (%d) reached, loop terminated', max_iterations)
    return cylinders


def random_unit_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0, np.pi*2)
    costheta = np.random.uniform(-1, 1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.array([x, y, z])


def random_cylinder(radius_limits=(0, 1), x_limits=(-1, 1), y_limits=(-1, 1), z_limits=(-1, 1)):
    """Generate a randomly oriented cylinder with uniform direction vector distribution"""
    r = uniform(*radius_limits)
    axis = random_unit_three_vector()
    offset = np.array([uniform(*l) for l in (x_limits, y_limits, z_limits)])
    return Cylinder.from_axis_offset(axis, offset, r)


def cylinder_mask(cylinder, sample_grid):
    """Returns a 3D array where True values are inside the 
    cylinder and False values lie outside.
    The `cylinder` is sampled on `sample_grid`.
    """
    X, Y, Z = sample_grid
    
    c = cylinder
    v = unit_vec(c.axis)
    a = (X - c.x0[0]) * v[0] + (Y - c.x0[1]) * v[1] + (Z - c.x0[2]) * v[2]
    rsq = (c.x0[0] + a * v[0] - X)**2 + (c.x0[1] + a * v[1] - Y)**2 + (c.x0[2] + a * v[2] - Z)**2
    return rsq <= c.radius**2


def estimate_cylinder_volume(voxel_shape, cylinders, N_samples=1e5):
    """Estimate the volume consumed by the cylinders within a voxel using Monte Carlo integration
    
    Parameters
    ----------
    voxel_shape : 3 vector
        height, width, length of a voxel
    cylinders : list of Cylinder
        Cylinders to estimate the volume of
    N_samples : int
        Number of samples to use when estimating the volume
        
    Returns
    -------
    float
        Estimated volume of cylinders contained in the voxel
    """
    N_samples = int(N_samples)
    samples = sample_voxel(voxel_shape, N_samples)
    mask = np.zeros(N_samples, dtype=bool)
    for c in cylinders:
        mask |= cylinder_mask(c, samples)
    return mask.sum() / float(N_samples)


def diffuse(points, time_step, diffusion_coefficient):
    """Simulate diffusion by randomly walking each point"""
    assert diffusion_coefficient >= 0
    assert points.shape[0] == 3, 'Must be a 3 x N vector'
    l = np.sqrt(2 * diffusion_coefficient * time_step)
    pt = lambda: l * (2 * np.round(uniform(0, 1, points.shape[1])) - 1)
    return points + np.array([pt(), pt(), pt()])


def is_extravascular(point, cylinders):
    for cylinder in cylinders:
        r = cylinder.distance_to_point(point)
        if r < cylinder.radius:
            return False
    return True


def voxel_signal(phases):
    """Compute the signal for a voxel given a list of phases,
    phases is 2d, with the second dimension representing each proton
    phase
    """
    return np.sum(np.exp(1j * phases), axis=1) / phases.shape[1]


def sample_cylinder_center(max_radius, num_protons):
    theta = uniform(0, 2*np.pi, num_protons)
    radius = max_radius * np.sqrt(uniform(0, 1))
    return np.array([
        radius * np.cos(theta),
        radius * np.sin(theta),
        np.zeros_like(theta)])


def sim_iv(params):
    start = time.time()
    log.info('Starting IV simulation')
    log.debug(params)
    time_step_count = int(np.ceil(params.time_total / params.time_step))
    time_points = np.arange(0, params.time_total + params.time_step, params.time_step)

    bs = []
    for i in range(params.N_iv_cylinders):
        radius = uniform(*params.cylinder_radius_limits)
        protons = sample_cylinder_center(radius, params.N_protons_iv)
        log.debug('IV cylinder created, radius: %fm protons: %d', radius, protons.shape[1])
        for i, proton in enumerate(jtmri.np.iter_axes(protons, axes=1)):
            log.debug('simulating proton %d of %d', i, protons.shape[1])
            rbcs = fill_cylinder_with_rbcs(radius,
                                           params.iv_cylinder_len,
                                           params.Hct,
                                           lambda: uniform(*params.rbc_radius_limits),
                                           max_overlap = params.iv_rbc_overlap)
            log.debug('generated %d rbcs in a cylinder with radius: %f', len(rbcs), radius)
            # Cylinder axis lies along z-axis, force theta to follow sin(theta) distribution
            theta = np.arccos(1 - 2 * np.random.uniform())
            log.debug('simulating cylinder with angle %f', theta)
            b = []
            for _ in time_points:
                field_shift = rbc_magnetic_field_shift(proton,
                                                       rbcs,
                                                       norm(params.Bo),
                                                       params.delta_chi_rbc,
                                                       theta=theta)
                b.append(field_shift)
                proton = diffuse(proton[:, np.newaxis],
                                 params.time_step,
                                 params.diffusion_coefficient_iv)[:, 0]
            bs.append(b)
    bs = np.array(bs).transpose()

    # Compute phases
    freqs = params.gyromagnetic_ratio * bs
    phases = scipy.integrate.cumtrapz(freqs, dx=params.time_step, axis=0, initial=0)
    end = time.time()
    log.info('IV simulation finished in %d seconds', end - start)
    return time_points, phases


def sim_ev(params):
    centered = lambda l: (-0.5*l, 0.5*l)
    time_step_count = int(np.ceil(params.time_total / params.time_step))
    time_points = np.arange(0, params.time_total + params.time_step, params.time_step)
    cylinder_func = lambda: random_cylinder(params.cylinder_radius_limits,
                                            centered(params.voxel_shape[0]),
                                            centered(params.voxel_shape[1]),
                                            centered(params.voxel_shape[2]))
    start = time.time()
    log.info('Starting EV simulation')
    log.info('Generating cylinders')
    cylinders = create_cylinders(params.vascular_fraction,
                                 params.voxel_shape,
                                 cylinder_func=cylinder_func)
    log.info('Generated %d cylinders with final vascular faction: %f',
             len(cylinders),
             estimate_cylinder_volume(params.voxel_shape, cylinders))
    protons = sample_voxel_ev(params.voxel_shape, params.N_protons_ev, cylinders)

    # Calculate magnetic fields for each proton at each time point
    positions = [protons]
    bs = []
    for i, t in enumerate(time_points):
        b = []
        log.debug('step: %d t: %0.2g steps_remaining: %d', i, t, time_step_count - i)
        for proton in jtmri.np.iter_axes(protons, 1):
            field_shift = 0
            for cylinder in cylinders:
                if not is_extravascular(proton, [cylinder]):
                    continue
                field_shift += extra_vascular_magnetic_field_shift(proton,
                                                                   cylinder,
                                                                   params.Bo, 
                                                                   params.delta_chi_blood)
            b.append(field_shift)
        bs.append(b)
        protons = diffuse(protons, params.time_step, params.diffusion_coefficient_ev)
        positions.append(protons)
    bs = np.array(bs)
    dists = ((positions[1:] - positions[0])**2).sum(axis=1)

    # Compute phases
    freqs = params.gyromagnetic_ratio * bs
    phases = scipy.integrate.cumtrapz(freqs, dx=params.time_step, axis=0, initial=0)
    
    end = time.time()
    log.info('Simulation finished in %d seconds', end - start)
    return time_points, phases


def signal(params, phases_iv, phases_ev):
    sig_iv = np.abs(voxel_signal(phases_iv))
    sig_ev = np.abs(voxel_signal(phases_ev))
    sig = (1 - params.vascular_fraction) * sig_ev + params.vascular_fraction * params.intrinsic_signal_ratio * sig_ev * sig_iv
    return sig_iv, sig_ev, sig
