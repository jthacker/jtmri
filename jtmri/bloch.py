from __future__ import division
from collections import namedtuple
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pylab as pl


_RF = namedtuple('RF', ['time', 'angle', 'rot'])

_rotx = lambda theta: np.array([[1,0,0],
                                [0, np.cos(theta), -1*np.sin(theta)],
                                [0, np.sin(theta), np.cos(theta)]])
    
_roty = lambda theta: np.array([[np.cos(theta), 0, np.sin(theta)],
                                [0, 1, 0],
                                [-1*np.sin(theta), 0, np.cos(theta)]])
    
_rotz = lambda theta: np.array([[np.cos(theta), -1*np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    
rfx = lambda t, angle: _RF(t, angle, _rotx(angle))
rfy = lambda t, angle: _RF(t, angle, _roty(angle))
rfz = lambda t, angle: _RF(t, angle, _rotz(angle))


def sim(t, T1, T2, df=0, rfs=tuple(), M0=None):
    '''
    Args:
    t   -- Time points to simulate at. The time step is calculated as the difference
           between successive points
    T1  -- T1 decay in seconds
    T2  -- T2 decay in seconds
    df  -- Off-resonance in Hz
    rfs -- List of RF pulses
    M0  -- Initial magnetization

    Returns:
    The magnetization vector at each time point.
    '''
    ident = np.eye(3)
    out = np.empty((len(t),3,1))
    if M0 is None:
        M0 = np.array([[0],[0],[1]])
    if len(t) > 0:
        rots = [rf.rot for rf in rfs if rf.time == t[0]]
        M = np.dot(reduce(np.dot, rots, ident), M0)
        out[0] = M

        for i,(t0,dt) in enumerate(zip(t[1:], np.diff(t)), 1):
            phi = 2 * np.pi * dt * df
            rots = [_rotz(phi)] + [rf.rot for rf in rfs if t0-dt < rf.time <= t0]
            rot = reduce(np.dot, rots, ident)
            e1 = np.exp(-dt / T1)
            e2 = np.exp(-dt / T2)
            A = np.array([[e2,  0,  0],
                          [0 , e2,  0],
                          [0 ,  0, e1]])
            A = np.dot(A, rot)
            B = np.array([[0], [0], [1-e1]])
            M = np.dot(A, M) + B
            out[i] = M
    return out


def plot_magnetization(t, M, ax=None):
    if ax is None:
        _,ax = pl.subplots()
    ax.plot(t, M[:,0], label='Mx')
    ax.plot(t, M[:,1], label='My')
    ax.plot(t, M[:,2], label='Mz')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal (a.u.)')
    ax.legend()
    ax.grid()
    return ax


def plot_rfs(t, rfs, ax=None):
    if ax is None:
        _,ax = pl.subplots()
    for rf in rfs:
        ax.plot((rf.time,rf.time), (0, np.rad2deg(rf.angle)), 'k-')
        ax.plot(rf.time, np.rad2deg(rf.angle), 'k^')
    ax.set_xlim(min(t), max(t))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.grid()
    return ax
