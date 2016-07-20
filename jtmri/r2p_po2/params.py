import numpy as np


##############################
## Configuration Parameters ##
##############################
class SimParams(object):
    N_protons_ev = 1000  # Number of protons to simulate
    N_protons_iv = 100
 
    # Used for simulating vessel with RBCs inside
    N_iv_cylinders = 1 # Number of cylinders to simulate in IV model
    N_iv_cylinder_orientations = 10 # Number of orientations to sim for each cylinder
    iv_cylinder_len = 200e-6  # Len of IV cylinder to fill with RBCs
    iv_rbc_overlap = 0.25  # Amount that generated RBCs are allowed to overlap

    diffusion_coefficient_ev = 1.5e-9  # (m^2 / s)
    diffusion_coefficient_iv = 1.5e-9  # (m^2 / s)

    voxel_shape = [160e-6, 160e-6, 160e-6]  # voxel lengths (m)
    time_step = 2e-3  # (s)
    time_total = 60e-3  # (s)

    # Radius of vessels
    cylinder_radius_limits = (25e-6, 25e-6)  # (m)

    vascular_fraction = 0.04  # unitless

    Bo = np.array([0, 0, 3])  # Main magnetic field vector (Tesla)
    Hct = 0.4
    SHb = 0.5

    # oxy-deoxy blood susceptibility difference (unitless)
    # In the source cited, the units are CGS, to convert them to MKS, the difference
    # must be multiplied by 4pi
    #
    # @see Spees, William M., et al. 
    # "Water proton MR properties of human blood at 1.5 Tesla ...
    # Magnetic resonance in medicine 45.4 (2001): 533-542.
    @property
    def delta_chi_rbc(self):
        return 4 * np.pi * 0.27e-6 * (1 - self.SHb)

    @property
    def delta_chi_blood(self):
        return self.delta_chi_rbc * self.Hct

    # ratio of intrinsic signals of IV and EV components, including the effect of relaxation
    # weighting.
    # @see Martindale J, Kennerley AJ, Johnston D, Zheng Y, Mayhew JE.
    # Theory and generalization of Monte Carlo models of the BOLD signal ...
    intrinsic_signal_ratio = 1.4

    gyromagnetic_ratio = 2.67513 * 10**8  # rad / s / T

    # Radius of Red Blood Cell limits
    rbc_radius_limits = (3e-6, 3e-6)


    def __init__(self, **kwargs):
        keys = dir(self)
        for k, v in kwargs.iteritems():
            assert k in keys, '%r is not a parameter key' % (k,)
            self.__dict__[k] = v
            
    def __str__(self):
        s = ['Simulation Parameters:']
        s.append('=' * 61)
        for attr in sorted(dir(self), key=lambda x: x.lower()):
            if attr.startswith('_'):
                continue
            val = getattr(self, attr)
            s.append('{:<30} {:>30}'.format(attr, val))
        s.append('=' * 61)
        return '\n'.join(s) 
