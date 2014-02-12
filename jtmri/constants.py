class Constants(object):
    @property
    def gamma(self):
        '''Gyromagnetic ratio for proton
        Units: Hz / T'''
        return 42.58 * 1e6

    @property
    def delta_chi_hb(self):
        '''Magnetic susceptibility difference of hemoglobin.
        delta_chi_hb = deoxyHb - oxyHb
        Units: no units'''
        return 0.18e-6

    def Bo(self, strength):
        '''Main mangnetic field strength.
        This method exists solely for documentation purposes.
        Units: Tesla'''
        return strength
