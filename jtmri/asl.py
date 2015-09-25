import numpy as np

rbf_human_params = {
    't1_blood': 1.55,
    'max_perfusion_value': 600,
}


rbf_rat_params = {
    't1_blood': 1.14,
    'max_perfusion_value': 1000
}


rbf_params = {
    'human': rbf_human_params,
    'rat': rbf_rat_params
}


def pwi(data):
    """Create a profusion weighted image from a dicom series of ASL images
    Args:
      data -- Data containing a ASL control/label images on last axis
    Returns a single profusion weighted image.

    The profusion weighted image is calculated by subtracting 
    """
    # Throw away the first pair because this may not be steady state
    controls = data[..., 2::2].astype(float)
    labels = data[..., 3::2].astype(float)
    return (controls - labels).mean(axis=-1)


def rbf(pwi, m0, inv_delay, rbf_params):
    """Compute a Renal Blood Flow map
    Args:
      pwi        -- perfusion weighted image
      m0         -- M0 image or constant value
      inv_delay  -- inversion time (seconds) (TI)
      rbf_params -- dict of params dependent on the subject
    Returns RBF map
    """
    blood_water_coeff = 80.  # (ml / 100g)
    inv_efficiency = 0.95

    rbf = blood_water_coeff / (2 * (inv_delay / 60.) * inv_efficiency) * (pwi / m0) * np.exp(inv_delay / rbf_params['t1_blood'])
    mask = (rbf > rbf_params['max_perfusion_value']) | (rbf < 0)
    return np.ma.array(rbf, mask=mask)
