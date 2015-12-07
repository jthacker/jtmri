"""Provides an interface to the Linear registration tool FLIRT from FSL"""
import jtmri.np
import logging
import nibabel as nib
from nipype.interfaces import fsl
import numpy as np
import re
from tempfile import NamedTemporaryFile

# Silence warning about FSLOUTPUTYPE
import os
os.environ['FSLOUTPUTTYPE'] = 'NIFTI'


log = logging.getLogger(__name__)


def _ndarray_to_nifti(ndarray):
    """Convert a ndarray to a nifti temporary file
    Nifti file will be automatically deleted when the returned
    object is garbage collected
    """
    tmp = NamedTemporaryFile(suffix='.nii')
    img = nib.Nifti1Image(ndarray, np.eye(4))
    img.to_filename(tmp.name)
    return tmp


def register(ref_img, in_img, cmd=None):
    """Find the 4D affine transform that registers in_img to ref_img
    Args:
        ref_img -- (type: ndarray) reference image
        in_img  -- (type: ndarray) input image to be registered
        cmd     -- (default: None) override default flirt parameters
                   an instance of nipype.interfaces.fsl.FLIRT

    Returns: warped input image
    """
    assert in_img.ndim == ref_img.ndim, 'Input and reference images must have the same '\
                                        'number of dimensions'
    in_file = _ndarray_to_nifti(in_img)
    ref_file = _ndarray_to_nifti(ref_img)
    out_file = NamedTemporaryFile(suffix='.nii.gz')
    log_file = NamedTemporaryFile(suffix='.txt')
   
    cmd = cmd or fsl.FNIRT()
    cmd.inputs.in_file = in_file.name
    cmd.inputs.ref_file = ref_file.name
    cmd.inputs.output_type = 'NIFTI_GZ'
    cmd.inputs.log_file = log_file.name
    cmd.inputs.warped_file = out_file.name
    log.debug('cmdline: {}'.format(cmd.cmdline))
    res = cmd.run()
    out_img = nib.load(out_file.name).get_data()
    return out_img


def register_all(ref_img, in_imgs, cmd=None):
    """Register all images in in_imgs to ref_img
    Args:
        ref_img -- n dimensional reference image
        in_imgs -- n + 1 dimensional input images
        cmd     -- override the default command

    Returns: warped input images

    The last dimension of in_imgs is assumed to contained the series
    to be iterated over.
    """
    assert (ref_img.ndim + 1) == in_imgs.ndim, 'ref_img should have one less dimension '\
                                               'than in_imgs'
    out_imgs = []
    for img in jtmri.np.iter_axis(in_imgs, -1):
        out_imgs.append(register(ref_img, img, cmd))
    return np.concatenate([im[..., np.newaxis] for im in out_imgs], axis=-1)
