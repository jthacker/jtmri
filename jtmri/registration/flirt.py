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


def _fsl_mat_to_ndarray(fsl_mat):
    """Convert a FSL ASCII matrix a ndarray"""
    regex = re.compile(r'\s+')
    out = []
    for line in fsl_mat.strip().split('\n'):
        out.append(np.array([float.fromhex(x) for x in regex.split(line.strip())]))
    return np.array(out)


def register(ref_img, in_img, ref_weights=None, in_weights=None, cmd=None):
    """Find the 4D affine transform that registers in_img to ref_img
    Args:
        ref_img     -- (type: ndarray) reference image
        in_img      -- (type: ndarray) input image to be registered
        ref_weights -- (type: ndarray, default: None) weights for reference image
        in_weights  -- (type: ndarray, default: None) weights for input image
        cmd         -- (default: None) override default flirt parameters
                       an instance of nipype.interfaces.fsl.FLIRT

    Returns:
        transform -- 4D affine transformation to register in_img to ref_img
        out_img   -- in_img with transform applied
    """
    assert in_img.ndim == ref_img.ndim, 'Input and reference images must have the same '\
                                        'number of dimensions'
    in_file = _ndarray_to_nifti(in_img)
    ref_file = _ndarray_to_nifti(ref_img)
    out_file = NamedTemporaryFile(suffix='.nii.gz')
    log_file = NamedTemporaryFile(suffix='.txt')
    matrix_file = NamedTemporaryFile(suffix='.mat')
   
    cmd = cmd or fsl.FLIRT()
    cmd.inputs.in_file = in_file.name
    cmd.inputs.reference = ref_file.name
    cmd.inputs.output_type = 'NIFTI_GZ'
    cmd.inputs.out_file = out_file.name
    cmd.inputs.out_log = log_file.name
    cmd.inputs.out_matrix_file = matrix_file.name
    if ref_weights is not None:
        cmd.inputs.ref_weight = _ndarray_to_nifti(ref_weights)
    if in_weights is not None:
        cmd.inputs.in_weight = _ndarray_to_nifti(in_weights)

    log.debug('cmdline: {}'.format(cmd.cmdline))
    res = cmd.run()
    transform = _fsl_mat_to_ndarray(matrix_file.read())
    out_img = nib.load(out_file.name).get_data()
    return transform, out_img


def register_all(ref_img, in_imgs, cmd=None):
    """Register all images in in_imgs to ref_img
    Args:
        ref_img -- n dimensional reference image
        in_imgs -- n + 1 dimensional input images
        cmd     -- override the default command

    Returns:
        matricies -- iterable of matricies to transform each input
                     image to the reference image
        images    -- transformed input images

    The last dimension of in_imgs is assumed to contained the series
    to be iterated over.
    """
    assert (ref_img.ndim + 1) == in_imgs.ndim, 'ref_img should have one less dimension '\
                                               'than in_imgs'
    matricies, out_imgs = [], []
    for img in jtmri.np.iter_axis(in_imgs, -1):
        matrix, out_img = register(ref_img, img, cmd)
        matricies.append(matrix)
        out_imgs.append(out_img)
    return matricies, np.concatenate([im[..., np.newaxis] for im in out_imgs], axis=-1)
