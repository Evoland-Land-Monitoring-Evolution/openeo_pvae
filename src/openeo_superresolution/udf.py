#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
Provide the user-defined function to call super-resolution model
"""
import hashlib
import os
from typing import Dict

import numpy as np
import onnxruntime as ort
import requests
import xarray as xr
from openeo.metadata import CollectionMetadata, SpatialDimension
from openeo.udf import XarrayDataCube

MODEL_URL = 'https://mycore.core-cloud.net/index.php/s/Zyc2RaFdi41gzth/download'
MODEL_SHA1 = 'f91d6419978cf3e34cccac52526c5cd19f7b21c5'


def fetch_model_parameters():
    """
    Download model
    """
    model_download_path = '/data/users/Public/' + os.environ[
        'USER'] + '/sisr_models'

    # Ensure path exists
    os.makedirs(model_download_path, exist_ok=True)

    model_local_path = os.path.join(model_download_path, 'model.onnx')

    # Check if we need to download model
    if not os.path.exists(model_local_path) or hashlib.sha1(
            model_local_path).hexdigest() != MODEL_SHA1:
        # In that case, download it with requests (timeout 10s)
        requests_result = requests.get(MODEL_URL, timeout=10)
        with open(model_local_path, 'wb') as fd:
            fd.write(requests_result.content)

    return model_local_path


def apply_metadata(metadata: CollectionMetadata,
                   _: dict) -> CollectionMetadata:
    """
    Modify metadata according to up-sampling factor of 2.
    """
    new_dimensions = metadata._dimensions.copy()

    for index, dim in enumerate(new_dimensions):
        if isinstance(dim, SpatialDimension):
            new_dim = SpatialDimension(name=dim.name,
                                       extent=dim.extent,
                                       crs=dim.crs,
                                       step=dim.step / 2.)
            new_dimensions[index] = new_dim
    return metadata._clone_and_update(dimensions=new_dimensions)


def check_datacube(cube: xr.DataArray):
    """
    """
    if cube.data.ndim != 4:
        raise RuntimeError('DataCube dimensions should be (t,bands, y, x)')

    if cube.data.shape[1] != 4:
        raise RuntimeError(
            'DataCube should have 4 bands exactly (B2, B3, B4, B8)')


def run_inference(input_data: np.ndarray) -> np.ndarray:
    """
    """
    model_file = fetch_model_parameters()

    # ONNX inference session options
    so = ort.SessionOptions()
    so.intra_num_threads = 1
    so.inter_num_threads = 1
    so.use_deterministic = True

    # Execute on cpu only
    ep_list = ['CPUExecutionProvider']

    # Create inference session
    ort_session = ort.InferenceSession(model_file,
                                       sess_options=so,
                                       providers=ep_list)

    ro = ort.RunOptions()
    ro.add_run_config_entry('log_severity_level', '3')

    outputs = ort_session.run(None, {'input': input_data}, run_options=ro)

    return outputs[0]


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    """
    Apply UDF function to datacube
    """
    # We get data from datacube
    cubearray: xr.DataArray = cube.get_array().copy()

    # NOTE: In the apply() process the cubes should always be (bands,y,x)
    assert cubearray.data.ndim == 4

    # Pixel size of the original image
    init_pixel_size_x = cubearray.coords['x'][-1] - cubearray.coords['x'][-2]
    init_pixel_size_y = cubearray.coords['y'][-1] - cubearray.coords['y'][-2]

    cube_collection = []
    # Perform inference for each date
    for c in range(len(cubearray.data)):
        predicted_array = run_inference(cubearray.data[c])

    # Stack all dates
    cube_collection = np.stack(cube_collection, axis=0)

    # Build new coordinates arrays
    coord_x = np.linspace(start=cube.get_array().coords['x'].min(),
                          stop=cube.get_array().coords['x'].max() +
                          init_pixel_size_x,
                          num=predicted_array.shape[-2],
                          endpoint=False)
    # FIX: y coordinates go from max to min.
    coord_y = np.linspace(start=cube.get_array().coords['y'].max(),
                          stop=cube.get_array().coords['y'].min() +
                          init_pixel_size_y,
                          num=predicted_array.shape[-1],
                          endpoint=False)

    # Build output data array
    # FIX: shape is (t,bands,y,x) on Terrascope backend. This is
    # different in execute_local_udf, issue has been logged.
    predicted_cube = xr.DataArray(cube_collection,
                                  dims=['t', 'bands', 'y', 'x'],
                                  coords=dict(x=coord_x, y=coord_y))

    return predicted_cube
