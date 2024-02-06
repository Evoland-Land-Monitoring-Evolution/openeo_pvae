#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
Provide the user-defined function to call super-resolution model
"""
import sys
from typing import Dict

import numpy as np
import xarray as xr
from openeo.metadata import CollectionMetadata, SpatialDimension
from openeo.udf import XarrayDataCube


def apply_metadata(metadata: CollectionMetadata, _: dict) -> CollectionMetadata:
    """
    Modify metadata according to up-sampling factor of 2.
    """
    new_dimensions = metadata._dimensions.copy()

    for index, dim in enumerate(new_dimensions):
        if isinstance(dim, SpatialDimension):
            new_dim = SpatialDimension(
                name=dim.name, extent=dim.extent, crs=dim.crs, step=dim.step / 2.0
            )
            new_dimensions[index] = new_dim
    return metadata._clone_and_update(dimensions=new_dimensions)


def check_datacube(cube: xr.DataArray):
    """ """
    if cube.data.ndim != 4:
        raise RuntimeError("DataCube dimensions should be (t,bands, y, x)")

    if cube.data.shape[1] != 10:
        raise RuntimeError(
            "DataCube should have 10 bands exactly (B2, B3, B4, B8, B5, B6, B7, B8A, B11, B12)"
        )


def fancy_upsample_function(array: np.ndarray) -> np.ndarray:
    """ """
    return array.repeat(2, axis=-1).repeat(2, axis=-2)


def run_inference(input_data: np.ndarray) -> np.ndarray:
    """ """
    # First get virtualenv
    sys.path.insert(0, "tmp/extra_venv")
    import onnxruntime as ort

    model_file = "tmp/extra_files/carn_3x3x64g4sw_bootstrap.onnx"

    # ONNX inference session options
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.use_deterministic_compute = True

    # Execute on cpu only
    ep_list = ["CPUExecutionProvider"]

    # Create inference session
    ort_session = ort.InferenceSession(model_file, sess_options=so, providers=ep_list)

    ro = ort.RunOptions()
    ro.add_run_config_entry("log_severity_level", "3")

    outputs = ort_session.run(None, {"input": input_data[None, ...]}, run_options=ro)

    return outputs[0][0, ...]


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    """
    Apply UDF function to datacube
    """
    # We get data from datacube
    cubearray: xr.DataArray
    if isinstance(cube, xr.DataArray):
        cubearray = cube
    else:
        cubearray: xr.DataArray = cube.get_array().copy()

    # NOTE: In the apply() process the cubes should always be (bands,y,x)
    # assert cubearray.data.ndim == 4

    # Pixel size of the original image
    init_pixel_size_x = cubearray.coords["x"][-1] - cubearray.coords["x"][-2]
    init_pixel_size_y = cubearray.coords["y"][-1] - cubearray.coords["y"][-2]

    # Build new coordinates arrays
    coord_x = np.linspace(
        start=cubearray.coords["x"].min(),
        stop=cubearray.coords["x"].max() + init_pixel_size_x,
        num=2 * cubearray.coords["x"].shape[0],
        endpoint=False,
    )
    # FIX: y coordinates go from max to min.
    coord_y = np.linspace(
        start=cubearray.coords["y"].max(),
        stop=cubearray.coords["y"].min() + init_pixel_size_y,
        num=2 * cubearray.coords["y"].shape[0],
        endpoint=False,
    )

    if cubearray.data.ndim == 4 and cubearray.data.shape[0] != 1:
        cube_collection = []
        # Perform inference for each date
        for c in range(len(cubearray.data)):
            predicted_array = run_inference(cubearray.data[c])
            # predicted_array = fancy_upsample_function(cubearray.data[c])
            cube_collection.append(predicted_array)

        # Stack all dates
        cube_collection = np.stack(cube_collection, axis=0)

        # Build output data array
        # FIX: shape is (t,bands,y,x) on Terrascope backend. This is
        # different in execute_local_udf, issue has been logged.
        predicted_cube = xr.DataArray(
            cube_collection,
            dims=["t", "bands", "y", "x"],
            coords=dict(x=coord_x, y=coord_y),
        )
    else:
        if cubearray.data.ndim == 4 and cubearray.data.shape[0] == 1:
            cubearray = cubearray[0]

        predicted_array = run_inference(cubearray.data)

        # Build output data array
        # FIX: shape is (t,bands,y,x) on Terrascope backend. This is
        # different in execute_local_udf, issue has been logged.
        predicted_cube = xr.DataArray(
            predicted_array,
            dims=["bands", "y", "x"],
            coords=dict(x=coord_x, y=coord_y),
        )
    return XarrayDataCube(predicted_cube)
