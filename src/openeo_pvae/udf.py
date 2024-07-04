#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
Provide the user-defined function to call super-resolution model
"""
import sys
from typing import Dict

import numpy as np
import xarray as xr
from openeo.metadata import CollectionMetadata, SpatialDimension, BandDimension, Band
from openeo.udf import XarrayDataCube


def apply_metadata(metadata: CollectionMetadata, _: dict) -> CollectionMetadata:
    """
    Modify metadata according to up-sampling factor of 2.
    """
    new_dimensions = metadata._dimensions.copy()

    for index, dim in enumerate(new_dimensions):
        if isinstance(dim, BandDimension):
            new_dim = BandDimension(
                name=dim.name, bands=["F01_mu", "F02_mu", "F03_mu", "F04_mu", "F05_mu", "F06_mu", "F07_mu", "F08_mu", "F09_mu", "F10_mu", "F11_mu",
                                      "F01_logvar", "F02_logvar", "F03_logvar", "F04_logvar", "F05_logvar", "F06_logvar", "F07_logvar", "F08_logvar", "F09_logvar", "F10_logvar", "F11_logvar"]
            )
            new_dimensions[index] = new_dim
    return metadata._clone_and_update(dimensions=new_dimensions)


def check_datacube(cube: xr.DataArray):
    """ """
    if cube.data.ndim != 4:
        raise RuntimeError("DataCube dimensions should be (t,bands, y, x)")

    if cube.data.shape[1] != 14:
        raise RuntimeError(
            "DataCube should have 14 bands exactly (B2, B3, B4, B8, B5, B6, B7, B8A, B11, B12, sunAzimuthAngles, sunZenithAngles, viewAzimuthMean, viewZenithMean)"
        )


def fancy_upsample_function(array: np.ndarray) -> np.ndarray:
    """ """
    return array.repeat(2, axis=-1).repeat(2, axis=-2)

#
# def pvae_batch(
#     s2_x: torch.Tensor,
#     s2_a: torch.Tensor
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     # TODO: use the bands selected for the model …
#
#     s2_x = (
#         s2_x.nan_to_num() / 10000
#     )  # MMDC S2 reflectances are *1000, while PVAE expects [0-1]
#     s2_a = s2_a.nan_to_num()
#     return s2_x, torch.cat(
#         (
#             s2_a[:, 1],  # sun_zen
#             s2_a[:, 3],  # view_zen
#             s2_a[:, 0] - s2_a[:, 2]  # sun_az-view_az
#         ),
#         dim=1,
#     ).nan_to_num()



def run_inference(input_data: np.ndarray) -> np.ndarray:
    """ """
    # First get virtualenv
    sys.path.insert(0, "tmp/extra_venv")
    import torch

    model_file = "models/pvae.torchscript"

    # ONNX inference session options
    # Lecture du modèle et inférence (par pixel donc rearrange)
    ts_net = torch.jit.load(model_file)
    ts_net.eval()
    s2_ref, s2_angles = torch.Tensor(input_data[:, :10]), torch.Tensor(input_data[:, 10:])
    s2_ref = (
            s2_ref.nan_to_num() / 10000
    )  # MMDC S2 reflectances are *1000, while PVAE expects [0-1]
    s2_angles = torch.cat(
        (
            s2_angles[:, 1],  # sun_zen
            s2_angles[:, 3],  # view_zen
            s2_angles[:, 0] - s2_angles[:, 2]  # sun_az-view_az
        ),
        dim=1,
    ).nan_to_num()
    b, c, h, w = s2_ref.shape

    s2_ref_reshape = s2_ref.permute(0, 2, 3, 1).reshape(-1, s2_ref.shape[1])
    s2_angles_reshape = s2_angles.permute(0, 2, 3, 1).reshape(-1, s2_angles.shape[1])
    res = ts_net(s2_ref_reshape, s2_angles_reshape)
    res_shaped = res.reshape(b, h, w, 11, 2).permute(b, 11, 2, h, w).reshape(b, 22, h, w)

    return res_shaped.detach().numpy()
    # return input_data


def apply_datacube(cube: XarrayDataCube, context: Dict) -> XarrayDataCube:
    """
    Apply UDF function to datacube
    """
    # We get data from datacube
    cubearray: xr.DataArray
    if isinstance(cube, xr.DataArray):
        cubearray = cube
    else:
        cubearray = cube.get_array().copy()

    # NOTE: In the apply() process the cubes should always be (bands,y,x)
    # assert cubearray.data.ndim == 4

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
            coords=dict(x=cubearray.coords["x"], y=cubearray.coords["y"]),
        )
    else:

        predicted_array = run_inference(cubearray.data)

        # Build output data array
        # FIX: shape is (t,bands,y,x) on Terrascope backend. This is
        # different in execute_local_udf, issue has been logged.
        predicted_cube = xr.DataArray(
            predicted_array[0],
            dims=["bands", "y", "x"],
            coords=dict(x=cubearray.coords["x"], y=cubearray.coords["y"]),
        )
    return XarrayDataCube(predicted_cube)
