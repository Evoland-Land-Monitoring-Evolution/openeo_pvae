#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
Provide the user-defined function to call prosailVAE model
for Sentinel-2 data embedding
"""
import logging
import sys
from typing import Dict

import numpy as np
import xarray as xr
from openeo.metadata import CollectionMetadata
from openeo.udf import XarrayDataCube

NEW_BANDS = ["F01_mu", "F02_mu", "F03_mu", "F04_mu", "F05_mu", "F06_mu", "F07_mu", "F08_mu", "F09_mu",
             "F10_mu", "F11_mu",
             "F01_logvar", "F02_logvar", "F03_logvar", "F04_logvar", "F05_logvar", "F06_logvar", "F07_logvar",
             "F08_logvar", "F09_logvar", "F10_logvar", "F11_logvar"]


def apply_metadata(metadata: CollectionMetadata, context: dict) -> CollectionMetadata:
    return metadata.rename_labels(
        dimension="band",
        target=NEW_BANDS
    )


def check_datacube(cube: xr.DataArray):
    """ """
    if cube.data.ndim != 4:
        raise RuntimeError("DataCube dimensions should be (t,bands, y, x)")

    if cube.data.shape[1] != 13:
        raise RuntimeError(
            "DataCube should have 14 bands exactly (B2, B3, B4, B8, B5, B6, B7, B8A, B11, B12, sunAzimuthAngles, "
            "sunZenithAngles, viewAzimuthMean, viewZenithMean)"
        )


def run_inference(input_data: np.ndarray) -> np.ndarray:
    """
    Inference function for Sentinel2 embeddings with prosailVAE.
    The input should be in shape (B, C, H, W)
    The output shape is (B, 22, H, W)
    """
    # First get virtualenv
    sys.path.insert(0, "tmp/extra_venv")
    import onnxruntime as ort

    model_file = "tmp/extra_files/prosailvae.onnx"

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
    logging.info(input_data.shape)

    # We transform input data in right format
    s2_ref, s2_angles = input_data[:, :10].astype(np.float32), input_data[:, 10:].astype(np.float32)
    s2_ref = np.nan_to_num(
        s2_ref / 10000
    )  # MMDC S2 reflectances are *10000, while PVAE expects [0-1]

    s2_angles = np.nan_to_num(s2_angles)

    b, c, h, w = s2_ref.shape
    s2_ref_reshape = s2_ref.transpose((0, 2, 3, 1)).reshape(-1, s2_ref.shape[1])
    s2_angles_reshape = s2_angles.transpose((0, 2, 3, 1)).reshape(-1, s2_angles.shape[1])

    input = {"refls": s2_ref_reshape, "angles": s2_angles_reshape}  # (B, N)

    # Get the ouput of the exported model
    res = ort_session.run(None, input, run_options=ro)[0]
    res_shaped = res.reshape(b, h, w, 11, 2).transpose(0, 3, 4, 1, 2).reshape(b, 22, h, w)

    return res_shaped


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

    # if cubearray.data.ndim == 4:
    #     cube_collection = []
    #     # Perform inference for each date
    #     for c in range(len(cubearray.data)):
    #         predicted_array = run_inference(cubearray.data[c][None, :, :, :])
    #         # predicted_array = fancy_upsample_function(cubearray.data[c])
    #         cube_collection.append(predicted_array[0])
    #
    #     # Stack all dates
    #     cube_collection = np.stack(cube_collection, axis=0)
    #
    #     # Build output data array
    #     # FIX: shape is (t,bands,y,x) on Terrascope backend. This is
    #     # different in execute_local_udf, issue has been logged.
    #     predicted_cube = xr.DataArray(
    #         cube_collection,
    #         dims=["t", "bands", "y", "x"],
    #         coords=dict(x=cubearray.coords["x"], y=cubearray.coords["y"]),
    #     )
    # else:
    #
    #     predicted_array = run_inference(cubearray.data[None, :, :, :])
    #
    #     # Build output data array
    #     # FIX: shape is (t,bands,y,x) on Terrascope backend. This is
    #     # different in execute_local_udf, issue has been logged.
    #     predicted_cube = xr.DataArray(
    #         predicted_array[0],
    #         dims=["bands", "y", "x"],
    #         coords=dict(x=cubearray.coords["x"], y=cubearray.coords["y"]),
    #     )

    if cubearray.data.ndim == 4 and cubearray.data.shape[0] != 1:
        cube_collection = run_inference(cubearray.data)
        # Build output data array
        predicted_cube = xr.DataArray(
            cube_collection,
            dims=["t", "bands", "y", "x"],
            coords=dict(t=cubearray.coords["t"], x=cubearray.coords["x"], y=cubearray.coords["y"]),
        )
    else:
        if cubearray.data.shape[0] == 1:
            predicted_array = run_inference(cubearray.data)
        else:
            predicted_array = run_inference(cubearray.data[None, :, :, :])

        # Build output data array
        predicted_cube = xr.DataArray(
            predicted_array[0],
            dims=["bands", "y", "x"],
            coords=dict(x=cubearray.coords["x"], y=cubearray.coords["y"]),
        )
    return XarrayDataCube(predicted_cube)
