#!/usr/bin/env python

# Copyright: (c) 2024 CESBIO / Centre National d'Etudes Spatiales
"""
Provide the user-defined function to call prosailVAE model
for Sentinel-2 data embedding
"""
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
    """Apply metadata"""
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
            "DataCube should have 13 bands exactly (B2, B3, B4, B8, B5, B6, B7, B8A, B11, B12, "
            "sunZenithAngles, viewZenithAngles, relativeAzimuthAngles)"
        )


def run_inference(input_data: np.ndarray) -> np.ndarray:
    """
    Inference function for Sentinel-2 embeddings with prosailVAE.
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

    cube_collection = run_inference(cubearray.data)

    predicted_cube = xr.DataArray(
        cube_collection,
        dims=cubearray.dims,
        coords=dict(t=cubearray.coords["t"], x=cubearray.coords["x"], y=cubearray.coords["y"]),
    )
    return XarrayDataCube(predicted_cube)
