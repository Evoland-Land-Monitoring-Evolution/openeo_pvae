"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = openeo_superresolution.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import openeo

from openeo_superresolution import __version__

__author__ = "Julien Michel"
__copyright__ = "Julien Michel"
__license__ = "AGPL-3.0-or-later"

_logger = logging.getLogger(__name__)

# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from openeo_superresolution.run import fib`,
# when using this Python module as a library.


def default_bands_list():
    return ['B02', 'B03', 'B04', 'B08']


@dataclass(frozen=True)
class Parameters:
    spatial_extent: Dict[str, float]
    start_date: str
    end_date: str
    output_file: str
    bands: List[str] = field(default_factory=default_bands_list)
    max_cloud_cover: int = 30
    openeo_instance: str = "openeo-dev.vito.be"
    patch_size: int = 256
    overlap: int = 20


def process(parameters: Parameters) -> None:
    """
    Main processing function
    """
    # First connect to OpenEO instance
    connection = openeo.connect(parameters.openeo_instance).authenticate_oidc()

    # Search for the S2 datacube
    s2_cube = connection.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=parameters.spatial_extent,
        temporal_extent=[parameters.start_date, parameters.end_date],
        bands=parameters.bands,
        max_cloud_cover=parameters.max_cloud_cover)

    udf_file = os.path.join(os.path.dirname(__file__), 'udf.py')
    udf = openeo.UDF.from_file(udf_file, runtime='Python-Jep')

    # Process the cube with the UDF
    sisr_s2_cube = s2_cube.apply_neighborhood(udf,
                                              size=[{
                                                  'dimension': 'x',
                                                  'value':
                                                  parameters.patch_size,
                                                  'unit': 'px'
                                              }, {
                                                  'dimension': 'y',
                                                  'value':
                                                  parameters.patch_size,
                                                  'unit': 'px'
                                              }],
                                              overlap=[])

    download_job = sisr_s2_cube.save_result('NetCDF').create_job(
        title='s2-sisr')
    download_job.start_and_wait()


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description=
        "Run Single-Image Super-Resolution on OpenEO and download results")
    parser.add_argument(
        "--version",
        action="version",
        version=f"openeo_superresolution {__version__}",
    )
    parser.add_argument('--start_date',
                        help="Start date (format: YYYY-MM-DD)",
                        type=str,
                        required=True)
    parser.add_argument('--end_date',
                        help="End date (format: YYYY-MM-DD)",
                        type=str,
                        required=True)
    parser.add_argument(
        '--extent',
        type=float,
        nargs=4,
        help='Extent (west lat, east lat, south lon, north lon)',
        required=True)
    parser.add_argument('--output',
                        type=str,
                        help='Path to ouptput NetCDF file',
                        required=True)

    return parser.parse_args(args)


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)

    # Build parameters
    parameters = Parameters(spatial_extent={
        'west': args.extent[0],
        'east': args.extent[1],
        'south': args.extent[2],
        'north': args.extent[3]
    },
                            start_date=args.start_date,
                            end_date=args.end_date,
                            output_file=args.output)

    _logger.info(f'Parameters : {parameters}')
    process(parameters)


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m openeo_superresolution.skeleton 42
    #
    run()
