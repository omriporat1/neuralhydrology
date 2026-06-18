"""Datasources package.

Base protocol types are re-exported here. Datasource classes must be imported
directly from their submodules to avoid loading optional dependencies at
package initialisation time:

    from src.datasources.mrms import MrmsAwsQpe1hPass1
    from src.datasources.rtma import RtmaAwsConusDataSource

Stage 1 forcing (MRMS + RTMA via AWS S3) requires: boto3, botocore, tenacity, tqdm.
Other datasources (GDAS, GFS, IFS, ERA5-Land, IMERG) have separate optional deps
and must not be imported unless those deps are installed.
"""

from src.datasources.base import CONUS_BBOX, DataSource, DerivedSpec, Region, RemoteObject

__all__ = [
	"CONUS_BBOX",
	"DataSource",
	"DerivedSpec",
	"Region",
	"RemoteObject",
]
