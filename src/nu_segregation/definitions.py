from pathlib import Path

from dagster_components.managers import (
    DataFrameFileManager,
    DataFramePostgresManager,
    GeoDataFrameFileManager,
    GeoDataFramePostGISManager,
)
from dagster_components.resources import PostgresResource

import dagster as dg
from nu_segregation.defs.resources import PathResource


@dg.definitions
def definitions() -> dg.Definitions:
    defs = dg.load_from_defs_folder(project_root=Path(__file__).parent.parent)

    path_resource = PathResource(
        data_path=dg.EnvVar("DATA_PATH"), out_path=dg.EnvVar("OUT_PATH")
    )
    postgres_resource = PostgresResource(
        host=dg.EnvVar("POSTGRES_HOST"),
        port=dg.EnvVar("POSTGRES_PORT"),
        user=dg.EnvVar("POSTGRES_USER"),
        password=dg.EnvVar("POSTGRES_PASSWORD"),
        db=dg.EnvVar("POSTGRES_DB"),
    )

    extra_defs = dg.Definitions(
        resources={
            "path_resource": path_resource,
            "postgres_resource": postgres_resource,
            "dataframe_postgres_manager": DataFramePostgresManager(
                postgres_resource=postgres_resource
            ),
            "geodataframe_postgis_manager": GeoDataFramePostGISManager(
                postgres_resource=postgres_resource
            ),
            "dataframe_manager": DataFrameFileManager(
                path_resource=path_resource, extension=".parquet"
            ),
            "geodataframe_file_manager": GeoDataFrameFileManager(
                path_resource=path_resource, extension=".geoparquet"
            ),
        }
    )
    return dg.Definitions.merge(defs, extra_defs)
