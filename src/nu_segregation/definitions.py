import dagster as dg
from pathlib import Path
from dagster_components.resources import PostGISResource
from nu_segregation.defs.resources import PathResource


@dg.definitions
def definitions() -> dg.Definitions:
    defs = dg.load_from_defs_folder(project_root=Path(__file__).parent.parent)
    extra_defs = dg.Definitions(
        resources={
            "path_resource": PathResource(data_path=dg.EnvVar("DATA_PATH")),
            "postgis_resource": PostGISResource(
                host=dg.EnvVar("POSTGRES_HOST"),
                port=dg.EnvVar("POSTGRES_PORT"),
                username=dg.EnvVar("POSTGRES_USER"),
                password=dg.EnvVar("POSTGRES_PASSWORD"),
                db=dg.EnvVar("POSTGRES_DB"),
            )
        }
    )
    return dg.Definitions.merge(defs, extra_defs)