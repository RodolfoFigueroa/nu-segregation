import dagster as dg


class PathResource(dg.ConfigurableResource):
    data_path: str
