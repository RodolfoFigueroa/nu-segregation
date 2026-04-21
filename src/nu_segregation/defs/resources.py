import dagster as dg


class PathResource(dg.ConfigurableResource):
    data_path: str
    out_path: str
