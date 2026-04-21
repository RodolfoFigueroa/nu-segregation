from dagster_components.partitions import zone_partitions

import dagster as dg

year_partitions = dg.StaticPartitionsDefinition(["2020"])
year_and_zone_partitions = dg.MultiPartitionsDefinition(
    {"year": year_partitions, "zone": zone_partitions}
)
