import geopandas as gpd
import pandas as pd

import dagster as dg
from nu_segregation import ipf
from nu_segregation.constants import LINKING_COLS


@dg.asset(
    key="income",
    ins={
        "df_census": dg.AssetIn(key="census"),
        "df_survey": dg.AssetIn(key="survey"),
        "df_census_geometry": dg.AssetIn(key="census_geometries"),
    },
    io_manager_key="dataframe_postgres_manager",
    group_name="income",
    metadata={
        "primary_key": "cvegeo",
        "table_name": "income_2020",
        "foreign_keys": [
            {
                "column": "cvegeo",
                "ref_table": "census_2020_ageb",
                "ref_column": "cvegeo",
            }
        ],
    },
)
def income_quantiles(
    context: dg.AssetExecutionContext,
    df_census: dict[str, pd.DataFrame],
    df_survey: dict[str, pd.DataFrame],
    df_census_geometry: dict[str, gpd.GeoDataFrame],
) -> pd.DataFrame:
    print(df_census)
    print(df_survey)
    print(df_census_geometry)

    seed_xr = ipf.generate_contingency_table(df_survey, LINKING_COLS)
    ds = ipf.apply_ipf(df_census, seed_xr)
    df_ind = ipf.generate_individual_weights(df_survey, ds)

    return ipf.get_income_df(ds, df_census, df_census_geometry, df_ind)[
        ["cvegeo", "income", "income_pc"]
    ]
