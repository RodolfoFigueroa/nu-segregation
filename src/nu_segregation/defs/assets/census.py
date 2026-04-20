import geopandas as gpd
import numpy as np
import pandas as pd
from dagster_components.partitions import zone_partitions
from dagster_components.resources import PostGISResource
from dagster_components.utils import cast_all_columns_to_numeric

import dagster as dg


def add_education_cols(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(
            Nivel_ninguno=lambda df: df["p15ym_se"] + df["p15pri_in"],
            Nivel_primaria=lambda df: df["p15pri_co"] + df["p15sec_in"],
        )
        .rename(
            columns={
                "p15sec_co": "Nivel_secundaria",
                "p18ym_pb": "Nivel_posbasica",
            }
        )
        .assign(
            missing_posbasic=lambda df: (
                df["p_15ymas"]
                - df["Nivel_primaria"]
                - df["Nivel_secundaria"]
                - df["Nivel_ninguno"]
                - df["Nivel_posbasica"]
            ),
            Nivel_posbasica=lambda df: df["Nivel_posbasica"] + df["missing_posbasic"],
        )
        .drop(columns=["missing_posbasic"])
    )


def add_internet_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        ConexionInt_internet=lambda df: (
            df["prom_ocup"] * df["vph_inter"] * df["p_15ymas"] / df["pobtot"]
        ).astype(int),
        ConexionInt_no_internet=lambda df: df["p_15ymas"] - df["ConexionInt_internet"],
    )


def add_married_cols(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.rename(
            columns={
                "p12ym_solt": "EstadoConyu_soltera",
                "p12ym_casa": "EstadoConyu_casada",
                "p12ym_sepa": "EstadoConyu_separada",
            }
        )
        .assign(
            missing_married=lambda df: (
                df["EstadoConyu_soltera"]
                + df["EstadoConyu_casada"]
                + df["EstadoConyu_separada"]
                - df["p_15ymas"]
            ),
            EstadoConyu_soltera=lambda df: (
                df["EstadoConyu_soltera"] - df["missing_married"]
            ),
        )
        .drop(columns=["missing_married"])
    )


def add_health_insurance_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(
        SeguroIMSS_imss=lambda df: (
            (df["PDER_IMSS"] / df["POBTOT"]) * df["P_15YMAS"]
        ).astype(int),
        SeguroPriv_privado=lambda df: (
            (df["PAFIL_IPRIV"] / df["POBTOT"]) * df["P_15YMAS"]
        ).astype(int),
        SeguroIMSS_no_imss=lambda df: df["P_15YMAS"] - df["SeguroIMSS_imss"],
        SeguroPriv_no_privado=lambda df: df["P_15YMAS"] - df["SeguroPriv_privado"],
    )


@dg.asset(key="census", partitions_def=zone_partitions)
def census(
    context: dg.AssetExecutionContext, postgis_resource: PostGISResource
) -> pd.DataFrame:
    with postgis_resource.connect() as conn:
        df_censo = (
            pd.read_sql(
                """
            SELECT
                census_2020_ageb.cvegeo,
                census_2020_ageb.pobtot,
                census_2020_ageb.p_15ymas,
                census_2020_ageb.p_15ymas_f,
                census_2020_ageb.p_15ymas_m,
                census_2020_ageb.pob15_64,
                census_2020_ageb.pob65_mas,
                census_2020_ageb.p15ym_se,
                census_2020_ageb.p15pri_in,
                census_2020_ageb.p15pri_co,
                census_2020_ageb.p15sec_in,
                census_2020_ageb.p15sec_co,
                census_2020_ageb.p18ym_pb,
                census_2020_ageb.pder_imss,
                census_2020_ageb.pafil_ipriv,
                census_2020_ageb.p12ym_solt,
                census_2020_ageb.p12ym_casa,
                census_2020_ageb.p12ym_sepa,
                census_2020_ageb.vph_inter,
                census_2020_ageb.prom_ocup
            FROM census_2020_ageb
            INNER JOIN census_2020_mun
                ON census_2020_ageb.cve_mun = census_2020_mun.cvegeo
            WHERE census_2020_mun.cve_met = %(zone)s
            """,
                conn,
                params={"zone": context.partition_key},
            )
            .pipe(cast_all_columns_to_numeric, ignore=["cvegeo"])
            .dropna()
            .loc[lambda df: df["p_15ymas"] > 20]
            .pipe(add_internet_cols)
            .pipe(add_education_cols)
            .pipe(add_married_cols)
            .rename(
                columns={
                    "p_15ymas_f": "Sexo_f",
                    "p_15ymas_m": "Sexo_m",
                    "pob15_64": "Edad_p15_64",
                    "pob65_mas": "Edad_p65mas",
                }
            )
            .drop(
                columns=[
                    "vph_inter",
                    "prom_ocup",
                    "p15ym_se",
                    "p15pri_in",
                    "p15pri_co",
                    "p15sec_in",
                    "pder_imss",
                    "pafil_ipriv",
                    "pobtot",
                ]
            )
            .pipe(cast_all_columns_to_numeric, ignore=["cvegeo"], make_valid_int=True)
        )

    cols = sorted(df_censo.columns.drop(["p_15ymas", "cvegeo"]))
    df_censo = df_censo[["cvegeo", "p_15ymas", *cols]]

    # Assert total counts equal total working pop
    prefixes = [c.split("_")[0] for c in cols if "_" in c]
    prefixes = np.unique(prefixes)
    for prefix in prefixes:
        pcols = [c for c in cols if prefix in c]

        if not (df_censo[pcols].sum(axis=1) == df_censo["p_15ymas"]).all():
            err = f"Counts for {prefix} do not sum up to total working population."
            raise ValueError(err)

    return df_censo


@dg.asset(key=["census_geometries"], partitions_def=zone_partitions)
def census_geometries(postgis_resource: PostGISResource) -> gpd.GeoDataFrame:
    with postgis_resource.connect() as conn:
        gdf = gpd.read_postgis(
            """
            SELECT cvegeo, geometry
            FROM census_2020_ageb
            """,
            conn,
            geom_col="geometry",
        )
    return gdf
