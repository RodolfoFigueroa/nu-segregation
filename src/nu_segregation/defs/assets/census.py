import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dagster_components.resources import PostGISResource
from dagster_components.utils import cast_all_columns_to_numeric

import dagster as dg


@dg.asset
def census(
    context: dg.AssetExecutionContext, postgis_resource: PostGISResource
) -> pd.DataFrame:
    with postgis_resource.connect() as conn:
        return (
            pd.read_sql(
                """
            SELECT
                "CVEGEO",
                "POBTOT",
                "P_15YMAS",
                "P_15YMAS_F",
                "P_15YMAS_M",
                "POB15_64",
                "POB65_MAS",
                "P15YM_SE",
                "P15PRI_IN",
                "P15PRI_CO",
                "P15SEC_IN",
                "P15SEC_CO",
                "P18YM_PB",
                "PDER_IMSS",
                "PAFIL_IPRIV",
                "P12YM_SOLT",
                "P12YM_CASA",
                "P12YM_SEPA",
                "VPH_INTER",
                "PROM_OCUP",
            FROM census_2020_ageb
            INNER_JOIN census_2020_mun
                ON census_2020_ageb."CVE_MUN" = census_2020_mun."CVEGEO"
            WHERE census_2020_mun."CVE_MET" = %(zone)s
            """,
                conn,
                params={"zone": context.partition_key},
            )
            .pipe(cast_all_columns_to_numeric, ignore=["CVEGEO"])
            .query("P_15YMAS > 20")
        )


def load_census(data_path: os.PathLike, met_zone_codes: list[int]) -> pd.DataFrame:
    data_path = Path(data_path)

    cols = [
        "ENTIDAD",
        "MUN",
        "LOC",
        "NOM_MUN",
        "NOM_LOC",
        "AGEB",
        "POBTOT",
        "P_15YMAS",
        "P_15YMAS_F",
        "P_15YMAS_M",
        "POB15_64",
        "POB65_MAS",
        "P15YM_SE",
        "P15PRI_IN",
        "P15PRI_CO",
        "P15SEC_IN",
        "P15SEC_CO",
        "P18YM_PB",
        "PDER_IMSS",
        "PAFIL_IPRIV",
        "P12YM_SOLT",
        "P12YM_CASA",
        "P12YM_SEPA",
        "VPH_INTER",
        "PROM_OCUP",
    ]

    # Split state codes and mun codes as required for census data
    s_codes = defaultdict(list)
    for c in met_zone_codes:
        s_code = c // 1000
        mun_code = c % 1000
        s_codes[s_code].append(mun_code)

    census_paths = [
        data_path / f"census/RESAGEBURB_{scode:02d}_2020_csv.zip" for scode in s_codes
    ]

    df_list = [
        pd.read_csv(cpath, usecols=cols, na_values=["*", "N/D"], low_memory=False)
        for cpath in census_paths
    ]

    # Keep only aggregates by AGEB
    df_list = [
        df[df["NOM_LOC"] == "Total AGEB urbana"].reset_index(drop=True)
        for df in df_list
    ]

    # Filter by state and met zone
    df_list: list[pd.DataFrame] = [
        df[(df["ENTIDAD"] == scode) & (df["MUN"].isin(s_codes[scode]))].copy()
        for df, scode in zip(df_list, s_codes.keys(), strict=True)
    ]

    # Create CVEGEO column, and drop columns no longer useful
    df_censo = (
        pd.concat(df_list, ignore_index=True)
        .assign(
            cvegeo=lambda df: df.apply(
                lambda x: f"{x.ENTIDAD:02}{x.MUN:03}{x.LOC:04}{x.AGEB.zfill(4)}",
                axis=1,
            )
        )
        .drop(columns=["ENTIDAD", "MUN", "NOM_LOC", "LOC", "AGEB", "NOM_MUN"])
        .dropna()
    )

    # Remove null values and make integer, except fractional counts
    int_cols = df_censo.columns.drop(["PROM_OCUP", "cvegeo"]).copy()
    df_censo = df_censo.astype(dict.fromkeys(int_cols, int))

    # Remove AGEBS with less than 20 in working population
    df_censo = df_censo.query("P_15YMAS > 20")

    # Build linking variables ###

    # Indicator variable for internet for the working population
    # Useful vars:
    #   - PROM_OCU: Promedio de ocupantes en viviendas particulares habitadas
    #   - VPH_INTER: Viviendas particulares habitadas que disponen de internet
    has_internet = df_censo["PROM_OCUP"] * df_censo["VPH_INTER"]
    has_internet_working = has_internet * df_censo["P_15YMAS"] / df_censo["POBTOT"]
    has_internet_working = has_internet_working.astype(int)
    df_censo["ConexionInt_internet"] = has_internet_working
    df_censo["ConexionInt_no_internet"] = (
        df_censo["P_15YMAS"] - df_censo["ConexionInt_internet"]
    )
    # Drop VPH_INTER, no longer used
    df_censo = df_censo.drop(columns=["VPH_INTER", "PROM_OCUP"])

    # Discretize education related variables
    #  - P15YM_SE: Población de 15 años y más sin escolaridad
    #  - P15PRI_IN: Población de 15 años y más con primaria incompleta
    #  - P15PRI_CO: Población de 15 años y más con primaria completa
    #  - P15SEC_IN: Población de 15 años y más con secundaria incompleta
    df_censo["Nivel_ninguno"] = df_censo["P15YM_SE"] + df_censo["P15PRI_IN"]
    df_censo["Nivel_primaria"] = df_censo["P15PRI_CO"] + df_censo["P15SEC_IN"]
    df_censo = df_censo.drop(
        columns=["P15YM_SE", "P15PRI_IN", "P15PRI_CO", "P15SEC_IN"],
    )
    # Rename variables
    df_censo = df_censo.rename(
        {"P15SEC_CO": "Nivel_secundaria", "P18YM_PB": "Nivel_posbasica"},
        axis=1,
    )
    # Make sure all education counts equal the total working population (>15)
    # In order to this, add missing counts to posbasic educaction assuming
    # they correspond to people < 18 with posbasic edu
    missing_posbasic = df_censo["P_15YMAS"] - (
        df_censo["Nivel_primaria"]
        + df_censo["Nivel_secundaria"]
        + df_censo["Nivel_ninguno"]
        + df_censo["Nivel_posbasica"]
    )
    df_censo["Nivel_posbasica"] = df_censo["Nivel_posbasica"] + missing_posbasic

    # Rename variables for working population,
    # which is the population we are interested in
    df_censo = df_censo.rename(
        {
            "P_15YMAS_F": "Sexo_f",
            "P_15YMAS_M": "Sexo_m",
            "POB15_64": "Edad_p15_64",
            "POB65_MAS": "Edad_p65mas",
        },
        axis=1,
    )

    # Marital Status
    df_censo = df_censo.rename(
        columns={
            "P12YM_SOLT": "EstadoConyu_soltera",
            "P12YM_CASA": "EstadoConyu_casada",
            "P12YM_SEPA": "EstadoConyu_separada",
        },
    )
    # Adjust counts assimung almost all 12-14 are single
    diff = (
        df_censo["EstadoConyu_soltera"]
        + df_censo["EstadoConyu_casada"]
        + df_censo["EstadoConyu_separada"]
        - df_censo["P_15YMAS"]
    )
    df_censo["EstadoConyu_soltera"] = df_censo["EstadoConyu_soltera"] - diff

    # Health insurance
    df_censo["SeguroIMSS_imss"] = (
        (df_censo["PDER_IMSS"] / df_censo["POBTOT"]) * df_censo["P_15YMAS"]
    ).astype(int)
    df_censo["SeguroPriv_privado"] = (
        (df_censo["PAFIL_IPRIV"] / df_censo["POBTOT"]) * df_censo["P_15YMAS"]
    ).astype(int)
    df_censo["SeguroIMSS_no_imss"] = df_censo["P_15YMAS"] - df_censo["SeguroIMSS_imss"]
    df_censo["SeguroPriv_no_privado"] = (
        df_censo["P_15YMAS"] - df_censo["SeguroPriv_privado"]
    )
    df_censo = df_censo.drop(columns=["PDER_IMSS", "PAFIL_IPRIV"])

    df_censo = df_censo.drop(columns="POBTOT")

    # Reorder cols
    cols = sorted(df_censo.columns.drop(["P_15YMAS", "cvegeo"]))
    df_censo = df_censo[["cvegeo", "P_15YMAS", *cols]]

    # Assert total counts equal total working pop
    prefixes = [c.split("_")[0] for c in cols if "_" in c]
    prefixes = np.unique(prefixes)
    for prefix in prefixes:
        pcols = [c for c in cols if prefix in c]

        if not (df_censo[pcols].sum(axis=1) == df_censo.P_15YMAS).all():
            err = f"Counts for {prefix} do not sum up to total working population."
            raise ValueError(err)

    # Set index to cvegeo
    return df_censo.set_index("cvegeo")
