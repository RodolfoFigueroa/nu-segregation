import dagster as dg
from dagster_components.resources import PostGISResource
import os
from pathlib import Path
import pandas as pd
from nu_segregation.defs.resources import PathResource
import zipfile
import tempfile


def extract_and_read_op_factory(top_dir_path: str, fname: str) -> dg.OpDefinition:
    @dg.op
    def _op(path_resource: PathResource) -> pd.DataFrame:
        with z

    return _op


@dg.asset
def survey(
    path_resource: PathResource, met_zone_codes: list, linking_cols: list[str], q: int
) -> pd.DataFrame:
    data_path = Path(data_path)

    # Folio de vivienda
    path_folio = data_path / "survey/enigh2018_ns_viviendas_csv.zip"
    # Ingreso
    path_ing = data_path / "survey/enigh2018_ns_ingresos_csv.zip"
    # conjunto de hogares para obtener la info de conexión a internet
    path_hog = data_path / "survey/enigh2018_ns_hogares_csv.zip"
    # Información demogŕafica
    path_pob = data_path / "survey/enigh2018_ns_poblacion_csv.zip"

    df_folio = pd.read_csv(path_folio, usecols=["folioviv", "ubica_geo"])

    df_ing = pd.read_csv(
        path_ing,
        usecols=["folioviv", "foliohog", "numren", "ing_tri"],
        dtype={
            "folioviv": int,
            "foliohog": int,
            "numren": int,
            "ing_tri": pd.Float64Dtype(),
        },
    )

    df_hog = pd.read_csv(
        path_hog,
        usecols=["folioviv", "foliohog", "conex_inte"],
        dtype={"folioviv": int, "foliohog": int, "conex_inte": pd.Int64Dtype()},
    )

    df_poblacion = pd.read_csv(
        path_pob,
        usecols=[
            "folioviv",
            "foliohog",
            "numren",
            "sexo",
            "edad",
            "edo_conyug",
            "nivelaprob",
            "inst_1",
            "inst_6",
        ],
        na_values=[" "],
        dtype={
            "folioviv": int,
            "foliohog": int,
            "numren": int,
            "sexo": pd.Int64Dtype(),
            "edad": pd.Int64Dtype(),
            "edo_conyug": pd.Int64Dtype(),
            "nivelaprob": pd.Int64Dtype(),
            "inst_1": pd.Int64Dtype(),
            "inst_6": pd.Int64Dtype(),
        },
    ).fillna(0)

    # Agregate income for duplicate individuals
    ing_agg = df_ing.groupby(["folioviv", "foliohog", "numren"]).agg(sum).reset_index()

    # Filter for state and metropolitan zone
    df_location = df_folio[df_folio.ubica_geo.isin(met_zone_codes)]

    # Filter for working population and add location data
    df_poblacion = df_poblacion[df_poblacion["edad"] >= 15]

    # Merge all dataframes
    df_ind_orig = (
        df_location.merge(df_poblacion, how="left")
        .merge(ing_agg, how="left")
        .merge(df_hog, how="left")
        .reset_index(drop=True)
    )

    # Rename columns to match link and targer variable names
    df_ind_orig = df_ind_orig.rename(
        columns={
            "des_mun": "Municipio",
            "sexo": "Sexo",
            "edad": "Edad",
            "nivelaprob": "Nivel",
            "edo_conyug": "EstadoConyu",
            "inst_1": "SeguroIMSS",
            "inst_6": "SeguroPriv",
            "conex_inte": "ConexionInt",
            "ing_tri": "Ingreso",
        }
    )

    # Keep a df with linking variables, make variables explicitly categorical.
    df_ind = df_ind_orig[[*linking_cols, "Ingreso"]].copy()

    df_ind["Sexo"] = df_ind["Sexo"].astype("category")
    df_ind["Sexo"] = df_ind["Sexo"].cat.rename_categories({1: "m", 2: "f"})

    df_ind["Edad"] = pd.cut(
        df_ind["Edad"],
        bins=[15, 64, 200],
        labels=["p15_64", "p65mas"],
        include_lowest=True,
    )

    df_ind["Nivel"] = pd.cut(
        df_ind["Nivel"],
        bins=[-1, 0, 2, 3, 100],
        labels=["ninguno", "primaria", "secundaria", "posbasica"],
    )

    df_ind["SeguroIMSS"] = df_ind["SeguroIMSS"].astype("category")
    df_ind["SeguroIMSS"] = df_ind["SeguroIMSS"].cat.rename_categories(
        {0: "no_imss", 1: "imss"},
    )

    df_ind["SeguroPriv"] = df_ind["SeguroPriv"].astype("category")
    df_ind["SeguroPriv"] = df_ind["SeguroPriv"].cat.rename_categories(
        {0: "no_privado", 6: "privado"},
    )

    df_ind["ConexionInt"] = df_ind["ConexionInt"].astype("category")
    df_ind["ConexionInt"] = df_ind["ConexionInt"].cat.rename_categories(
        {1: "internet", 2: "no_internet"},
    )

    if "EstadoConyu" in linking_cols:
        df_ind["EstadoConyu"] = pd.cut(
            df_ind["EstadoConyu"],
            bins=[-1, 2, 5, 100],
            labels=["casada", "separada", "soltera"],
        )

    # Find the bin ranges
    df_ind["Ingreso"] = pd.qcut(df_ind["Ingreso"], q, labels=list(range(1, q + 1)))

    df_ind["Ingreso_orig"] = df_ind_orig.Ingreso

    return df_ind
