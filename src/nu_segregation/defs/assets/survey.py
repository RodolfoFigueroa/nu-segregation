import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from dagster_components.partitions import zone_partitions
from dagster_components.resources import PostGISResource
from dagster_components.utils import cast_all_columns_to_numeric

import dagster as dg
from nu_segregation.defs.resources import PathResource


def fix_folioviv(s: pd.Series) -> pd.Series:
    return s.astype(str).str.zfill(10)


@dg.op
def get_muns_from_zone(
    context: dg.OpExecutionContext, postgis_resource: PostGISResource
) -> list[str]:
    with postgis_resource.connect() as conn:
        df = pd.read_sql(
            """
            SELECT cvegeo FROM census_2020_mun
            WHERE cve_met = %(zone)s
            """,
            conn,
            params={"zone": context.partition_key},
        )
    return df["cvegeo"].tolist()


stems = ["vivienda", "ingresos", "hogares", "poblacion"]


@dg.op(out={stem: dg.Out() for stem in stems})
def extract_enigh_census(
    path_resource: PathResource,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = Path(path_resource.data_path)

    out = []
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        zipfile.ZipFile(data_path / "conjunto_de_datos_enigh_2018_ns_csv.zip") as zf,
    ):
        zf.extractall(tmpdir)
        tmpdir_path = Path(tmpdir)

        for stem in stems:
            subdir_name = f"conjunto_de_datos_{stem}_enigh_2018_ns"
            fpath = next(
                (tmpdir_path / subdir_name / "conjunto_de_datos").glob(
                    "conjunto_de_datos_*.csv"
                )
            )
            out.append(pd.read_csv(fpath))

    return tuple(out)


@dg.op
def process_fol_df(df_fol: pd.DataFrame, muns: list[str]) -> pd.DataFrame:  # noqa: ARG001
    return (
        df_fol[["folioviv", "ubica_geo"]]
        .assign(
            folioviv=lambda df: df["folioviv"].transform(fix_folioviv),
            ubica_geo=lambda df: df["ubica_geo"].astype(str).str.zfill(5),
        )
        .query("ubica_geo in @muns")
    )


@dg.op
def process_income_df(df_income: pd.DataFrame) -> pd.DataFrame:
    return (
        df_income[["folioviv", "foliohog", "numren", "ing_tri"]]
        .assign(folioviv=lambda df: df["folioviv"].transform(fix_folioviv))
        .pipe(cast_all_columns_to_numeric, ignore=["folioviv"], errors="raise")
        .groupby(["folioviv", "foliohog", "numren"])
        .agg(sum)
        .reset_index()
    )


@dg.op
def process_home_df(df_home: pd.DataFrame) -> pd.DataFrame:
    return (
        df_home[["folioviv", "foliohog", "conex_inte"]]
        .assign(folioviv=lambda df: df["folioviv"].transform(fix_folioviv))
        .pipe(cast_all_columns_to_numeric, ignore=["folioviv"], errors="raise")
    )


@dg.op
def process_pop_df(df_pop: pd.DataFrame) -> pd.DataFrame:
    return (
        df_pop[
            [
                "folioviv",
                "foliohog",
                "numren",
                "sexo",
                "edad",
                "edo_conyug",
                "nivelaprob",
                "inst_1",
                "inst_6",
            ]
        ]
        .replace(" ", pd.NA)
        .fillna(0)
        .assign(folioviv=lambda df: df["folioviv"].transform(fix_folioviv))
        .pipe(cast_all_columns_to_numeric, ignore=["folioviv"], errors="raise")
        .query("edad >= 15")
    )


@dg.op
def merge_dfs(
    df_fol: pd.DataFrame,
    df_income: pd.DataFrame,
    df_home: pd.DataFrame,
    df_pop: pd.DataFrame,
) -> pd.DataFrame:
    out = (
        df_fol.merge(df_pop, how="left")
        .merge(df_income, how="left")
        .merge(df_home, how="left")
        .reset_index(drop=True)
        .rename(
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
        .filter(
            [
                "Sexo",
                "Edad",
                "Nivel",
                "SeguroIMSS",
                "SeguroPriv",
                "ConexionInt",
                "Ingreso",
            ],
            axis="columns",
        )
        .assign(
            Sexo=lambda df: (
                df["Sexo"].astype("category").cat.rename_categories({1: "m", 2: "f"})
            ),
            Edad=lambda df: pd.cut(
                df["Edad"],
                bins=[15, 64, 200],
                labels=["p15_64", "p65mas"],
                include_lowest=True,
            ),
            Nivel=lambda df: pd.cut(
                df["Nivel"],
                bins=[-1, 0, 2, 3, 100],
                labels=["ninguno", "primaria", "secundaria", "posbasica"],
            ),
            SeguroIMSS=lambda df: (
                df["SeguroIMSS"]
                .astype("category")
                .cat.rename_categories({0: "no_imss", 1: "imss"})
            ),
            SeguroPriv=lambda df: (
                df["SeguroPriv"]
                .astype("category")
                .cat.rename_categories({0: "no_privado", 6: "privado"})
            ),
            ConexionInt=lambda df: (
                df["ConexionInt"]
                .astype("category")
                .cat.rename_categories({1: "internet", 2: "no_internet"})
            ),
            Ingreso_new=lambda df: pd.qcut(df["Ingreso"], 5, labels=list(range(1, 6))),
        )
        .rename(columns={"Ingreso": "Ingreso_orig"})
        .rename(columns={"Ingreso_new": "Ingreso"})
    )
    out.to_parquet("./survey.parquet")
    return out


@dg.graph_asset(partitions_def=zone_partitions, group_name="survey")
def survey():
    muns = get_muns_from_zone()

    enigh_census_map = extract_enigh_census()

    df_folio = process_fol_df(enigh_census_map[0], muns)
    df_income = process_income_df(enigh_census_map[1])
    df_home = process_home_df(enigh_census_map[2])
    df_pop = process_pop_df(enigh_census_map[3])
    return merge_dfs(df_folio, df_income, df_home, df_pop)
