import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nu_segregation.defs.assets import ipf, seg


def reshape_results(results_dict):
    results_reshaped = {
        "median_MZ": [results_dict["median_MZ"]],
        "H": [results_dict["H"]],
    }

    for quantile in results_dict["cent_idx"]:
        for k in results_dict["cent_idx"][quantile]:
            for i, elem in enumerate(results_dict["cent_idx"][quantile][k]):
                key = f"cent_idx.{quantile}.{k}.{i}"
                results_reshaped[key] = [elem]

    return pd.DataFrame.from_dict(results_reshaped)


def get_seg_full(
    df_censo: pd.DataFrame,
    df_survey: pd.DataFrame,
    *,
    out_path: Path,
    q: int = 5,
    data_path: Path = Path("./data/"),
    write_to_disk: bool = False,
):
    k_list = [5, 100]
    linking_cols = [
        "Sexo",
        "Edad",
        "Nivel",
        "SeguroIMSS",
        "SeguroPriv",
        "ConexionInt",
    ]

    # Configure out_path variables
    if out_path is not None:
        out_path = Path(out_path)

    # Create results dict
    results_dict = {}

    # Create the global contingency table/distribution
    seed_xr = (
        pd.crosstab(
            [df_survey[c] for c in linking_cols],
            df_survey["Ingreso"],
            dropna=False,
        )
        .stack()
        .to_xarray()
        .astype(float)
    )

    if not isinstance(seed_xr, xr.DataArray):
        err = "Seed contingency table must be a DataArray. Check the dimensions of the crosstab output."
        raise TypeError(err)

    # Apply ipf to each zone
    ds = ipf.apply_ipf(df_censo, seed_xr)
    agebs = df_censo.index.to_list()

    # Create a df of individuals from survey dataframe
    # with multi index on linking categories
    df_ind = (
        df_survey.set_index(list(df_survey.columns.drop("Ingreso_orig")))
        .sort_index()
        .pipe(
            lambda df: ipf.weight_ind_fast(df, ds)
        )  # Add individual weights for each ageb
        .assign(
            w_MZ=lambda df: df.drop(columns=["Ingreso_orig"]).sum(axis=1)
        )  # Sum weights for all met zone
    )

    # if write_to_disk:
    #     df_ind.to_csv(out_path / "weight_tables.csv")

    # Find weighted median
    results_dict["median_MZ"] = ipf.weighted_mean(
        df_ind.Ingreso_orig.values, df_ind.w_MZ.values
    )

    # Get global segregations and segregation profile
    # H is the global index
    # df_cdf contains the empirical cdf for all agebs (columns)
    # entropy_index_df contains the binary H index for all
    # empirical partitions p/1-p.
    # kl_df the same as above but unormalized, so E[KL] over all agebs
    # local_dev_df: contains local H deviations for all agebs(cols) for
    # all percentiles (rows)
    (H, df_cdf, norm_H_series, mean_kl_series, local_kl) = seg.global_H_index(
        df_ind,
        agebs,
    )

    results_dict["H"] = H
    return results_dict

    # Create a dataframe with population per income bracket per ageb
    # Marginalizing over all other variables in all local
    # contingency tables.

    # Also calculates total and per capita income
    pop_income = ipf.get_income_df(ds, df_censo, df_ind, data_path, agebs)

    # Keep only agebs witg geometry (error in marco geo?)
    agebs = pop_income[~pop_income.geometry.isna()].cvegeo.to_list()
    pop_income = pop_income.dropna()

    # Find local centralization index for top and low percentiles
    cent_idx_dict = {}
    C_list = []
    for qq in range(1, q + 1):
        xname = f"q_{qq}"
        cent_idx_dict[xname] = {}
        C, nlist, dlist = seg.local_cent(pop_income, x_name=xname)

        max_k = C.shape[1] - 1
        for k in k_list:
            if k > max_k:
                k = max_k
                warnings.warn(
                    "k greater than number of entries in DataFrame. Value has been automatically adjusted.",
                    stacklevel=2,
                )
            cent_idx_dict[xname][f"k_{k}"] = C[:, k].copy()

        if write_to_disk:
            C_list.append(C)

    results_dict["cent_idx"] = cent_idx_dict

    # Return a dataframe
    results = reshape_results(results_dict)

    if write_to_disk:
        C_xr = xr.DataArray(
            data=np.stack(C_list, axis=0),
            coords={
                "income_quantile": list(range(1, q + 1)),
                "ageb": agebs,
                "k_neighbors": list(range(len(agebs))),
            },
        )
        n_info = xr.DataArray(
            data=np.stack([nlist, dlist]),
            coords={
                "info": ["n_idx", "n_distance"],
                "ageb": agebs,
                "k_neighbors": list(range(len(agebs))),
            },
        )
        C_ds = xr.Dataset({"centrality": C_xr, "n_info": n_info})
        C_ds.to_netcdf(path=out_path / "centrality_index.nc", engine="netcdf4")

        df_cdf.to_csv(out_path / "ecdf_income_per_ageb.csv")
        norm_H_series.to_csv(out_path / "H_index_per_percentile.csv")
        mean_kl_series.to_csv(out_path / "mean_KL_per_percentile.csv")
        local_kl.to_csv(out_path / "KL_per_ageb_per_pecentile.csv")

        pop_income.to_file(out_path / "income_quantiles.gpkg")

        with open(out_path / "results.pkl", "wb") as f:
            pickle.dump(results, f)

    return results
