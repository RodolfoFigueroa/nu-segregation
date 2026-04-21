import pickle
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from nu_segregation import ipf
from nu_segregation.constants import LINKING_COLS
from nu_segregation.defs.assets import seg


def reshape_results(results_dict: dict) -> pd.DataFrame:
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
    df_census: pd.DataFrame,
    df_census_geometry: gpd.GeoDataFrame,
    df_survey: pd.DataFrame,
    *,
    out_path: Path,
    q: int = 5,
    write_to_disk: bool = False,
):
    agebs = df_census.index.to_list()

    k_list = [5, 100]

    # Configure out_path variables
    if out_path is not None:
        out_path = Path(out_path)

    # Create results dict
    results_dict = {}

    seed_xr = ipf.generate_contingency_table(df_survey, LINKING_COLS)
    ds = ipf.apply_ipf(df_census, seed_xr)
    df_ind = ipf.generate_individual_weights(df_survey, ds)

    pop_income = ipf.get_income_df(ds, df_census, df_census_geometry, df_ind)

    # Find weighted median
    results_dict["median_MZ"] = ipf.weighted_mean(
        df_ind["Ingreso_orig"].to_numpy(), df_ind["w_MZ"].to_numpy()
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
        C_list.append(C)

    results_dict["cent_idx"] = cent_idx_dict

    C_xr = xr.DataArray(
        data=np.stack(C_list, axis=0),
        coords={
            "income_quantile": list(range(1, q + 1)),
            "ageb": agebs,
            "k_neighbors": list(range(len(agebs))),
        },
    )

    # Return a dataframe
    results = reshape_results(results_dict)
    return results_dict, results, C_xr, C_list

    if write_to_disk:
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
