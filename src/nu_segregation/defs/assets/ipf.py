from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


def weight_ind(ctable, df, column_name="weight"):
    # Appends weight column in place to df
    # Super slow due to inneficient lookups
    # on duplicated multi index

    # Make a df from ctable
    # Should have same multi index as df
    ctable_df = ctable.to_dataframe("counts")
    ctable_df = ctable_df[ctable_df.counts > 0]

    # Create weight column
    df[column_name] = 0

    # Apply weights uniformly distributed on repeated
    # individuals of the same class
    for idx, row in ctable_df.iterrows():
        single_class_df = df.loc[idx]
        n_ind = len(single_class_df)
        df.loc[idx, column_name] = row.counts / n_ind


def weighted_mean(x: np.ndarray, w):
    """Compute the weighted median of x using weights w.

    The function sorts ``x`` and ``w`` by ``x``, normalizes weights to sum to \
    one, and returns the first value where the cumulative weight reaches 0.5.

    Args:
        x: One-dimensional array-like of values.
        w: One-dimensional array-like of non-negative weights aligned with \
            ``x``.

    Returns:
        The weighted median value from ``x``.
    """
    # Sort both arrays acording to x values
    # and normalize weights
    idxs = np.argsort(x)
    x = x[idxs]
    w = w[idxs] / w.sum()

    # Find mid value
    ws = np.cumsum(w)
    idx = ws.searchsorted(0.5)

    return x[idx]


def get_marginals(seed: xr.DataArray, row: pd.Series) -> list[tuple[int, int, float]]:
    """
    Extract marginal values from a pandas Series based on coordinates in a DataArray. \
    This function iterates through the coordinates of a DataArray and matches them \
    against the index of a pandas Series to extract corresponding marginal values.

    Args:
        seed (xr.DataArray): xarray DataArray containing coordinate dimensions to iterate over.
        row (pd.Series): pandas Series with index labels matching the coordinate names \
            in the format "{dimension_name}_{coordinate_value}".

    Returns:
        list[tuple[int, int, float]]: A list of tuples containing:
            - int: dimension index in the DataArray
            - int: position index within the dimension's coordinates
            - float: the corresponding marginal value from the Series

    Example:
        >>> seed = xr.DataArray([1, 2], coords={'x': [0, 1]})
        >>> row = pd.Series({'x_0': 0.5, 'x_1': 0.5})
        >>> get_marginals(seed, row)
        [(0, 0, 0.5), (0, 1, 0.5)]
    """

    marginals_np = []
    for dim, (k, v) in enumerate(seed.coords.items()):
        for i, vv in enumerate([f"{k}_{vv}" for vv in v.to_numpy()]):
            if vv in row.index:
                marginals_np.append((dim, i, row[vv]))
    return marginals_np


def ipf(
    seed: xr.DataArray,
    marginals: list[tuple[int, int, int | float]],
    maxiters: int = 100,
    rel_tol: float = 1e-5,
) -> xr.DataArray:
    """Run iterative proportional fitting on a seed contingency table.

    The algorithm repeatedly rescales slices of ``seed`` so their sums match \
    the provided marginal constraints, stopping when the maximum relative error \
    across constrained marginals is below ``rel_tol``.

    Args:
        seed: Initial contingency table to be fitted.
        marginals: Marginal constraints as ``(dim, index, target_count)`` \
            tuples, where ``dim`` is the axis in ``seed``, ``index`` is the \
            coordinate position on that axis, and ``target_count`` is the \
            desired marginal total.
        maxiters: Maximum number of IPF iterations.
        rel_tol: Convergence tolerance for the maximum relative marginal error.

    Returns:
        xarray DataArray with the same shape and coordinates as ``seed`` and \
        fitted cell counts.

    Raises:
        ValueError: If the algorithm does not converge within ``maxiters``.
    """
    ctable = seed.to_numpy().copy()
    converged = False

    for _ in range(maxiters):
        for constraint in marginals:
            dim, i, m_count = constraint
            idxs = [slice(None)] * ctable.ndim
            idxs[dim] = i  # ty:ignore[invalid-assignment]
            idxs = tuple(idxs)

            c_count = ctable[idxs].sum()
            if c_count == 0:
                continue
            ctable[idxs] *= m_count / c_count

        # Evaluate convergence
        delta = 0.0
        for constraint in marginals:
            dim, i, m_count = constraint
            idxs = [slice(None)] * ctable.ndim
            idxs[dim] = i  # ty:ignore[invalid-assignment]
            idxs = tuple(idxs)

            if m_count == 0:
                pass
            else:
                c_count = ctable[idxs].sum()
                delta = max(delta, abs(1 - c_count / m_count))
        if delta <= rel_tol:
            converged = True
            break

    if not converged:
        err = f"IPF did not converge after {maxiters} iterations. Final delta: {delta}"
        raise ValueError(err)

    return seed.copy(data=ctable)


def apply_ipf(df_censo: pd.DataFrame, seed: xr.DataArray) -> xr.DataArray:
    """Apply IPF to each census row and stack results into one DataArray.

    The function iterates over each row in df_censo, builds marginal \
    constraints for that geographic unit using seed coordinates, and runs \
    iterative proportional fitting. Each fitted contingency table is then \
    concatenated along a new cvegeo dimension.

    Args:
        df_censo: Census dataframe indexed by geographic key (for example, \
            cvegeo) with columns used to derive marginals.
        seed: Seed contingency table as an xarray DataArray.

    Returns:
        xarray DataArray containing one fitted contingency table per \
        geographic unit, with an added cvegeo dimension. Coordinate values \
        include all identifiers from df_censo.index plus a final "seed" \
        entry with the original seed table.

    Raises:
        ValueError: If IPF does not converge for any row.
    """
    ageb_dict = {}

    for cvegeo, row in df_censo.iterrows():
        marginals = get_marginals(seed, row)
        ctable = ipf(seed, marginals)
        ageb_dict[cvegeo] = ctable

    ageb_dict["seed"] = seed
    return xr.concat(ageb_dict.values(), pd.Index(ageb_dict.keys(), name="cvegeo"))


def weight_ind_fast(df: pd.DataFrame, ds: xr.DataArray) -> pd.DataFrame:
    """Build per-area individual weights from fitted contingency tables.

    For each combination of linking variables in ``df.index``, this function \
    reads the corresponding fitted counts from each contingency table in \
    ``ds`` and distributes each count uniformly across survey individuals that \
    share that same index combination.

    Args:
        df: Survey individuals indexed by the same linking-variable MultiIndex \
            used to build the IPF contingency tables.
        ds: Dataset of fitted contingency tables keyed by area identifier \
            (for example, ``cvegeo``).

    Returns:
        pandas DataFrame containing all original columns from ``df`` plus one \
        weight column per variable in ``ds``. Each added column stores the \
        individual weight contribution for that area.
    """
    # Create multiindex dataframe from contingency tables with
    # counts for each variable combination
    ctables_df = pd.concat(
        [
            subarr.squeeze(dim="cvegeo", drop=True).to_dataframe(cvegeo)
            for cvegeo, subarr in ds.groupby("cvegeo")
            if cvegeo != "seed"
        ],
        axis=1,
    )

    # Drop zero counts which are missing from survey
    ctables_df = ctables_df[ctables_df.sum(axis=1) > 0]

    # Ageb list
    columns = ctables_df.columns

    # Add zeroed columns for all agebs and
    # Create the weight table
    df_w = pd.concat(
        [
            df,
            pd.DataFrame(
                data=np.zeros((len(df), len(columns))),
                index=df.index,
                columns=columns,
            ),
        ],
        axis=1,
    )

    # Assign appropriate weight to individuals
    # for each unique combination of variables
    for idx, row in ctables_df.iterrows():
        # This is probably much faster using integer indexing

        # A small dataframe with individuals with the same
        # combination of variables
        # single_class_df = df_w.loc[idx]
        n_ind = len(df_w.loc[idx])

        # Distribute weight uniformly among all individuals
        df_w.loc[idx, columns] = np.broadcast_to(row.values / n_ind, (n_ind, len(row)))
        # df_w.loc[idx] = single_class_df

    return df_w


def get_income_df(
    ds: xr.Dataset,
    df_censo: pd.DataFrame,
    df_ind: pd.DataFrame,
    data_path: Path,
    agebs: list[str],
) -> pd.DataFrame:
    dim = [d for d in ds.dims if d != "Ingreso"]

    pop_income = pd.concat(
        [
            ctable.sum(dim=dim).to_dataframe(name=cvegeo).transpose()
            for cvegeo, ctable in ds.items()
            if cvegeo in agebs
        ]
    ).assign(total_ipf=lambda df: df.sum(axis=1))

    pop_income.index.name = "cvegeo"
    pop_income = pop_income.join(df_censo["P_15YMAS"]).rename(columns={"P_15YMAS": "total_census"})

    income_by_ageb = (
        df_ind[agebs]
        .multiply(df_ind["Ingreso_orig"], axis="index")
        .sum()
        .rename("income")
    )
    pop_income = pop_income.join(income_by_ageb).assign(
        income_pc=lambda df: df["income"] / df["total_ipf"]
    )

    # Import geo data
    scodes = np.unique([a[:2] for a in agebs])
    agebs_gdf = (
        pd.concat(
            [
                gpd.read_file(f"{data_path}/agebs.zip", layer=f"{scode}a")
                for scode in scodes
            ],
        )
        .pipe(gpd.GeoDataFrame, crs="EPSG:6372", geometry="geometry")
        .pipe(lambda df: df.to_crs(df.estimate_utm_crs()))
    )
    agebs_gdf.columns = [i.lower() for i in agebs_gdf.columns]
    agebs_gdf = agebs_gdf.set_index("cvegeo")[["geometry"]]

    return (
        agebs_gdf.join(pop_income, how="right")
        .reset_index()
        .rename(columns={i: f"q_{i}" for i in range(10)})
    )
