import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.special import xlogy
from sklearn.neighbors import KDTree


def binary_entropy(p):
    """Compute binary entropy elementwise for probabilities in p.

    Entropy is defined as ``-p*log2(p) - (1-p)*log2(1-p)`` for values in the \
    open interval ``(0, 1)``. Values outside that interval return ``0``.

    Args:
        p: Scalar or array-like of probabilities.

    Returns:
        numpy.ndarray: Array of binary entropy values with the same shape as \
        ``p`` after conversion with ``numpy.asanyarray``.
    """
    p = np.asanyarray(p)
    e = np.zeros_like(p)
    mask = np.logical_and(p > 0, p < 1)
    pm = p[mask]
    e[mask] = -pm * np.log2(pm) - (1 - pm) * np.log2(1 - pm)
    return e


def local_binary_KL(p_local, p_global):
    """Compute local binary KL divergence against a global probability.

    The divergence is computed elementwise as:
    ``KL(p_local || p_global) = p_local*log2(p_local/p_global) + \
    (1-p_local)*log2((1-p_local)/(1-p_global))``.

    Args:
        p_local: Scalar or array-like of local probabilities.
        p_global: Scalar or array-like of reference global probabilities, \
            broadcastable to ``p_local``.

    Returns:
        numpy.ndarray: Elementwise KL divergence values in base-2 units.
    """
    p_local = np.asarray(p_local)
    p_global = np.asarray(p_global)

    KL = xlogy(p_local, p_local / p_global)
    # KL = p_local*np.log2(p_local/p_global)
    KL += xlogy(1 - p_local, (1 - p_local) / (1 - p_global))
    # KL += (1 - p_local)*np.log2((1 - p_local)/(1 - p_global))

    return KL / np.log(2)


def global_H_index(df_ind: pd.DataFrame, agebs: list[str]):
    """Compute the global segregation profile and global H index.

    The function builds the income distribution for each area (``p(y|n)``), \
    derives cumulative distributions ``F(y|n)``, computes local binary KL \
    divergences relative to the metropolitan distribution, and then aggregates \
    them using area population shares. The final global index ``H`` is the \
    numerical integral of the expected KL profile over the global income CDF.

    Args:
        df_ind: Individual-level weighted table containing at least \
            ``Ingreso_orig``, ``w_MZ``, and one column per area in ``agebs``. \
            Area columns are interpreted as weights by area.
        agebs: List of area identifiers (column names in ``df_ind``) to include \
            in the segregation computation.

    Returns:
        tuple: A 5-element tuple with:
            - float: Global segregation index ``H``.
            - pandas.DataFrame: ``df_cdf`` with cumulative income distributions \
                for each area and ``w_MZ``.
            - pandas.Series: ``norm_H_series``, expected KL normalized by binary \
                entropy of the global CDF (excluding the last point).
            - pandas.Series: ``mean_kl_series``, expected KL across areas for \
                each global percentile (excluding the last point).
            - pandas.DataFrame: ``local_kl`` by area, indexed by global CDF \
                values (excluding the final point where CDF equals 1).
    """
    # The probability distribution of income for each ageb, p(y|n)
    # with n indexing agebs
    df_prob = df_ind.reset_index(drop=True).groupby("Ingreso_orig").sum()
    df_prob = df_prob / df_prob.sum()

    # The cdf F(y|n) = \sum_{y'=0}^y p(y|n)
    df_cdf = df_prob.cumsum()

    # The local deviations for each ageb, for all percentiles of global
    # income distribution, (E(p) - E(p_n))/E(p)
    # local_deviations = df_cdf[agebs].apply(
    #     lambda x: local_bin_normalized_dev(x, df_cdf.w_MZ))
    local_kl = df_cdf[agebs].apply(lambda x: local_binary_KL(x, df_cdf.w_MZ))

    # Since the last row corresponds to F(y) = 1,
    # and reflects a single group, we drop it
    # local_deviations.drop(local_deviations.tail(1).index, inplace=True)
    local_kl = local_kl.drop(local_kl.tail(1).index)

    # The fraction population of each ageb are the
    # probabilities p(n)
    pn = df_ind[agebs].sum() / df_ind.w_MZ.sum()

    # The entropy indices for each percentile are a weighted mean
    # of local deviations
    # entropy_index_df = local_deviations.multiply(pn).sum(axis=1)
    mean_kl_series = local_kl.multiply(pn).sum(axis=1)
    norm_H_series = mean_kl_series / binary_entropy(df_cdf.w_MZ.values[:-1])

    # But the above is not the function to integrate,
    # we must multiply by E(p) to recovr the
    # expected KL divergene.
    # Reme,ber remove last row
    # kl_df = entropy_index_df * binary_entropy(df_cdf.w_MZ.values[:-1])

    # Since the flutuations have been attenuated by the values of the
    # global entropy , it seems safe to integrate numerically the KL
    # function directly, despite the high level of noise at the tails
    # of H (see plots)
    H = integrate.simpson(y=mean_kl_series.values, x=df_cdf.w_MZ.values[:-1])

    # Return the cdf, the local_h, the expected kl, and H
    return (
        H,
        df_cdf,
        norm_H_series,
        mean_kl_series,
        local_kl.set_index(df_cdf.w_MZ.iloc[:-1]),
    )


def local_cent(
    gdf: gpd.GeoDataFrame, x_name: str = "q_5", total_name: str = "total_ipf"
):
    """Compute local centralization indices across all neighborhood scales.

    For each location, the function orders all other locations by centroid \
    distance and computes cumulative centralization values for progressively \
    larger neighbor sets.

    Args:
        gdf: GeoDataFrame with point or polygon geometry and population \
            columns.
        x_name: Column name for the subgroup count of interest.
        total_name: Column name for total population count.

    Returns:
        tuple: A 3-element tuple with:
                        - numpy.ndarray: ``C`` matrix of centralization indices with shape \
              ``(n, n)``.
            - numpy.ndarray: ``nlist`` neighbor indices returned by KDTree.
            - numpy.ndarray: ``dlist`` neighbor distances returned by KDTree.
    """
    # Get centroids as an array of x,y points
    # build and get sorted neighbors lists
    xp = gdf["geometry"].centroid.x.to_numpy()[:, None]
    yp = gdf["geometry"].centroid.y.to_numpy()[:, None]
    points = np.hstack([xp, yp])
    tree = KDTree(points)
    dlist, nlist = tree.query(
        points,
        k=len(points),
        sort_results=True,
        return_distance=True,
    )

    # Get an array of population counts for the required quantile
    totals_list = gdf[total_name].to_numpy()
    x_list = gdf[x_name].to_numpy()
    y_list = totals_list - x_list

    # Create array to hold cent indices
    n = len(x_list)
    C = np.zeros((n, n))

    for i in range(n):
        # For location i, we need to sort the vectors
        i_idxs = nlist[i]
        x = x_list[i_idxs].cumsum()
        y = y_list[i_idxs].cumsum()

        # Get the cumulative populations
        XY = x * y

        # The shifted products
        x_j_1_y_j = x[:-1] * y[1:]
        x_j_y_j_1 = x[1:] * y[:-1]

        # The shifted cumsums
        X_j_1_Y_j = x_j_1_y_j.cumsum()
        X_j_Y_j_1 = x_j_y_j_1.cumsum()

        # The index array for all scales
        for k in range(1, len(x)):
            C[i, k] = (X_j_1_Y_j[k - 1] - X_j_Y_j_1[k - 1]) / XY[k]

    return C, nlist, dlist
