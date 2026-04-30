import geopandas as gpd
import numpy as np
from sklearn.neighbors import KDTree


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
