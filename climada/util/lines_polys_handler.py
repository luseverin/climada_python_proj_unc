"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

"""
import logging
import copy
from collections import Counter

import cartopy.crs as ccrs
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import scipy as sp
import shapely as sh
import shapely.geometry as shgeom

from climada.engine import Impact
from climada.util import coordinates as u_coord


LOGGER = logging.getLogger(__name__)


def calc_geom_impact(
        exp, impf_set, haz, res,
        to_meters=False, disagg_met='avg', disagg_val=None, agg_met='sum'):
    """
    Compute impact for exposure with (multi-)polygons and/or (multi-)lines.
    Lat/Lon values in exp.gdf are ignored, only exp.gdf.geometry is considered.

    The geometries are first disaggregated to points. Polygons: grid with
    resolution res*res. Lines: points along the line separated by distance res.
    The impact per point is then re-aggregated for each geometry.

    Parameters
    ----------
    exp : Exposures
        The exposure instance with exp.gdf.geometry containing (multi-)polygons
        and/or (multi-)lines
    impf_set : ImpactFuncSet
        The set of impact functions.
    haz : Hazard
        The hazard instance.
    res : float
        Resolution of the disaggregation grid (polygon) or line (lines).
    to_meters : bool, optional
       If True, res is interpreted as meters, and geometries are projected to
       an equal area projection for disaggregation.  The exposures are
       then projected back to the original projections before  impact
       calculation. The default is False.
    disagg_met : string, optional
        Disaggregation method of the shapes's original value onto its inter-
        polated points. 'avg': Divide the value evenly over all the new points;
        'fix': Replicate the value onto all the new points. Default is 'avg'.
        Works in combination with the kwarg 'disagg_val'.
        The default is 'avg'.
    disagg_val: float, optional
        Specifies from where and what number should be taken as the value, which
        is to be disaggregated according to the method provided in disagg_met.
        None: default. The shape's value is taken from the exp.gdf.value column.
        Will throw an error if no value column is present.
        float: This given number will be disaggregated according to the method.
        The default is None.
    agg_met : string, optional
        Aggregation method of the point impacts into impact for respective
        parent-geometry.
        If 'avg', the impact is averaged over all points in each geometry.
        If 'sum', the impact is summed over all points in each geometry.
        The default is 'sum'.

    Returns
    -------
    Impact
        Impact object with the impact per geometry (rows of exp.gdf). Contains
        two additional attributes 'geom_exp' and 'coord_exp', the first one
        being the origninal line or polygon geometries for which impact was
        computed.

    See Also
    --------
    exp_geom_to_pnt: disaggregate exposures

    """

    # disaggregate exposure
    exp_pnt = exp_geom_to_pnt(
        exp=exp, res=res,
        to_meters=to_meters, disagg_met=disagg_met,
        disagg_val=disagg_val
        )
    exp_pnt.assign_centroids(haz)

    # compute point impact
    impact_pnt = Impact()
    impact_pnt.calc(exp_pnt, impf_set, haz, save_mat=True)

    # re-aggregate impact to original exposure geometry
    impact_agg = impact_pnt_agg(impact_pnt, exp_pnt, agg_met)

    return impact_agg

def impact_pnt_agg(impact_pnt, exp_pnt_gdf, agg_met):
    """
    Aggregate the impact per geometry.

    The output Impact object contains an extra attribute 'geom_exp'
    containing the geometries.

    Parameters
    ----------
    impact_pnt : Impact
        Impact object with impact per exposure point (lines of exp_pnt)
    exp_pnt_gdf : GeoDataFrame
        Geodataframe of an exposures featuring a multi-index. First level indicating
        membership of original geometries, second level the disaggregated points.
        The exposure is obtained for instance with the disaggregation method
        exp_geom_to_pnt().
    agg_met : string, optional
        If 'agg', the impact is averaged over all points in each geometry.
        If 'sum', the impact is summed over all points in each geometry.
        The default is 'sum'.

    Returns
    -------
    impact_agg : Impact
        Impact object with the impact per original geometry. Original geometry
        additionally stored in attribute 'geom_exp'; coord_exp contains only
        representative points (lat/lon) of those geometries.

    See also
    --------
        exp_geom_to_pnt: exposures disaggregation method
    """

    # aggregate impact
    mat_agg = _aggregate_impact_mat(impact_pnt, exp_pnt_gdf, agg_met)

    # write to impact obj
    impact_agg = set_imp_mat(impact_pnt, mat_agg)

    # add exposure representation points as coordinates
    repr_pnts = gpd.GeoSeries(
        exp_pnt_gdf['geometry_orig'][:,0].apply(
            lambda x: x.representative_point()))
    impact_agg.coord_exp = np.array([repr_pnts.y, repr_pnts.x]).transpose()

    # Add original geometries for plotting
    impact_agg.geom_exp = exp_pnt_gdf.xs(0, level=1)\
        .set_geometry('geometry_orig')\
            .geometry.rename('geometry')

    return impact_agg


def _aggregate_impact_mat(imp_pnt, gdf_pnt, agg_met):
    """
    Aggregate point impact matrix given the geodataframe of disaggregated
    geometries.

    Parameters
    ----------
    imp_pnt : Impact
        Impact object with impact per point (rows of gdf_pnt)
    gdf_pnt : GeoDataFrame
        Exposures geodataframe with a multi-index, as obtained from disaggregation
        method exp_geom_to_pnt(). First level indicating
        membership of original geometries, second level the disaggregated points
    agg : string
        If 'agg', the impact is averaged over all points in each polygon.
        If 'sum', the impact is summed over all points in each polygon.

    Returns
    -------
    sparse.csr_matrix
        matrix of shape #events x #original geometries with impacts.

    """

    col_geom = gdf_pnt.index.get_level_values(level=0)
    # Converts string multi-index level 0 to integer index
    col_geom = np.sort(np.unique(col_geom, return_inverse=True)[1])
    row_pnt = np.arange(len(col_geom))
    if agg_met == 'avg':
        geom_sizes = Counter(col_geom).values()
        mask = np.concatenate([np.ones(l) / l for l in geom_sizes])
    elif agg_met == 'sum':
        mask = np.ones(len(col_geom))
    else:
        raise ValueError(f"Please choose a valid aggregation method. {agg_met} is not valid")
    csr_mask = sp.sparse.csr_matrix(
        (mask, (row_pnt, col_geom)),
         shape=(len(row_pnt), len(np.unique(col_geom)))
        )
    return imp_pnt.imp_mat.dot(csr_mask)


def plot_eai_exp_geom(imp_geom, centered=False, figsize=(9, 13), **kwargs):
    """
    Plot the average impact per exposure polygon.

    Parameters
    ----------
    imp_geom : Impact
        Impact instance with imp_geom set (i.e. computed from exposures with polygons)
    centered : bool, optional
        Center the plot. The default is False.
    figsize : (float, float), optional
        Figure size. The default is (9, 13).
    **kwargs : dict
        Keyword arguments for GeoDataFrame.plot()

    Returns
    -------
    ax:
        matplotlib axes instance

    """
    kwargs['figsize'] = figsize
    if 'legend_kwds' not in kwargs:
        kwargs['legend_kwds'] = {
            'label': f"Impact [{imp_geom.unit}]",
            'orientation': "horizontal"
            }
    if 'legend' not in kwargs:
        kwargs['legend'] = True
    gdf_plot = gpd.GeoDataFrame(imp_geom.geom_exp)
    gdf_plot['impact'] = imp_geom.eai_exp
    if centered:
        # pylint: disable=abstract-class-instantiated
        xmin, xmax = u_coord.lon_bounds(imp_geom.coord_exp[:,1])
        proj_plot = ccrs.PlateCarree(central_longitude=0.5 * (xmin + xmax))
        gdf_plot = gdf_plot.to_crs(proj_plot)
    return gdf_plot.plot(column = 'impact', **kwargs)

def exp_geom_to_pnt(exp, res, to_meters, disagg_met, disagg_val):
    """
    Disaggregate exposures with (multi-)polygons and/or (multi-)lines
    geometries to points.

    Parameters
    ----------
    exp : Exposures
        The exposure instance with exp.gdf.geometry containing lines or polygons
    res : float
        Resolution of the disaggregation grid / distance. Can also be a
        tuple of [x_grid, y_grid] numpy arrays. In this case, to_meters is
        ignored. This is only possible for Polygon-only exposures.
    to_meters : bool
       If True, res is interpreted as meters, and geometries are projected to
       an equal area projection for disaggregation.  The exposures are
       then projected back to the original projections before  impact
       calculation. The default is False.
    disagg_met : string
        Disaggregation method for the `value` column of the exposure gdf.
        if 'avg', average value over points
        if 'fix', same value for each point
    disagg_val: float
        if 'None', value is taken from exp.gdf.value column, else the indicated
        value is taken for disaggregation according to disagg_met

    Returns
    -------
    exp_pnt : Exposures
        Exposures with a double index geodataframe, first level indicating
        membership of the original geometries of exp,
        second for the point disaggregation within each geometries.

    """

    gdf_geom = exp.gdf.copy()

    if disagg_val is not None:
        gdf_geom.value = disagg_val

    if ((disagg_val is None) and ('value' not in gdf_geom.columns)):
        raise ValueError('There is no value column in the exposure gdf to'+
                         ' disaggregate from. Please set disagg_val explicitly.')

    if disagg_met not in ['fix', 'avg']:
        raise ValueError(f"{disagg_met} is not a valid argument for 'disagg_met'."+
                         " Please choose one of ['fix', 'avg'] ")

    gdf_pnt = gdf_to_pnts(gdf_geom, res, to_meters, disagg_met)

    # set lat lon and centroids
    exp_pnt = exp.copy()
    exp_pnt.set_gdf(gdf_pnt)
    exp_pnt.set_lat_lon()

    return exp_pnt


def _pnt_line_poly_mask(gdf):
    """Mask for points, lines and polygons"""
    pnt_mask =  gdf.geometry.apply(lambda x: isinstance(x, shgeom.Point))

    line_mask =  gdf.geometry.apply(lambda x: isinstance(x, shgeom.LineString))
    line_mask |=  gdf.geometry.apply(lambda x: isinstance(x, shgeom.MultiLineString))

    poly_mask =  gdf.geometry.apply(lambda x: isinstance(x, shgeom.Polygon))
    poly_mask |=  gdf.geometry.apply(lambda x: isinstance(x, shgeom.MultiPolygon))

    return pnt_mask, line_mask, poly_mask


def gdf_to_pnts(gdf, res, to_meters, disagg_met):
    """
    Disaggregate geodataframe with (multi-)polygons
    geometries to points.

    Parameters
    ----------
    gdf : GeoDataFrame
        Feodataframe instance with gdf.geometry containing (multi)-lines or
        (multi-)polygons. Points are ignored.
    res : float
        Resolution of the disaggregation grid. Can also be a
        tuple of [x_grid, y_grid] numpy arrays. In this case, to_meters is
        ignored.
    to_meters : bool
       If True, the geometries are projected to an equal area projection before
       the disaggregation. res is then in meters. The exposures are
       then reprojected into the original projections before the impact
       calculation.
    disagg_met : string, optional
        Disaggregation method for the `value` column of the exposure gdf.
        if 'avg', average value over points
        if 'fix', same value for each point

    Returns
    -------
    exp_pnt : Exposures
        Exposures with a double index geodataframe, first for the geometries of exp,
        second for the point disaggregation of the geometries.

    """
    if gdf.empty:
        return gdf

    pnt_mask, line_mask, poly_mask = _pnt_line_poly_mask(gdf)

    # Concatenating an empty dataframe with an index  together with
    # a dataframe with a multi-index breaks the multi-index
    gdf_pnt = gpd.GeoDataFrame([])
    if pnt_mask.any():
        gdf_pnt = gpd.GeoDataFrame(pd.concat([
            gdf_pnt,
            gdf[pnt_mask]
        ]))
    if line_mask.any():
        gdf_pnt = gpd.GeoDataFrame(pd.concat([
            gdf_pnt,
            _line_to_pnts(gdf[line_mask], res, to_meters)
        ]))
    if poly_mask.any():
        gdf_pnt = gpd.GeoDataFrame(pd.concat([
            gdf_pnt,
            _poly_to_pnts(gdf[poly_mask], res, to_meters)
        ]))

    # disaggregate value column
    if disagg_met == 'avg':
        gdf_pnt = _disagg_values_avg(gdf_pnt)

    return gdf_pnt


def _disagg_values_avg(gdf_pnts):
    """
    Disaggregate value column of original gdf to disaggregated point gdf

    Parameters
    ----------
    gdf_pnts : geodataframe
        Geodataframe with a double index, first for geometries (lines, polygons),
        second for the point disaggregation of the polygons. The value column is assumed
        to represent values per polygon / line (first index).

    Returns
    -------
    gdf_disagg : geodataframe
        The value per geometry are evenly distributed over the points per geometry.

    """

    gdf_disagg = gdf_pnts.copy()

    group = gdf_pnts.groupby(axis=0, level=0)
    vals = group.value.mean() / group.value.count()

    vals = vals.reindex(gdf_pnts.index, level=0)
    gdf_disagg['value'] = vals

    return gdf_disagg


def _poly_to_pnts(gdf, res, to_meters=False):
    """
    Disaggregate (multi-)polygons geodataframe to points.
    Note: If polygon is smaller than specified resolution, a representative
    point within the polygon will be chosen, nevertheless. This may lead to
    inaccuracies during value assignments / disaggregations.

    Parameters
    ----------
    gdf : geodataframe
        Can have any CRS
    res : float
        Resolution (same units as gdf crs)

    Returns
    -------
    geodataframe
        Geodataframe with a double index, first for polygon geometries,
        second for the point disaggregation of the polygons.

    """

    if gdf.empty:
        return gdf

    # Needed because gdf.explode() requires numeric index
    idx = gdf.index.to_list() #To restore the naming of the index

    gdf_points = gdf.copy().reset_index(drop=True)

    if isinstance(res, tuple):
        x_grid, y_grid = res
        gdf_points['geometry_pnt'] = gdf_points.apply(
            lambda row: _interp_one_poly_grid(row.geometry, x_grid, y_grid), axis=1)
    elif to_meters:
        gdf_points['geometry_pnt'] = gdf_points.apply(
            lambda row: _interp_one_poly_m(row.geometry, res, gdf.crs), axis=1)
    else:
        gdf_points['geometry_pnt'] = gdf_points.apply(
            lambda row: _interp_one_poly(row.geometry, res), axis=1)

    gdf_points = _swap_geom_cols(
        gdf_points, geom_to='geometry_orig', new_geom='geometry_pnt')

    gdf_points = gdf_points.explode()
    gdf_points.index = gdf_points.index.set_levels(idx, level=0)
    return gdf_points


def _interp_one_poly_grid(poly, x_grid, y_grid):
    """
    Disaggregate a single polygon to points on grid (does not have to be a
    regular raster)

    Parameters
    ----------
    poly : shapely Polygon
        Polygon
    x_grid : np.array
        1D array of x-coordinates of grid points.
    y_grid : np.array
        1D array of y-coordinates of grid points.

    Returns
    -------
    shapely multipoint
        Grid of points inside the polygon

    """

    if poly.is_empty:
        return shgeom.MultiPoint([])
    in_geom = sh.vectorized.contains(poly, x_grid, y_grid)

    if sum(in_geom.flatten()) > 1:
        return shgeom.MultiPoint(list(zip(x_grid[in_geom], y_grid[in_geom])))

    LOGGER.warning('Polygon smaller than resolution. Setting a representative point.')
    return shgeom.MultiPoint([poly.representative_point()])


def _interp_one_poly(poly, res):
    """
    Disaggregate a single polygon to points

    Parameters
    ----------
    poly : shapely Polygon
        Polygon
    res : float
        Resolution (same units as gdf crs)

    Returns
    -------
    shapely multipoint
        Grid of points rasterizing the polygon

    """

    if poly.is_empty:
        return shgeom.MultiPoint([])

    height, width, trafo = u_coord.pts_to_raster_meta(poly.bounds, (res, res))
    x_grid, y_grid = u_coord.raster_to_meshgrid(trafo, width, height)
    in_geom = sh.vectorized.contains(poly, x_grid, y_grid)

    if sum(in_geom.flatten()) > 1:
        return shgeom.MultiPoint(list(zip(x_grid[in_geom], y_grid[in_geom])))

    LOGGER.warning('Polygon smaller than resolution. Setting a representative point.')
    return shgeom.MultiPoint([poly.representative_point()])

def _interp_one_poly_m(poly, res, orig_crs):
    """
    Disaggregate a single polygon to points for resolution given in meters.
    Transforms coordinates into an adequate projected equal-area crs for this.

    Parameters
    ----------
    poly : shapely Polygon
        Polygon
    res : float
        Resolution in meters
    orig_crs: pyproj.CRS
        CRS of the polygon

    Returns
    -------
    shapely multipoint
        Grid of points rasterizing the polygon

    """

    if poly.is_empty:
        return shgeom.MultiPoint([])

    m_crs = _get_equalarea_proj(poly)
    poly_m = reproject_poly(poly, orig_crs, m_crs)

    height, width, trafo = u_coord.pts_to_raster_meta(poly_m.bounds, (res, res))
    x_grid, y_grid = u_coord.raster_to_meshgrid(trafo, width, height)

    in_geom = sh.vectorized.contains(poly_m, x_grid, y_grid)
    if sum(in_geom.flatten()) > 1:
        x_poly, y_poly = reproject_grid(
            x_grid[in_geom], y_grid[in_geom], m_crs, orig_crs)
        return shgeom.MultiPoint(list(zip(x_poly, y_poly)))

    LOGGER.warning('Polygon smaller than resolution. Setting a representative point.')
    return shgeom.MultiPoint([poly.representative_point()])


def _get_equalarea_proj(poly):
    """
    Find an adequate Equal Area Cylindrical projection
    using a representative point as lat/lon reference.

    https://proj.org/operations/projections/cea.html
    """
    repr_pnt = poly.representative_point()
    lon_0, lat_0 = repr_pnt.x, repr_pnt.y
    return "+proj=cea +lat_0=%f +lon_0=%f +units=m" %(lat_0, lon_0)

def _get_pyproj_trafo(orig_crs, dest_crs):
    """
    Get pyproj projection from orig_crs to dest_crs
    """
    return pyproj.Transformer.from_proj(pyproj.Proj(orig_crs),
                                        pyproj.Proj(dest_crs),
                                        always_xy=True)

def reproject_grid(x_grid, y_grid, orig_crs, dest_crs):
    """
    Reproject a grid from one crs to another

    Parameters
    ----------
    x_grid :
        x-coordinates
    y_grid :
        y-coordinates
    orig_crs: pyproj.CRS
        original CRS of the grid
    dest_crs: pyproj.CRS
        CRS of the grid to be reprojected to

    Returns
    -------
    x_trafo, y_trafo :
        Grid coordinates in reprojected crs
    """
    project = _get_pyproj_trafo(orig_crs, dest_crs)
    x_trafo, y_trafo = project.transform(x_grid, y_grid)
    return x_trafo, y_trafo


def reproject_poly(poly, orig_crs, dest_crs):
    """
    Reproject a polygon from one crs to another

    Parameters
    ----------
    poly : shapely Polygon
        Polygon
    orig_crs: pyproj.CRS
        original CRS of the polygon
    dest_crs: pyproj.CRS
        CRS of the polygon to be reprojected to

    Returns
    -------
    poly : shapely Polygon
        Polygon in desired projection
    """

    project = _get_pyproj_trafo(orig_crs, dest_crs)
    return sh.ops.transform(project.transform, poly)

def _line_to_pnts(gdf_lines, res, to_meters=False):

    """
    Convert a GeoDataframe with LineString geometries to
    Point geometries, where Points are placed at a specified distance
    (in meters, if applicable) along the original LineString. Each line is
    reduced to at least two points.

    Parameters
    ----------
    gdf_lines : gpd.GeoDataframe
        Geodataframe with line geometries
    res : float
        Resolution (distance) apart from which the generated Points
        should be approximately placed.

    Returns
    -------
    gdf_points : gpd.GeoDataFrame
        Geodataframe with a double index, first for line geometries,
        second for the point disaggregation of the lines (i.e. one Point
        per row).

    See also
    --------
    * util.coordinates.compute_geodesic_lengths()
    """

    if gdf_lines.empty:
        return gdf_lines

    # Needed because gdf.explode() requires numeric index
    idx = gdf_lines.index.to_list() #To restore the naming of the index
    gdf_points = gdf_lines.copy().reset_index(drop=True)

    if to_meters:
        line_lengths = u_coord.compute_geodesic_lengths(gdf_points)
    else:
        line_lengths = gdf_lines.length

    line_fractions = [
        np.linspace(0, 1, num=_pnts_per_line(length, res))
        for length in line_lengths
        ]

    gdf_points['geometry_pnt'] = [
        shgeom.MultiPoint([
            line.interpolate(dist, normalized=True)
            for dist in fractions
            ])
        for line, fractions in zip(gdf_points.geometry, line_fractions)
        ]

    gdf_points = _swap_geom_cols(
        gdf_points, geom_to='geometry_orig', new_geom='geometry_pnt')

    gdf_points = gdf_points.explode()
    gdf_points.index = gdf_points.index.set_levels(idx, level=0)
    return gdf_points

def _pnts_per_line(length, res):
    return int(np.ceil(length / res) + 1)

def _swap_geom_cols(gdf, geom_to, new_geom):
    """
    Change which column is the geometry column
    Parameters
    ----------
    gdf : GeoDataFrame
        Input geodatafram
    geom_to : string
        New name of the current 'geometry' column
    new_geom : string
        Column that should be set as the 'geometry' column. The column
        new_geom is renamed to 'geometry'
    Returns
    -------
    gdf_swap : GeoDataFrame
        Copy of gdf with the new geometry column
    """
    gdf_swap = gdf.rename(columns = {'geometry': geom_to})
    gdf_swap.rename(columns = {new_geom: 'geometry'}, inplace=True)
    gdf_swap.set_geometry('geometry', inplace=True)
    return gdf_swap


# TODO: To be removed in a future iteration and included directly into the impact class
def set_imp_mat(impact, imp_mat):
    """
    Set Impact attributes from the impact matrix. Returns a copy.
    Overwrites eai_exp, at_event, aai_agg, imp_mat.

    Parameters
    ----------
    impact : Impact
        Impact instance.
    imp_mat : sparse.csr_matrix
        matrix num_events x num_exp with impacts.

    Returns
    -------
    imp : Impact
        Copy of impact with eai_exp, at_event, aai_agg, imp_mat set.

    """
    imp = copy.deepcopy(impact)
    imp.eai_exp = eai_exp_from_mat(imp_mat, imp.frequency)
    imp.at_event = at_event_from_mat(imp_mat)
    imp.aai_agg = aai_agg_from_at_event(imp.at_event, imp.frequency)
    imp.imp_mat = imp_mat
    return imp

def eai_exp_from_mat(imp_mat, freq):
    """
    Compute impact for each exposures from the total impact matrix

    Parameters
    ----------
    imp_mat : sparse.csr_matrix
        matrix num_events x num_exp with impacts.
    frequency : np.array
        annual frequency of events

    Returns
    -------
    eai_exp : np.array
        expected annual impact for each exposure

    """
    freq_mat = freq.reshape(len(freq), 1)
    return imp_mat.multiply(freq_mat).sum(axis=0).A1

def at_event_from_mat(imp_mat):
    """
    Compute impact for each hazard event from the total impact matrix

    Parameters
    ----------
    imp_mat : sparse.csr_matrix
        matrix num_events x num_exp with impacts.

    Returns
    -------
    at_event : np.array
        impact for each hazard event

    """
    return np.squeeze(np.asarray(np.sum(imp_mat, axis=1)))

def aai_agg_from_at_event(at_event, freq):
    """
    Aggregate impact.at_event

    Parameters
    ----------
    at_event : np.array
        impact for each hazard event
    frequency : np.array
        annual frequency of event

    Returns
    -------
    float
        average annual impact aggregated

    """
    return sum(at_event * freq)