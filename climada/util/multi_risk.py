"""
This file is part of CLIMADA.
Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.
CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.
CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.
---
Define functions to handle impact_yearsets
"""

import copy

import numpy as np
import pandas as pd
import scipy as sp
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix

from climada.engine import Impact
from climada.engine.impact_calc import ImpactCalc
from climada.util import yearsets, coordinates
from climada.util.yearsets import set_imp_mat


def get_indices_drivers(impact, n_events, list_string1, list_string2):
    """Return the indices of the event names ordered based on the given lists of strings."""

    indices = {str2: {str1: np.unique([i for i, elem in enumerate(impact.event_name) if
                                                ((str(str2) in elem) & (str(str1) in elem))]) for str1 in list_string1} for
               str2 in list_string2}
    combinations = 1
    if type(list_string1) is list:
        combinations = len(list_string1)
    if type(list_string2) is list:
        combinations = combinations*len(list_string2)
    n_samples = n_events/combinations

    if not n_samples.is_integer():
        raise ValueError("Please provide a number of events that can be divided by the number of combination")
    else:
        n_samples = int(n_samples)
    indices = [np.random.choice(indices[model][year]) for model in list_string2 for year in list_string1 for
               n in range(n_samples)]
    return indices


def order_events_by_name(impact, n_events=1, list_string1=None, list_string2=None):
    """
    Order event names based on given strings contained in the event names.

    Parameters
    ----------
    impact: Impact
        with event_name based on the given strings
    n_events: Int
        Number of events in the output. Default: 1
    list_string1 : list
        A list of string based on which to order the events.
        For example climate models ['miroc5','ipsl-cm5a-lr','gfdl-esm2m','hadgem2-es']
        default is None
    list_string2 : list
         A list of string based on which to order the events.
        For example climate models ['2020','2021','2022','2023']
        default is None

    Raises
    ------
    AttributeError
        If no list is providing

    Returns
    -------
    impact : Impact
        Impact yearset.

    """
    if list_string1 is None and list_string2 is None:
        raise ValueError('provide at least one list of string to use to order the events')
    indices = get_indices_drivers(impact, n_events, list_string1=list_string1, list_string2=list_string2)
    impact_ordered = Impact()
    impact_ordered.imp_mat = impact.imp_mat[indices]
    impact_ordered.event_name = [impact.event_name[index] for index in indices]
    impact_ordered.event_id = np.arange(len(impact_ordered.event_name))
    frequency = impact.frequency[indices]
    impact_ordered.frequency = frequency*(len(impact.event_id)/len(impact_ordered.event_id))
    impact_ordered = set_imp_mat(impact_ordered, impact_ordered.imp_mat)
    impact_ordered.coord_exp = impact.coord_exp
    return impact_ordered


def sparse_min(A, B):
    """
    Return the element-wise minimum of sparse matrices `A` and `B`.
    """
    AgtB = (A < B).astype(int)
    M = np.multiply(AgtB, A - B) + B
    return sp.sparse.csr_matrix(M)


def aggregate_impact_from_event_name(imp, how='sum', exp=None):
    """
    Aggregate the impact per year to make yearsets. Maximum impact per year
    at each exposure point is exposure value if exp is not None.

    Parameters
    ----------
    imp : Impact
        Impact with an impact matrix and events with dates per year

    exp : Exposure
        Exposure of Impact.
    Raises
    ------
    AttributeError
        If impact matrix is empty.

    Returns
    -------
    impact : Impact
        Impact yearset.

    """
    #if imp.imp_mat.nnz == 0:
    #    raise AttributeError("The impact matrix from imp.imp_mat is empty.")

    impact = copy.deepcopy(imp)
    if how == 'sum':
        imp_mat = sum_impact_by_event_name(impact)
    elif how == 'max':
        imp_mat = sp.sparse.csr_matrix(sp.sparse.vstack(
        [impact.imp_mat[(np.array(imp.event_name) == event).astype(bool)].max(axis=0)
         for event in np.unique(impact.event_name)]))

    if exp is not None:
        exp_mat = np.stack([exp.gdf.value.to_numpy() for n in np.unique(imp.event_name)])
        imp_mat = sparse_min(imp_mat, exp_mat)

    impact.frequency = np.ones(imp_mat.shape[0])/imp_mat.shape[0]
    impact = set_imp_mat(impact, imp_mat)
    impact.date = np.arange(1, len(impact.at_event) + 1)
    impact.event_id = np.arange(1, len(impact.at_event) + 1)
    impact.event_name = np.unique(imp.event_name)
    impact.tag['yimp object'] = True
    return impact


def sum_impact_by_event_name(imp):
    """
    Sum the impact for all events with the same name.
    Parameters
    ----------
    imp : Impact
        Impact with impact matrix and events over several years

    Returns
    -------
    sp.sparse.csr_matrix
        Impact matrix with one event per year

    """
    mat = imp.imp_mat
    mask =[np.ma.make_mask(np.array(imp.event_name) == event).astype(int)
           for event in np.unique(imp.event_name)]
    mask_matrix =  sp.sparse.csr_matrix(mask)
    sum_mat = mask_matrix.dot(mat)
    return sum_mat


def downscale_impact(impact, impact2):
    new_imp_mat = griddata(impact.coord_exp,
                           impact.imp_mat.todense().T, impact2.coord_exp,
                           method='nearest').T

    new_impact = copy.deepcopy(impact2)
    new_imp_mat = new_imp_mat*(len(impact.coord_exp)/len(impact2.coord_exp))
    new_impact = set_imp_mat(new_impact, csr_matrix(new_imp_mat))
    return new_impact



def combine_yearly_impacts(impact_list, how='sum', exp=None):
    """
    Parameters
    ----------
    impact_list : sparse.csr_matrix
        matrix num_events x num_exp with impacts.
    how : how to combine the impacts, options are 'sum', 'max' or 'min'
    exposures : If the exposures are given, the impacts are caped at their value

    Returns
    -------
    imp : Impact
        Combined impact
    """

    existing_methods = ['sum', 'max', 'min']
    imp0 = copy.deepcopy(impact_list[0])

    if how == 'sum':
        imp_mat = np.sum([impact.imp_mat for impact in impact_list], axis=0)

    elif how == 'min':
        imp_mat_min = imp0.imp_mat
        for imp in impact_list[1:-1]:
            imp_mat_min = imp_mat_min.minimum(imp.imp_mat)
        imp_mat = imp_mat_min

    elif how == 'max':
        imp_mat_max = impact_list[0]
        for imp_mat in impact_list[1:-1]:
            imp_mat_max = imp_mat_max.maximum(imp_mat)
        imp_mat = imp_mat_max
    else:
        raise ValueError(f"'{how}' is not a valid method. The implemented methods are sum, max or min")

    if exp is not None:
        exp_mat = np.stack([exp.gdf.value.to_numpy() for n in range(len(imp0.event_name))])
        imp_mat = sparse_min(imp_mat, exp_mat)

    imp = set_imp_mat(imp0, imp_mat)
    return imp


def sample_events(impact, years, lam=1):
    #create sampling vector
    events_per_year = yearsets.sample_from_poisson(len(years), lam)
    sampling_vect = yearsets.sample_events(events_per_year, impact.frequency)
    impact_sample = yearsets.impact_from_sample(impact, years, sampling_vect)
    return impact_sample


def make_yearset_samples(impact, years, exposures=None, n_samples=1):
    [make_yearset(impact, years, exposures) for sample in range(n_samples)]


def make_yearset(impact, years, exposures=None):
    lam = np.sum(impact.frequency)
    lam = np.round(lam, 10)
    events_per_year = yearsets.sample_from_poisson(len(years), lam)
    sampling_vect = yearsets.sample_events(events_per_year, impact.frequency)
    yearset = yearsets.impact_from_sample(impact, years, sampling_vect)
    if yearset.imp_mat.shape[0]>len(years):
        yearset = yearsets.aggregate_impact_to_year(yearset, exp=exposures)
    return(yearset)


def make_consistent_drivers_yearsets(impact_dict, n_events, list_string1=None, list_string2=None):
    yearset_dict = \
        {hazard: order_events_by_name(impact_dict[hazard], n_events, list_string1, list_string2) for hazard in impact_dict}
    return yearset_dict


def make_correlation_matrix(impact_dict, temporal=True,spatial=False):
    if temporal is True and spatial is False:
        df = pd.DataFrame.from_dict({hazard: impact_dict[hazard].at_event for hazard in impact_dict})
    if spatial is True and temporal is False:
        df = pd.DataFrame.from_dict({hazard: impact_dict[hazard].eai_exp for hazard in impact_dict})
    if spatial is True and temporal is True:
        df = pd.DataFrame.from_dict({hazard: np.array(impact_dict[hazard].imp_mat.todense().flatten())[0] for hazard in impact_dict})
    return df.corr()


def make_country_matrix(impact, countries):
    lat = np.array([lat for lat,lon in impact.coord_exp])
    lon = np.array([lon for lat,lon in impact.coord_exp])
    countries_num = coordinates.get_country_code(lat, lon)
    countries_num = np.array([format(num, '03d') for num in countries_num])
    country_matrices = {country: np.array([impact.imp_mat[:,countries_num==country].sum(axis=1)])
                        for country in countries}
    return country_matrices


def correlation_impacts_per_country(impact, countries):
    country_matrices = make_country_matrix(impact)
    corr_df = pd.DataFrame({'correlation': pd.Series(country_matrices[combi[0]],
                                                    country_matrices[combi[1]]), 'region_id': countries}
                           for combi in list(country_matrices.keys(), 2))
    return corr_df
