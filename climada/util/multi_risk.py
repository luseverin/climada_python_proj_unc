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

import numpy as np
import copy

import pandas as pd
import scipy as sp
from scipy.interpolate import griddata
from scipy.sparse import csr_matrix

from climada.util import yearsets

from climada.engine import Impact
from climada.entity import LOGGER
from climada.util.api_client import Client


def aggregate_impact_from_event_name(imp, exp=None):
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
    imp_mat = sum_impact_by_event_name(impact, exp)
    impact.frequency = np.ones(imp_mat.shape[0])/imp_mat.shape[0]
    impact = impact.set_imp_mat(imp_mat)
    impact.date = np.arange(1, len(impact.at_event) + 1)
    impact.event_id = np.arange(1, len(impact.at_event) + 1)
    impact.event_name = np.unique(imp.event_name)
    impact.tag['yimp object'] = True
    return impact


def sum_impact_by_event_name(imp, exp=None):
    """
    Sum the impact for all events in the same year. Impact per year cannot
    exceed exposures value.

    Parameters
    ----------
    imp : Impact
        Impact with impact matrix and events over several years
    exp: Exposure, optional
        Exposure of the Impact. If none, impact is simply summed.
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
    if exp is not None:
        exp_mat = np.stack([exp.gdf.value.to_numpy() for n in np.unique(imp.event_name)])
        def sparse_min(A, B):
             """
             Return the element-wise maximum of sparse matrices `A` and `B`.
             """
             AgtB = (A < B).astype(int)
             M = np.multiply(AgtB, A - B) + B
             return sp.sparse.csr_matrix(M)
        sum_mat = sparse_min(sum_mat, exp_mat)
    return sum_mat


def get_indices_drivers(impact, n_events, list_years, list_climate_models=None):
    indices = {climate_model: {year: np.unique([i for i, elem in enumerate(impact.event_name) if
                                                ((climate_model in elem) & (str(year) in elem))]) for year in list_years} for
               climate_model in list_climate_models}

    indices = [np.random.choice(indices[model][year]) for year in list_years for model in list_climate_models for
               n in range(n_events)]

    return indices


def order_climate_driver(impact, n_events, list_climate_models=None, list_years=None):
    indices = get_indices_drivers(impact, n_events, list_years=list_years, list_climate_models=list_climate_models)
    impact_ordered = Impact()
    impact_ordered.imp_mat = impact.imp_mat[indices]
    impact_ordered.event_name = [impact.event_name[index] for index in indices]
    impact_ordered.frequency = np.ones(len(impact_ordered.event_name))/len(impact_ordered.event_name)
    impact_ordered = impact_ordered.set_imp_mat(impact_ordered.imp_mat)
    impact_ordered.coord_exp = impact.coord_exp
    impact_ordered.event_id = np.arange(len(impact_ordered.event_name))
    return impact_ordered


def downscale_impact(impact, impact2):
    new_imp_mat = griddata(impact.coord_exp,
                           impact.imp_mat.todense().T, impact2.coord_exp,
                           method='nearest').T

    new_impact = copy.deepcopy(impact2)
    new_imp_mat = new_imp_mat*(len(impact.coord_exp)/len(impact2.coord_exp))
    new_impact = new_impact.set_imp_mat(csr_matrix(new_imp_mat))
    return new_impact

#upscale prob hazard wilfires

def get_consecutive_years():
    return dict

def combine_yearly_impacts(impact_list, how='sum', exp=None):
    """

    Parameters
    ----------
    impact_list : sparse.csr_matrix
        matrix num_events x num_exp with impacts.
    how : how to combine the impacts, options are 'sum', 'max'
    exposures : If the exposures are given, the impacts are caped at their value

    Returns
    -------
    imp : Impact
        Combined impact
    """
    imp = copy.deepcopy(impact_list[0])

    if how == 'sum':
        sum_mat = np.sum([impact.imp_mat for impact in impact_list], axis=0)
    if exp is not None:
        exp_mat = np.stack([exp.gdf.value.to_numpy() for n in np.unique(imp.shape[0])])

        def sparse_min(A, B):
            """
            Return the element-wise maximum of sparse matrices `A` and `B`.
            """
            AgtB = (A < B).astype(int)
            M = np.multiply(AgtB, A - B) + B
            return sp.sparse.csr_matrix(M)

        sum_mat = sparse_min(sum_mat, exp_mat)
    imp = imp.set_imp_mat(sum_mat)
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
    yearset = sample_events(impact, lam=lam, years=years)
    if yearset.imp_mat.shape[0]>len(years):
        yearset = yearsets.aggregate_impact_to_year(yearset, exp=exposures)
    return(yearset)


def make_consistent_drivers_yearsets(impact_dict, n_events, list_climate_models=None, list_years=None):
    yearset_dict = \
        {hazard: order_climate_driver(impact_dict[hazard], n_events, list_climate_models, list_years) for hazard in impact_dict}
    return yearset_dict


def make_correlation_matrix(impact_dict, temporal=True,spatial=False):
    if temporal is True and spatial is False:
        df = pd.DataFrame.from_dict({hazard: impact_dict[hazard].at_event for hazard in impact_dict})
    if spatial is True and temporal is False:
        df = pd.DataFrame.from_dict({hazard: impact_dict[hazard].eai_exp for hazard in impact_dict})
    if spatial is True and temporal is True:
        df = pd.DataFrame.from_dict({hazard: np.array(impact_dict[hazard].imp_mat.todense().flatten())[0] for hazard in impact_dict})
    return df.corr()