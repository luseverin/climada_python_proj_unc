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
import logging
import numpy as np
import scipy as sp

from numpy.random import default_rng

import climada.util.dates_times as u_dt
from climada.engine import ImpactCalc

LOGGER = logging.getLogger(__name__)

def impact_yearset(imp, sampled_years, lam=None, correction_fac=True, seed=None):
    """Create a yearset of impacts (yimp) containing a probabilistic impact for each year
    in the sampled_years list by sampling events from the impact received as input with a
    Poisson distribution centered around lam per year (lam = sum(imp.frequency)).
    In contrast to the expected annual impact (eai) yimp contains impact values that
    differ among years. When correction factor is true, the yimp are scaled such
    that the average over all years is equal to the eai.

    Parameters
    -----------
        imp : climada.engine.Impact()
            impact object containing impacts per event
        sampled_years : list
            A list of years that shall be covered by the resulting yimp.
        seed : Any, optional
            seed for the default bit generator
            default: None

    Optional parameters
        lam: int
            The applied Poisson distribution is centered around lam events per year.
            If no lambda value is given, the default lam = sum(imp.frequency) is used.
        correction_fac : boolean
            If True a correction factor is applied to the resulting yimp. It is
            scaled in such a way that the expected annual impact (eai) of the yimp
            equals the eai of the input impact

    Returns
    -------
        yimp : climada.engine.Impact()
             yearset of impacts containing annual impacts for all sampled_years
        sampling_vect : 2D array
            The sampling vector specifies how to sample the yimp, it consists of one
            sub-array per sampled_year, which contains the event_ids of the events used to
            calculate the annual impacts.
            Can be used to re-create the exact same yimp.
      """

    n_sampled_years = len(sampled_years)

    #create sampling vector
    if not lam:
        lam = np.sum(imp.frequency)
    events_per_year = sample_from_poisson(n_sampled_years, lam, seed=seed)
    sampling_vect = sample_events(events_per_year, imp.frequency, seed=seed)

    #compute impact per sampled_year
    imp_per_year = compute_imp_per_year(imp, sampling_vect)

    #copy imp object as basis for the yimp object
    yimp = copy.deepcopy(imp)

    #save imp_per_year in yimp
    if correction_fac: #adjust for sampling error
        yimp.at_event = imp_per_year / calculate_correction_fac(imp_per_year, imp)
    else:
        yimp.at_event = imp_per_year

    #save calculations in yimp
    yimp.event_id = np.arange(1, n_sampled_years+1)
    yimp.tag['yimp object'] = True
    yimp.date = u_dt.str_to_date([str(date) + '-01-01' for date in sampled_years])
    yimp.frequency = np.ones(n_sampled_years)*sum(len(row) for row in sampling_vect
                                                            )/n_sampled_years

    return yimp, sampling_vect

def impact_yearset_from_sampling_vect(imp, sampled_years, sampling_vect, correction_fac=True):
    """Create a yearset of impacts (yimp) containing a probabilistic impact for each year
    in the sampled_years list by sampling events from the impact received as input following
    the sampling vector provided.
    In contrast to the expected annual impact (eai) yimp contains impact values that
    differ among years. When correction factor is true, the yimp are scaled such
    that the average over all years is equal to the eai.

    Parameters
    -----------
        imp : climada.engine.Impact()
            impact object containing impacts per event
        sampled_years : list
            A list of years that shall be covered by the resulting yimp.
        sampling_vect : 2D array
            The sampling vector specifies how to sample the yimp, it consists of one
            sub-array per sampled_year, which contains the event_ids of the events used to
            calculate the annual impacts.
            It needs to be obtained in a first call,
            i.e. [yimp, sampling_vect] = climada_yearsets.impact_yearset(...)
            and can then be provided in this function to obtain the exact same sampling
            (also for a different imp object)

    Optional parameter
        correction_fac : boolean
            If True a correction factor is applied to the resulting yimp. It is
            scaled in such a way that the expected annual impact (eai) of the yimp
            equals the eai of the input impact

    Returns
    -------
        yimp : climada.engine.Impact()
             yearset of impacts containing annual impacts for all sampled_years

    """

    #compute impact per sampled_year
    imp_per_year = compute_imp_per_year(imp, sampling_vect)

    #copy imp object as basis for the yimp object
    yimp = copy.deepcopy(imp)


    if correction_fac: #adjust for sampling error
        imp_per_year = imp_per_year / calculate_correction_fac(imp_per_year, imp)

    #save calculations in yimp
    yimp.at_event = imp_per_year
    n_sampled_years = len(sampled_years)
    yimp.event_id = np.arange(1, n_sampled_years+1)
    yimp.tag['yimp object'] = True
    yimp.date = u_dt.str_to_date([str(date) + '-01-01' for date in sampled_years])
    yimp.frequency = np.ones(n_sampled_years)*sum(len(row) for row in sampling_vect
                                                            )/n_sampled_years

    return yimp

def years_from_imp(imp):
    """
    Extract the years of all events in the impact

    Parameters
    ----------
    imp : Impact
        Impact with events

    Returns
    -------
    list
        Years of each event in imp (same ordering).

    """
    return [int(u_dt.date_to_str(date)[0:4]) for date in imp.date]

def sum_impact_year_per_year(imp, exp=None):
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
    years = years_from_imp(imp)
    mask =[np.ma.make_mask(years == year).astype(int)
           for year in np.unique(years)]
    mask_matrix =  sp.sparse.csr_matrix(mask)
    sum_mat = mask_matrix.dot(mat)
    if exp is not None:
        exp_mat = np.stack([exp.gdf.value.to_numpy() for n in np.unique(years)])
        def sparse_min(A, B):
             """
             Return the element-wise maximum of sparse matrices `A` and `B`.
             """
             AgtB = (A < B).astype(int)
             M = np.multiply(AgtB, A - B) + B
             return sp.sparse.csr_matrix(M)
        sum_mat = sparse_min(sum_mat, exp_mat)
    return sum_mat

def aggregate_impact_to_year(imp, exp=None):
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
    imp_mat = sum_impact_year_per_year(impact, exp)
    impact.frequency = np.ones(imp_mat.shape[0])/imp_mat.shape[0]
    impact = impact.set_imp_mat(imp_mat)
    impact.date = np.unique(u_dt.str_to_date([str(date) + '-01-01' for date in years_from_imp(impact)]))
    impact.event_id = np.arange(1, len(impact.at_event) + 1)
    impact.event_name = impact.event_id
    impact.tag['yimp object'] = True
    return impact


def impact_from_sample(imp, years, sampling_vec):
    """
    Set impact object from sample of events

    Note: frequency is set to 1 for all events.

    Parameters
    ----------
    imp : Impact
        Impact to sample event impact from
    years : list
        List of the years from the sampling vector
    sampling_vec : list[np.array]
        Array of ids (row index) of selected events per year.

    Returns
    -------
    impact: Impact
        Impact reduced to sample of events

    """
    #if imp.imp_mat.nnz == 0:
    #    raise AttributeError("The impact matrix from imp.imp_mat is empty.")

    impact = copy.deepcopy(imp)
    impact.date = year_date_event_in_sample(years=years, dates=impact.date,
                                            sampling_vec=sampling_vec)
    impact.frequency = frequency_for_sample(sampling_vec)
    imp_mat = extract_event_matrix(mat=impact.imp_mat, sampling_vec=sampling_vec)
    impact = set_imp_mat(impact, imp_mat)
    impact.event_id = np.arange(1, len(impact.at_event) + 1)
    impact.event_name = list(np.array(imp.event_name)[np.concatenate(sampling_vec)])
    return impact


def extract_event_matrix(mat, sampling_vec):
    """
    Extract sampled events from impact matrix

    Parameters
    ----------
    mat : scipy.sparse.csr_matrix
        Impact matrix to sample from
    sampling_vec : list[array()]
        Array of ids (row index) of selected events per year.

    Returns
    -------
    scipy.sparse.csr_matrix
        Impact matrix of selected sample of events

    """
    sampling_vec = np.concatenate(sampling_vec)
    return mat[sampling_vec, :]

def year_date_event_in_sample(years, dates, sampling_vec):
    """
    Change the year for the sampled events

    Parameters
    ----------
    years : list[int]
        Years of sampled events (length equal to lenght of sampling_vec)
    dates : list[dates]
        List of dates in ordinal format for the whole event set
    sampling_vec : list[np.array]
        Array of ids (row index) of selected events per year.

    Raises
    ------
    ValueError
        length of years must equal length of sampling_vec

    Returns
    -------
    list[dates]
        List with dates in ordinal format of the sampled events.

    """
    if len(years) != len(sampling_vec):
        raise ValueError("The number of years is different from the length" +
                         "of the sampling vector")
    def change_year(old_date, year):
        old_date = u_dt.date_to_str(old_date)
        if old_date[5:7] == '02' and old_date[8:10] == '29':
            old_date = old_date.replace('29', '28')
        new_date = old_date.replace(old_date[0:4], str(year))
        return u_dt.str_to_date(new_date)

    return [
        change_year(date, year)
        for year, events in zip(years, sampling_vec)
        for date in np.array(dates)[events]
        ]

def frequency_for_sample(sampling_vec):
    """
    Generate frequency vector for selected sample of events

    Parameters
    ----------
    sampling_vec : list[np.array]
        List of selected events per year

    Returns
    -------
    np.array
        Frequency (1/year) of each sampled event

    """
    n_years = np.size(sampling_vec)
    n_events = np.size(np.concatenate(sampling_vec))
    return np.ones(n_events)/n_years


def sample_from_poisson(n_sampled_years, lam, seed=None):
    """Sample the number of events for n_sampled_years

    Parameters
    -----------
        n_sampled_years : int
            The target number of years the impact yearset shall contain.
        lam: int
            the applied Poisson distribution is centered around lambda events per year
        seed : int, optional
            seed for numpy.random, will be set if not None
            default: None

    Returns
    -------
        events_per_year : array
            Number of events per sampled year
    """
    if seed is not None:
        np.random.seed(seed)
    if lam != 1:
        events_per_year = np.round(np.random.poisson(lam=lam,
                                                     size=n_sampled_years)).astype('int')
    else:
        events_per_year = np.ones(n_sampled_years).astype('int')


    return events_per_year

def sample_events(events_per_year, freqs_orig, seed=None):
    """Sample events uniformely from an array (indices_orig) without replacement
    (if sum(events_per_year) > n_input_events the input events are repeated
    (tot_n_events/n_input_events) times, by ensuring that the same events doens't
    occur more than once per sampled year).

    Parameters
    -----------
        events_per_year : array
            Number of events per sampled year
        freqs_orig : array
            Frequency of each input event
        seed : Any, optional
            seed for the default bit generator.
            Default: None

    Returns
    -------
        sampling_vect : 2D array
            The sampling vector specifies how to sample the yimp, it consists of one
            sub-array per sampled_year, which contains the event_ids of the events used to
            calculate the annual impacts.
    """

    sampling_vect = []

    indices_orig = np.arange(len(freqs_orig))

    freqs = freqs_orig
    indices = indices_orig

    #sample events for each sampled year
    for amount_events in events_per_year:
        #if there are not enough input events, choice with no replace will fail
        if amount_events > len(freqs_orig):
            raise ValueError(f"cannot sample {amount_events} distinct events for a single year"
                             f" when there are only {len(freqs_orig)} input events")

        #add the original indices and frequencies to the pool if there are less events
        #in the pool than needed to fill the year one is sampling for
        #or if the pool is empty (not covered in case amount_events is 0)
        if len(np.unique(indices)) < amount_events or len(indices) == 0:
            indices = np.append(indices, indices_orig)
            freqs = np.append(freqs, freqs_orig)

        #ensure that each event only occurs once per sampled year
        unique_events = np.unique(indices, return_index=True)[0]
        probab_dis = freqs[np.unique(indices, return_index=True)[1]]/(
            np.sum(freqs[np.unique(indices, return_index=True)[1]]))

        #sample events
        rng = default_rng(seed)
        selected_events = rng.choice(unique_events, size=amount_events, replace=False,
                                     p=probab_dis).astype('int')

        #determine used events to remove them from sampling pool
        idx_to_remove = [np.where(indices == event)[0][0] for event in selected_events]
        indices = np.delete(indices, idx_to_remove)
        freqs = np.delete(freqs, idx_to_remove)

        #save sampled events in sampling vector
        sampling_vect.append(selected_events)

    return sampling_vect

def compute_imp_per_year(imp, sampling_vect):
    """Sample annual impacts from the given event_impacts according to the sampling dictionary

    Parameters
    -----------
        imp : climada.engine.Impact()
            impact object containing impacts per event
        sampling_vect : 2D array
            The sampling vector specifies how to sample the yimp, it consists of one
            sub-array per sampled_year, which contains the event_ids of the events used to
            calculate the annual impacts.

    Returns
    -------
        imp_per_year: array
            Sampled impact per year (length = sampled_years)
    """

    imp_per_year = [np.sum(imp.at_event[list(sampled_events)]) for sampled_events in
                    sampling_vect]

    return imp_per_year


def set_imp_mat(impact, imp_mat):
    """
    Set Impact attributes from the impact matrix. Returns a copy.
    Overwrites eai_exp, at_event, aai_agg, imp_mat

    Parameters
    ----------
    impact: Impact
    imp_mat : sparse.csr_matrix
        matrix num_events x num_exp with impacts.
    Returns
    -------
    imp : Impact
        Copy of impact with eai_exp, at_event, aai_agg, imp_mat set.
    """
    imp = copy.deepcopy(impact)
    imp.at_event, imp.eai_exp, imp.aai_agg = ImpactCalc.risk_metrics(imp_mat, imp.frequency)
    imp.imp_mat = imp_mat
    return imp


def calculate_correction_fac(imp_per_year, imp):
    """Calculate a correction factor that can be used to scale the yimp in such
    a way that the expected annual impact (eai) of the yimp amounts to the eai
    of the input imp

    Parameters
    -----------
        imp_per_year : array
            sampled yimp
        imp : climada.engine.Impact()
            impact object containing impacts per event

    Returns
    -------
        correction_factor: int
            The correction factor is calculated as imp_eai/yimp_eai
    """

    yimp_eai = np.sum(imp_per_year)/len(imp_per_year)
    imp_eai = np.sum(imp.frequency*imp.at_event)
    correction_factor = imp_eai/yimp_eai
    LOGGER.info("The correction factor amounts to %s", (correction_factor-1)*100)

    # if correction_factor > 0.1:
    #     tex = raw_input("Do you want to exclude small events?")

    return correction_factor
