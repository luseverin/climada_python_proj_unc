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

Define HazardSet class.
"""

__all__ = ['HazardSet']


import logging

from climada.hazard.base import Hazard
from climada.entity.tag import Tag

LOGGER = logging.getLogger(__name__)

class HazardSet():
    """Contains hazards of type Hazard. Loads from
    files with format defined in FILE_EXT.

    Attributes:
        tag (Tag): information about the source data
        _data (dict): contains Hazard classes. It's not suppossed to be
            directly accessed. Use the class methods instead.
    """

    def __init__(self):
        """Empty initialization of a HazardSet consisting of several hazards,
        identified by an unique id.

            >>> haz_1 = Hazard()
            >>> haz_1.tag.haz_type = 'TC'
            >>> haz_set = HazardSet()
            >>> haz_set.append(haz1, 'TC1')

        """
        self.clear()

    def clear(self):
        """Reinitialize attributes."""
        self.tag = Tag()
        self._data = dict()  # {haz_id : Hazard}


    def append(self, haz, haz_id=None):
        """Append a Hazard. Give next unalloced integer id to hazards without id
        or wiht an hazard id already contained in the HazardSet.

        Parameters
        ----------
        haz : climada.hazard.Hazard()
            Hazard instance.
        haz_id : string
            Hazard id. The default is None and sets the hazard id to the lowest
            integer not contained in the hazard_set starting with '1'.

        Raises
        ------
        ValueError
        """

        if not isinstance(haz, Hazard):
            raise TypeError("Input variable haz is not of type Hazard.")

        if haz_id is None or haz_id in list(self._data.keys()):
            idx = '1'
            while idx in list(self._data.keys()):
                idx = str(int(idx)+1)

            if haz_id in list(self._data.keys()):
                LOGGER.warning("A hazard with  the id %s is already contained"
                               "in this hazard set. The id is set to %s.", haz_id, idx)
            haz_id = idx

        self._data[haz_id] = haz


    def remove_haz(self, haz_id):
        """Remove hazard(s) with provided hazard id.
        If no input provided, all hazards are removed.

       Parameters
        ----------
        haz : climada.hazard.Hazard()
            Hazard instance.
        haz_id : str
            id of hazard to be removed from the hazard set.
        """


        if haz_id == 'all':
            self._data = dict()
        else:
            try:
                del self._data[haz_id]
            except KeyError:
                LOGGER.warning("The hazard with haz_id %s is not contained"
                               "in this hazard set.", haz_id)


    def get_haz(self, haz_id):
        """Get Hazard(s) of input hazard id.

        Parameters
        ----------
        haz_id : string
            Hazard id.

        Returns
        ----------
        Hazard : climada.hazard.Hazard()
            Hazard object with input hazard id.
        """

        try:
            return self._data[haz_id]
        except KeyError:
            LOGGER.warning("The hazard with haz_id %s is not contained in this hazard set.", haz_id)
            return None

    def get_haz_types(self, haz_id='all'):
        """Get hazard types contained for the id(s) provided.
        Return all hazard types if no input id.

        Parameters
        ----------
        haz_id : string
            Hazard id.

        Returns
        ----------
        haz_types : list
            List of hazard types with specified hazard id
            Default: returns a list  of all hazard types
        """

        haz_types = []

        if haz_id == 'all':
            haz_types = [
                self._data[haz].tag.haz_type
                for haz in list(self._data.keys())
            ]
        else:
            haz_types = [self._data[haz_id].tag.haz_type]

        return haz_types


    def get_ids(self, haz_type='all'):
        """
        Get hazard ids (of a specific hazard type).

        Parameters
        ----------
        haz_type : str, optional
            Hazard type to return ids for. The default is None and returns a list of the ids.

        Returns
        -------
        haz_ids : list
            List of hazard ids of the specified hazard type.

        """
        if haz_type == 'all':
            haz_ids = list(self._data.keys())
        else:
            haz_ids = [haz_id for haz_id, haz in self._data.items() if haz.tag.haz_type == haz_type]

        return haz_ids


    def extend(self, haz_set):
        """Append hazard set of input HazardSet to current
        HazardSet. Overwrite Hazard if same id.

        Parameters
        ----------
            haz_set (HazardSet): HazardSet instance to extend

        """

        self.tag.append(haz_set.tag)

        new_haz_set = haz_set.get_ids()
        for haz_id in new_haz_set:
            self.append(haz_set.get_haz(haz_id), haz_id)

    def select(self, haz_type='all', event_names=None, date=None, orig=None,
               reg_id=None, reset_frequency=False):

        """Select events matching provided criteria

        The frequency of events may need to be recomputed (see `reset_frequency`)!

        Parameters
        ----------
        haz_type: str
            Hazard type to be extracted. Default 'al' returns all Hazards contained
            in the HazardSet that fulfill the rest of the criteria (taken from hazard.select method)
        event_names : list of str, optional
            Names of events.
        date : array-like of length 2 containing str or int, optional
            (initial date, final date) in string ISO format ('2011-01-02') or datetime
            ordinal integer.
        orig : bool, optional
            Select only historical (True) or only synthetic (False) events.
        reg_id : int, optional
            Region identifier of the centroids' region_id attibute.
        reset_frequency : bool, optional
            Change frequency of events proportional to difference between first and last
            year (old and new). Default: False.

        Returns
        -------
        haz_set : HazardSet
            HazardSet containing only the hazards that fulfill the provided criteria
        """

        haz_set = HazardSet()

        haz_ids = self.get_ids(haz_type)
        for haz_id in haz_ids:
            haz = self.get_haz(haz_id).select(event_names, date, orig, reg_id,
                                      reset_frequency)
            if haz is not None:
                haz_set.append(haz, haz_id)

        return haz_set
