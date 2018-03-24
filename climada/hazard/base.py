"""
Define Hazard.
"""

__all__ = ['Hazard']

#from datetime import date
from array import array
import logging
import itertools
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
from scipy import sparse

from climada.hazard.tag import Tag as TagHazard
from climada.hazard.centroids.base import Centroids
from climada.hazard.source import read  as read_source
from climada.util.files_handler import to_str_list, get_file_names
import climada.util.plot as plot
import climada.util.checker as check

LOGGER = logging.getLogger(__name__)

class Hazard(object):
    """Contains events of some hazard type defined at centroids. Interface.

    Attributes
    ----------
        tag (TagHazard): information about the source
        units (str): units of the intensity
        centroids (Centroids): centroids of the events
        event_id (np.array): id (>0) of each event
        event_name (list): name of each event (set as event_id if no provided)
        frequency (np.array): frequency of each event in seconds
        intensity (sparse.csr_matrix): intensity of the events at centroids
        fraction (sparse.csr_matrix): fraction of affected exposures for each
            event at each centroid
    """

    def __init__(self, file_name='', haz_type='NA', description='', \
                 centroids=None):
        """Initialize values from given file, if given.

        Parameters
        ----------
            file_name (str or list(str), optional): absolute file name(s) or 
                folder name containing the files to read
            haz_type (str, optional): acronym of the hazard type (e.g. 'TC')
            description (str or list(str), optional): one description of the
                data or a description of each data file
            centroids (Centroids or list(Centroids), optional): Centroids

        Raises
        ------
            ValueError
        """
        self.clear()
        if file_name != '':
            if haz_type == 'NA':
                LOGGER.error("Provide hazard type acronym.")
                raise ValueError
            else:
                self.read(file_name, haz_type, description, centroids)

    def clear(self):
        """Reinitialize attributes."""    
        self.tag = TagHazard()
        self.units = 'NA'
        # following values are defined for each event
        self.centroids = Centroids()
        self.event_id = np.array([], np.int64)
        self.frequency = np.array([])
        self.event_name = list()
        #self.date = [date(1,1,1)]  # size: num_events
        # following values are defined for each event and centroid
        self.intensity = sparse.csr_matrix([]) # events x centroids
        self.fraction = sparse.csr_matrix([])  # events x centroids

    def check(self):
        """ Check if the attributes contain consistent data.

        Raises
        ------
            ValueError
        """
        self.centroids.check()
        self._check_events()

    def read(self, files, haz_type, description='', centroids=None, \
             var_names=None):
        """Read and check hazard, and centroids if not provided. Parallel 
        through files.

        Parameters
        ----------
            file_name (str or list(str), optional): absolute file name(s) or 
                folder name containing the files to read
            haz_type (str, optional): acronym of the hazard type (e.g. 'TC')
            description (str or list(str), optional): one description of the
                data or a description of each data file
            centroids (Centroids or list(Centroids), optional): Centroids
            var_names (dict or list(dict), default): name of the variables in 
                the file (default: DEF_VAR_NAME defined in the source modules)

        Raises
        ------
            ValueError
        """
        # Construct absolute path file names
        all_files = get_file_names(files)
        num_files = len(all_files)
        desc_list = to_str_list(num_files, description, 'description')
        centr_list = to_str_list(num_files, centroids, 'centroids')
        var_list = to_str_list(num_files, var_names, 'var_names')
        self.clear()
        haz_part = Pool().map(self._read_one, all_files, \
            itertools.repeat(haz_type, num_files), desc_list, centr_list, \
            var_list)
        for haz, file in zip(haz_part, all_files):
            LOGGER.info('Read file: %s', file)    
            self.append(haz)

    def plot_stats(self):
        """Plots describing hazard."""
        # TODO

    def plot_intensity(self, event=None, centr_id=None):
        """Plot intensity values for a selected event or centroid.

        Parameters
        ----------
            event (int or str, optional): If event > 0, plot intensities of
                event with id = event. If event = 0, plot maximum intensity in
                each centroid. If event < 0, plot abs(event)-largest event. If
                event is string, plot event with name in event.
            centr_id (int, optional): If centr_id > 0, plot intensities
                of all events at centroid with id = centr_id. If centr_id = 0,
                plot maximum intensity of each event. If centr_id < 0,
                plot abs(centr_id)-largest centroid where higher intensities
                are reached.

        Returns
        -------
            matplotlib.figure.Figure (optional)

        Raises
        ------
            ValueError
        """
        col_label = 'Intensity %s' % self.units

        if event is not None:
            if isinstance(event, str):
                event = self.event_name_to_id(event)
            return self._event_plot(event, self.intensity, col_label)
        elif centr_id is not None:
            return self._centr_plot(centr_id, self.intensity, col_label)
        
        LOGGER.error("Provide one event id or one centroid id.")
        raise ValueError

    def plot_fraction(self, event=None, centr_id=None):
        """Plot fraction values for a selected event or centroid.

        Parameters
        ----------
            event (int or str, optional): If event > 0, plot fraction of event
                with id = event. If event = 0, plot maximum fraction in each
                centroid. If event < 0, plot abs(event)-largest event. If event
                is string, plot event with name in event.
            centr_id (int, optional): If centr_id > 0, plot fraction
                of all events at centroid with id = centr_id. If centr_id = 0,
                plot maximum fraction of each event. If centr_id < 0,
                plot abs(centr_id)-largest centroid where highest fractions
                are reached.

        Returns
        -------
            matplotlib.figure.Figure (optional)

        Raises
        ------
            ValueError
        """
        col_label = 'Fraction'
        if event is not None:
            if isinstance(event, str):
                event = self.event_name_to_id(event)
            return self._event_plot(event, self.fraction, col_label)
        elif centr_id is not None:
            return self._centr_plot(centr_id, self.fraction, col_label)
        
        LOGGER.error("Provide one event id or one centroid id.")
        raise ValueError

    def event_name_to_id(self, event_name):
        """"Get an event id from its name.

        Parameters
        ----------
            event_name (str): Event name

        Returns
        -------
            int

        Raises
        ------
            ValueError
        """
        try:
            return self.event_id[self.event_name.index(event_name)]
        except (ValueError, IndexError):
            LOGGER.error("No event with name: %s", event_name)
            raise ValueError

    def calc_future(self, conf):
        """ Compute the future hazard following the configuration """
        # TODO

    def append(self, hazard):
        """Check and append variables of input Hazard to current Hazard. 
        Repeated events and centroids will be overwritten.
        
        Parameters
        ----------
            hazard (Hazard): Hazard instance to append to current

        Raises
        ------
            ValueError
        """
        hazard._check_events()
        if self.event_id.size == 0:
            self.__dict__ = hazard.__dict__.copy()
            return
        
        n_ini_cen = self.centroids.id.size
        n_ini_ev = self.event_id.size
        self.tag.append(hazard.tag)
        if (self.units == 'NA') and (hazard.units != 'NA'):
            LOGGER.warning("Initial hazard does not have units.")
            self.units = hazard.units
        elif hazard.units == 'NA':
            LOGGER.warning("Appended hazard does not have units.")
        elif self.units != hazard.units:
            LOGGER.error("Hazards with different units can't be appended"\
                             + ": %s != %s.", self.units, hazard.units)
            raise ValueError
        
        # Add not repeated centroids
        # for each input centroid, position in the final centroid vector
        new_cen_pos = self.centroids.append(hazard.centroids)
        n_add_cen = np.where(np.array(new_cen_pos) >= n_ini_cen)[0].size
        if n_add_cen:
            self.intensity = sparse.hstack([self.intensity, \
                np.zeros((self.intensity.shape[0], n_add_cen))]).tolil()
            self.fraction = sparse.hstack([self.fraction, \
                np.zeros((self.fraction.shape[0], n_add_cen))]).tolil()

        # Add not repeated events
        # for each input event, position in the final event vector
        new_ev_pos = array('L')
        new_name = []
        new_id = array('L')
        self._append_events(hazard, new_ev_pos, new_name, new_id)
        n_add_ev = np.where(np.array(new_ev_pos) >= n_ini_ev)[0].size
        sparse_add = n_add_ev - 1
        if n_ini_cen != 0 or n_ini_ev != 0:
            sparse_add = n_add_ev
        if n_add_ev:
            self.event_name = self.event_name + new_name
            self.event_id = np.append(self.event_id, np.array(new_id)).\
                astype(int)
            self.intensity = sparse.vstack([self.intensity, \
                np.zeros((sparse_add, self.fraction.shape[1]))]).tolil()   
            self.fraction = sparse.vstack([self.fraction, \
                np.zeros((sparse_add, self.fraction.shape[1]))]).tolil()
            self.frequency = np.append(self.frequency, np.zeros(n_add_ev))
        # fill intensity, fraction and frequency
        for i_ev in range(hazard.event_id.size):
            self.intensity[new_ev_pos[i_ev], new_cen_pos] = \
            hazard.intensity[i_ev, :]
            self.fraction[new_ev_pos[i_ev], new_cen_pos] = \
            hazard.fraction[i_ev, :]
            self.frequency[new_ev_pos[i_ev]] = hazard.frequency[i_ev]

        self.intensity = self.intensity.tocsr()
        self.fraction = self.fraction.tocsr()
    
    @staticmethod
    def _read_one(file_name, haz_type, description, centroids, var_names=None):
        """ Read input file. If centroids are not provided, they are read
        from file_name.

        Parameters
        ----------
            file_name (str): name of the source file
            haz_type (str): acronym of the hazard type (e.g. 'TC')
            description (str): description of the source data
            centroids (Centroids, optional): Centroids instance
            var_names (dict): name of the variables in the file (e.g. 
                      DEF_VAR_NAME defined in the source modules)
            
        Raises
        ------
            ValueError, KeyError
            
        Returns
        ------
            Hazard
        """
        new_haz = Hazard()
        read_source(new_haz, file_name, haz_type, description, centroids, \
                    var_names)
        return new_haz
            
    def _check_events(self):
        """ Check that all attributes but centroids contain consistent data.

        Raises
        ------
            ValueError
        """
        num_ev = len(self.event_id)
        num_cen = len(self.centroids.id)
        if np.unique(self.event_id).size != num_ev:
            LOGGER.error("There are events with the same identifier.")
            raise ValueError
        check.size(num_ev, self.frequency, 'Hazard.frequency')
        if num_ev == 0 and num_cen == 0:
            check.shape(1, num_cen, self.intensity, 'Hazard.intensity')
            check.shape(1, num_cen, self.fraction, 'Hazard.fraction')
        else:
            check.shape(num_ev, num_cen, self.intensity, 'Hazard.intensity')
            check.shape(num_ev, num_cen, self.fraction, 'Hazard.fraction')
        check.array_default(num_ev, self.event_name, 'Hazard.event_name', \
                            list(self.event_id))
    
    def _append_events(self, hazard, new_ev_pos, new_name, new_id):
        """Iterate over hazard events and collect their new position"""
        try:
            max_id = int(np.max(self.event_id))
        except ValueError:
            max_id = 0
        for ev_id, ev_name in zip(hazard.event_id, hazard.event_name):
            try:
                found = self.event_name.index(ev_name)
                new_ev_pos.append(found)
                if ((ev_id in self.event_id) or (ev_id in new_id)) and \
                (ev_id != self.event_id[found]):
                    max_id += 1
                    self.event_id[found] = max_id
                else:
                    self.event_id[found] = ev_id
                    max_id = max(max_id, ev_id)
            except ValueError:
                new_ev_pos.append(len(self.event_id) + len(new_name))
                new_name.append(ev_name)
                if (ev_id in self.event_id) or (ev_id in new_id):
                    max_id += 1
                    new_id.append(max_id)
                else:
                    new_id.append(ev_id)
                    max_id = max(max_id, ev_id)
    
    def _event_plot(self, event_id, mat_var, col_name):
        """"Plot an event of the input matrix.

        Parameters
        ----------
            event_id (int): If event_id > 0, plot mat_var of
                event with id = event_id. If event_id = 0, plot maximum
                mat_var in each centroid. If event_id < 0, plot
                abs(event_id)-largest event.
            mat_var (sparse matrix): Sparse matrix where each row is an event
            col_name (sparse matrix): Colorbar label

        Returns
        -------
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if event_id > 0:
            try:
                event_pos = np.where(self.event_id == event_id)[0][0]
            except IndexError:
                LOGGER.error('Wrong event id: %s.', event_id)
                raise ValueError from IndexError
            array_val = mat_var[event_pos, :].todense().transpose()
            title = 'Event ID %s: %s' % (str(self.event_id[event_pos]), \
                                      self.event_name[event_pos])
        elif event_id < 0:
            max_inten = np.squeeze(np.asarray(np.sum(mat_var, axis=1)))
            event_pos = np.argpartition(max_inten, event_id)[event_id:]
            event_pos = event_pos[np.argsort(max_inten[event_pos])][0]
            array_val = mat_var[event_pos, :].todense().transpose()
            title = '%s-largest Event. ID %s: %s' % (np.abs(event_id), \
                str(self.event_id[event_pos]), self.event_name[event_pos])
        else:
            array_val = np.max(mat_var, axis=0).todense().transpose()
            title = '%s max intensity at each point' % self.tag.haz_type

        return plot.geo_im_from_array(self.centroids.coord, array_val, \
                                   col_name, title)

    def _centr_plot(self, centr_id, mat_var, col_name):
        """"Plot a centroid of the input matrix.

        Parameters
        ----------
            centr_id (int): If centr_id > 0, plot mat_var
                of all events at centroid with id = centr_id. If centr_id = 0,
                plot maximum mat_var of each event. If centr_id < 0,
                plot abs(centr_id)-largest centroid where highest mat_var
                are reached.
            mat_var (sparse matrix): Sparse matrix where each column represents
                a centroid
            col_name (sparse matrix): Colorbar label

        Returns
        -------
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if centr_id > 0:
            try:
                centr_pos = np.where(self.centroids.id == centr_id)[0][0]
            except IndexError:
                LOGGER.error('Wrong centroid id: %s.', centr_id)
                raise ValueError from IndexError
            array_val = mat_var[:, centr_pos].todense()
            title = 'Centroid ID %s: (%s, %s)' % (str(centr_id), \
                    self.centroids.coord[centr_pos, 0], \
                    self.centroids.coord[centr_pos, 1])
        elif centr_id < 0:
            max_inten = np.squeeze(np.asarray(np.sum(mat_var, axis=0)))
            centr_pos = np.argpartition(max_inten, centr_id)[centr_id:]
            centr_pos = centr_pos[np.argsort(max_inten[centr_pos])][0]
            array_val = mat_var[:, centr_pos].todense()
            title = '%s-largest Centroid. ID %s: (%s, %s)' % \
                (np.abs(centr_id), str(self.centroids.id[centr_pos]), \
                 self.centroids.coord[centr_pos, 0], \
                 self.centroids.coord[centr_pos, 1])
        else:
            array_val = np.max(mat_var, axis=1).todense()
            title = '%s max intensity at each event' % self.tag.haz_type

        graph = plot.Graph2D(title)
        graph.add_subplot('Event number', col_name)
        graph.add_curve(range(len(array_val)), array_val, 'b')
        graph.set_x_lim(range(len(array_val)))
        return graph.get_elems()
