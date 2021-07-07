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

Test ImpactFuncSet class.
"""

import unittest

from climada.hazard.relative_cropyield import RelativeCropyield
from climada.hazard.river_flood import RiverFlood
from climada.hazard.hazard_set import HazardSet

FLOOD = RiverFlood()
FLOOD.event_name = list(['2002', 'innundation7', 'test3'])

RELATIVE_CROPYIELD = RelativeCropyield()

class TestContainer(unittest.TestCase):
    """Test HazardSet as container."""

    def test_append(self):
        """Test append hazard correctly."""
        hazard_set1 = HazardSet()
        hazard_set1.append(haz=FLOOD, haz_id='flood1')
        hazard_set1.append(RELATIVE_CROPYIELD)
        hazard_set1.append(RELATIVE_CROPYIELD, 'crop2')
        hazard_set1.append(FLOOD, 'flood1')
        self.assertEqual(4, len(hazard_set1._data))
        self.assertEqual(list(['flood1', '1', 'crop2', '2']), list(hazard_set1._data.keys()))

        with self.assertRaises(TypeError) as cm:
            hazard_set1.append(45)
        self.assertIn("Input variable haz is not of type Hazard.", str(cm.exception))

    def test_remove_haz_pass(self):
        """Test remove_haz removes Hazards of HazardSet correctly."""
        hazard_set1 = HazardSet()
        hazard_set1.append(haz=FLOOD, haz_id='flood1')
        hazard_set1.append(RELATIVE_CROPYIELD)
        hazard_set1.append(RELATIVE_CROPYIELD, 'crop2')
        hazard_set1.append(haz=FLOOD)

        with self.assertLogs('climada.hazard.hazard_set',
                             level='WARNING') as cm:
            hazard_set1.get_haz('drought')
        self.assertIn("The hazard with haz_id drought is not contained "
                                   "in this hazard set.", cm.output[0])

        #remove a string
        hazard_set1.remove_haz('1')
        self.assertEqual(3, len(hazard_set1._data))
        self.assertEqual(list(['flood1', 'crop2', '2']), list(hazard_set1._data.keys()))

        #remove all hazards
        hazard_set1.remove_haz('all')
        self.assertEqual(0, len(hazard_set1._data))

    def test_get_haz(self):
        """Test get_haz gets Hazard of HazardSet correctly."""
        hazard_set1 = HazardSet()
        hazard_set1.append(haz=FLOOD, haz_id='flood1')
        hazard_set1.append(RELATIVE_CROPYIELD)
        hazard_set1.append(RELATIVE_CROPYIELD, 'crop2')
        hazard_set1.append(haz=FLOOD)

        haz = hazard_set1.get_haz('flood1')
        self.assertEqual(haz.tag.haz_type, 'RF')

        with self.assertLogs('climada.hazard.hazard_set',
                             level='WARNING') as cm:
            hazard_set1.get_haz('drought')
        self.assertIn("The hazard with haz_id drought is not contained in this hazard set.",
                      cm.output[0])

    def test_get_haz_types(self):
        """Test get_haz_types gets hazrd types of HazardSet correctly."""
        hazard_set1 = HazardSet()
        hazard_set1.append(haz=FLOOD, haz_id='flood1')
        hazard_set1.append(RELATIVE_CROPYIELD)
        hazard_set1.append(RELATIVE_CROPYIELD, 'crop2')
        hazard_set1.append(haz=FLOOD)

        #of a single hazard
        haz_type = hazard_set1.get_haz_types('flood1')
        self.assertEqual(['RF'], haz_type)

        #of all contained hazards
        haz_types = hazard_set1.get_haz_types()
        self.assertEqual(list(['RF', 'RC', 'RC', 'RF']), haz_types)

    def test_get_ids(self):
        """Test get_haz_ids returns hazard ids of HazardSet correctly."""
        hazard_set1 = HazardSet()
        hazard_set1.append(haz=FLOOD, haz_id='flood1')
        hazard_set1.append(RELATIVE_CROPYIELD)
        hazard_set1.append(RELATIVE_CROPYIELD, 'crop2')
        hazard_set1.append(haz=FLOOD)

        #of a certain hazard type
        haz_ids = hazard_set1.get_ids('RC')
        self.assertEqual(haz_ids, list(['1', 'crop2']))

        #of all hazards
        haz_ids = hazard_set1.get_ids()
        self.assertEqual(haz_ids, list(['flood1','1', 'crop2', '2']))

    def test_extend(self):
        """Test extend appends HazardSet by a second HazardSet correctly."""
        hazard_set1 = HazardSet()
        hazard_set1.append(haz=FLOOD, haz_id='flood1')
        hazard_set1.append(RELATIVE_CROPYIELD)

        hazard_set2 = HazardSet()
        hazard_set2.append(RELATIVE_CROPYIELD, 'crop2')
        hazard_set2.append(haz=FLOOD)

        self.assertEqual(2, len(hazard_set1._data))
        self.assertEqual(2, len(hazard_set2._data))

        hazard_set1.extend(hazard_set2)
        #hazards with the same id as in the extending hazard set get overwritten ('1')
        self.assertEqual(4, len(hazard_set1._data))
        self.assertEqual(list(['flood1', '1', 'crop2', '2']), list(hazard_set1._data.keys()))
        #assert the two hazards with same ide remain in the HazardSet and the second one has been renamed
        haz_type = hazard_set1.get_haz_types('1')
        self.assertEqual(haz_type, ['RC'])
        haz_type = hazard_set1.get_haz_types('2')
        self.assertEqual(haz_type, ['RF'])
    
    def test_select(self):
        """Test select allows to extract a part of the Hazards of HazardSet correctly."""
        
        
        


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestContainer)
    unittest.TextTestRunner(verbosity=2).run(TESTS)