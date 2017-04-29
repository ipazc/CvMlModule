#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from main.model.tools.gender import Gender, GENDER_FEMALE, GENDER_MALE, GENDER_UNKNOWN


__author__ = 'Iván de Paz Centeno'


class GenderTest(unittest.TestCase):
    """
    Tests for the Gender class.
    """

    def test_gender_from_string(self):
        """
        Gender class is able to be created from a string.
        """

        valid_genders = {
            Gender.from_string("f"): GENDER_FEMALE,
            Gender.from_string("female"): GENDER_FEMALE,
            Gender.from_string("FEMALE"): GENDER_FEMALE,
            Gender.from_string("m"): GENDER_MALE,
            Gender.from_string("Male"): GENDER_MALE,
            Gender.from_string("MALE"): GENDER_MALE
        }

        invalid_genders = [
            Gender.from_string(" F"),
            Gender.from_string("fe ma le"),
            Gender.from_string("fem"),
            Gender.from_string(" M a L E"),
            Gender.from_string("          "),
            Gender.from_string("\n"),
            Gender.from_string(""),
            Gender.from_string(None)
        ]

        for gender, expected_value in valid_genders.items():
            self.assertTrue(gender.is_valid())
            self.assertEqual(gender.get_gender(), expected_value)

        for gender in invalid_genders:
            self.assertFalse(gender.is_valid())

    def test_get_gender(self):
        """
        Gender is able to retrieve the gender component.
        """
        gender = Gender(GENDER_FEMALE)
        self.assertEqual(gender.get_gender(), GENDER_FEMALE)

        gender = Gender(GENDER_MALE)
        self.assertEqual(gender.get_gender(), GENDER_MALE)

        gender = Gender(GENDER_UNKNOWN)
        self.assertEqual(gender.get_gender(), GENDER_UNKNOWN)

    def test_str(self):
        """
        Gender is able to describe itself when invoking __str__()
        """
        gender = Gender(GENDER_FEMALE)
        self.assertEqual(gender.__str__(), '{"Gender": "Female"}')

        gender = Gender(GENDER_MALE)
        self.assertEqual(gender.__str__(), '{"Gender": "Male"}')

        gender = Gender(GENDER_UNKNOWN)
        self.assertEqual(gender.__str__(), '{"Gender": "Unknown"}')


if __name__ == '__main__':
    unittest.main()
