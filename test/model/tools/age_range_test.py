#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from main.model.tools.age_range import AgeRange


__author__ = 'Iván de Paz Centeno'


class AgeRangeTest(unittest.TestCase):
    """
    Tests for the AgeRange class.
    """

    def setUp(self):
        """
        Some basics set ups of ages and parameters.
        """
        #                   age1     age2     distance_intersection  percentage
        self.age_sets = [[[1, 30], [25, 60], 5,                     17.24],
                         [[5, 10], [30,60],  0,                     0.00]]

    def test_age_range_from_string(self):
        """
        Age_range class is able to be created from a string.
        """

        valid_age_ranges = [
            AgeRange.from_string("(5, 10)"),
            AgeRange.from_string("( 5, 10)"),
            AgeRange.from_string("( 5, 10 )"),
            AgeRange.from_string("(5,10)"),
            AgeRange.from_string("(      5,         10      )"),
            AgeRange.from_string("( 5, 10)"),
            AgeRange.from_string("5, 10)"),
        ]

        invalid_age_ranges = [
            AgeRange.from_string("[5, 10)"),
            AgeRange.from_string("( 5 10 )"),
            AgeRange.from_string("(5,)"),
            AgeRange.from_string("(      5,"),
            AgeRange.from_string("I am not a range"),
            AgeRange.from_string(""),
            AgeRange.from_string(None)
        ]

        for age_range in valid_age_ranges:
            self.assertTrue(age_range.is_valid())
            self.assertEqual(age_range.get_range()[0], 5)
            self.assertEqual(age_range.get_range()[1], 10)

        for age_range in invalid_age_ranges:
            self.assertFalse(age_range.is_valid())

    def test_get_distance(self):
        """
        Distance in an age range.
        """
        age_range = AgeRange(0, 100)
        self.assertEqual(age_range.get_distance(), 100)

        age_range = AgeRange(-5, -1)
        self.assertEqual(age_range.get_distance(), 4)

        age_range = AgeRange(-5, 5)
        self.assertEqual(age_range.get_distance(), 10)

        age_range = AgeRange(5, 10)
        self.assertEqual(age_range.get_distance(), 5)

        age_range = AgeRange(10, 10)
        self.assertEqual(age_range.get_distance(), 0)

        age_range = AgeRange(15, 10)
        self.assertEqual(age_range.get_distance(), -5)

    def test_get_mean(self):
        """
        Mean of an age range
        """
        age_range = AgeRange(0, 100)
        self.assertEqual(age_range.get_mean(), 50)

        age_range = AgeRange(-5, -1)
        self.assertEqual(age_range.get_mean(), -3)

        age_range = AgeRange(-5, 5)
        self.assertEqual(age_range.get_mean(), 0)

        age_range = AgeRange(5, 10)
        self.assertEqual(age_range.get_mean(), 7)

        age_range = AgeRange(10, 10)
        self.assertEqual(age_range.get_mean(), 10)

        age_range = AgeRange(15, 10)
        self.assertEqual(age_range.get_mean(), 12)

    def test_age_range_str(self):
        """
        Age range serialize as string
        """
        age = AgeRange(3, 5)
        self.assertEqual('{"Age_range": "(3, 5)"}', age.__str__())

        age = AgeRange.from_string("(3, 5)")
        self.assertEqual('{"Age_range": "(3, 5)"}', age.__str__())

    def test_intersection_with_age_range_class(self):
        """
        Intersection with other ranges of ages.
        """
        for age_set in self.age_sets:
            age1 = age_set[0]
            age2 = age_set[1]
            expected_distance = age_set[2]
            expected_percentage = age_set[3]

            age_range1 = AgeRange(*age1)
            age_range2 = AgeRange(*age2)

            intersection = age_range1.intersect_with(age_range2)
            intersection_distance = intersection.get_distance()

            # Which one is the smaller rectangle?
            distance_age1 = age_range1.get_distance()
            distance_age2 = age_range2.get_distance()

            lesser_distance = min(distance_age1, distance_age2)
            percentage = round((intersection_distance / lesser_distance) * 100, 2)

            self.assertEqual(intersection_distance, expected_distance)
            self.assertEqual(percentage, expected_percentage)


if __name__ == '__main__':
    unittest.main()
