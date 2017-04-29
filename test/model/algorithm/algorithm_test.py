import unittest

from main.model.algorithm.algorithm import Algorithm
from main.model.resource.resource import Resource


class AlgorithmTest(unittest.TestCase):
    """
    Algorithm should be a virtual class, however we need to test the inheritable methods.
    """

    def test_name(self):
        """
        Algorithm's name is successfully stored.
        """
        algorithm = Algorithm("test", "test description")
        self.assertEqual(algorithm.get_name(), "test")

    def test_description(self):
        """
        Algorithm's description is successfully stored.
        """
        algorithm = Algorithm("test", "test description")
        self.assertEqual(algorithm.get_description(), "test description")

    def test_str(self):
        """
        Algorithm correctly describes itself when __str__() invoked
        """
        algorithm = Algorithm("test", "test description")
        self.assertEqual(algorithm.__str__(),
                         "[Algorithm {}: \"{}\"; admits resources of type \"{}\"]".format(
                             "test", "test description", algorithm.kind_of_resource()))

    def test_generate_uri(self):
        """
        Algorithm generates new URIs successfully.
        """
        algorithm = Algorithm("test", "test description")
        resource = Resource(uri="/tmp/test")
        new_uri = algorithm.__generate_new_uri__(resource)

        self.assertEqual(new_uri, "/tmp_test/test")

if __name__ == '__main__':
    unittest.main()
