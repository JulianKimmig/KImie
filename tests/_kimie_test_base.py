import unittest

from KImie.utils.sys import enter_test_mode


class KImieTest(unittest.TestCase):
    def setUp(self) -> None:
        enter_test_mode()
