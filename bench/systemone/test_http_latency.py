import unittest

from http_latency import percentile


class PercentileTests(unittest.TestCase):
    def test_nearest_rank(self):
        self.assertEqual(percentile([5, 1, 4, 2, 3], .5), 3)
        self.assertEqual(percentile([5, 1, 4, 2, 3], .95), 5)
        self.assertEqual(percentile([7], .99), 7)


if __name__ == "__main__":
    unittest.main()
