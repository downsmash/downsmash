import unittest

loader = unittest.TestLoader()
suite = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule())

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)
