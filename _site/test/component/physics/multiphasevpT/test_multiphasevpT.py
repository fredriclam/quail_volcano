import numpy as np
import pytest
import sys
sys.path.append('../src')

import physics.multiphasevpT.multiphasevpT as multiphasevpT


def test_basic():
    '''Verify that I can run a basic test.'''

    print("Hello world!")

    assert True