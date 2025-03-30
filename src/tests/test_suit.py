import pytest
#from test_qdrant import *
from test_data_manager import *
from test_model_manager import *
from test_service import *
from test_metrics import *

def run_all_tests():
    pytest.main(['-v', '-s'])

if __name__ == '__main__':
    run_all_tests()
