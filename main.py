import os
import sys
from utils import Tester

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    # Tester.test_lenet_300_100()
    # Tester.test_lenet_1epoch()
    # Tester.test_lenet_5()
    # Tester.test_masked_lstm(128, "res/lstm_masked-b.png")  # lstm-a
    # Tester.test_masked_lstm(256, "res/lstm_masked-b.png")  # lstm-b
    # Tester.test_masked_vgg16()
    Tester.test_masked_wideres(8)
    #Tester.test_model()
