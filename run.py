'''
run train/test/demo

Also, it should handle argument too.
'''
import argparse

# set configurations
parser = argparse.ArgumentParser()
parser.add_argument('--setting', help='Set configurations file path', type=str, default='experiments/setting/celeba_train_v0.yaml')

'''TODO: add more arguments'''

args = parser.parse_args()


