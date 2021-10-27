
import argparse
from ast import parse

parser = argparse.ArgumentParser(description='Multimodal Entity Tagging')
parser.add_argument('--data_path', dest='data_path')
parser.add_argument('--img_save_name', dest='img_save_name')
parser.add_argument('--prefix', dest='prefix')

parser.add_argument('--mode', dest='mode')
parser.add_argument('--local_rank', dest='local_rank', default=-1, type=int)

args = parser.parse_args()
print(args)