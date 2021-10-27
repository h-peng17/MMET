
import argparse
import logging
import time 

parser = argparse.ArgumentParser(description='Multimodal Entity Tagging Rerank')

# dataset config 
parser.add_argument('--max_seq_length', default=64, type=int)
parser.add_argument('--recall_count', default=128, type=int)
parser.add_argument('--data_prefix', default='train_', type=str)

# train params config 
parser.add_argument('--num_train_epochs', default=100, type=int)
parser.add_argument('--batch_size_per_gpu', default=1, type=int)
parser.add_argument('--save_epoch', default=10, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--loss', default='focal', type=str,
                    help="Loss type")
parser.add_argument('--optim', default='sgd', type=str,
                    help="optimizer")


# model config 
parser.add_argument('--train', action='store_true')
parser.add_argument('--freeze_image_encoder', action='store_true')
parser.add_argument('--model', dest='model', type=str)
parser.add_argument('--ckpt', dest='ckpt', type=str)
parser.add_argument('--ckpt_dir', default='ckpt', type=str)
parser.add_argument('--encoder', default='br', type=str, 
                    help="Encoder type: Bert-Resnet or CLIP")

# other config 
parser.add_argument('--record_step', default=10, type=int)
parser.add_argument('--tcp', default='tcp://127.0.0.1:12345', type=str)
parser.add_argument('--expert_count', default=1, type=int)
parser.add_argument('--image_expert_count', default=1, type=int)

# distributed training config
parser.add_argument('--local_rank', dest='local_rank', default=-1, type=int)
parser.add_argument('--n_gpu', dest='n_gpu', type=int, default=4)


# parser args
args = parser.parse_args()

# set extra configs 
if args.data_prefix == 'train_':
    args.training = True 
else:
    args.training = False 

# logger 
# logging.basicConfig(filename='running.log', level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# ch = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# logger.addHandler(ch)

f = open("running.log", 'a+')

args_info = ''
for k in list(vars(args).keys()):
    args_info += '%s: %s\t' % (k, vars(args)[k])

cur_time = time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()) 
f.write(cur_time + '\n')
f.write(args_info + '\n')
f.write("-"*20 + '\n')
