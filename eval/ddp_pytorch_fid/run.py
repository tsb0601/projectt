from fid_score import main, parser
import torch_xla.distributed.xla_multiprocessing as xmp

if __name__ == '__main__':
    args = parser.parse_args()
    xmp.spawn(main, args=(args,), start_method='fork') # use fork to reproduce error in main codebase