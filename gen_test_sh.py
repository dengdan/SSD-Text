import sys
import util

path = util.io.get_absolute_path(sys.argv[1])
start = int(sys.argv[2])

ckpts = util.tf.get_all_ckpts(path);
for ckpt in ckpts:
    iter_ = util.tf.get_iter(ckpt);
    if iter_ >= start:
        print "./scripts/run.sh 0 test test %s"%(ckpt)
