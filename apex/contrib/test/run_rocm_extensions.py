import argparse
import os
import unittest
import sys


#test_dirs = ["fused_dense", "layer_norm", "multihead_attn", "transducer", "focal_loss", "index_mul_2d", "optimizers", ".", "groupbn"] # "." for test_label_smoothing.py
#ROCM_BLACKLIST = [
#    "layer_norm"
#]

TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
TEST_DIRS = [
    "fused_dense",
    "layer_norm",    # not fully supported on ROCm
    "conv_bias_relu",# not fully supported on ROCm
    "fmha",          # not fully supported on ROCm
    #"cudnn_gbn",    # not fully supported on ROCm
    #"bottleneck",   # not fully supported on ROCm
    "multihead_attn",
    "transducer",
    "focal_loss",
    "index_mul_2d",
    "optimizers",
    "xentropy",
    "clip_grad",
    "groupbn",
]

DEFAULT_TEST_DIRS = [
    "fused_dense",
    "multihead_attn",
    "transducer",
    "focal_loss",
    "index_mul_2d",
    "optimizers",
    "xentropy",
    "clip_grad",
    "groupbn",
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extension test runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--include",
        nargs="+",
        choices=TEST_DIRS,
        default=DEFAULT_TEST_DIRS,
        help="select a set of tests to run (defaults to ALL tests).",
    )
    args, _ = parser.parse_known_args()
    return args

def main(args):
    runner = unittest.TextTestRunner(verbosity=2)
    errcode = 0
    for test_dir in args.include:
        test_dir = os.path.join(TEST_ROOT, test_dir)
        print(test_dir)
        suite = unittest.TestLoader().discover(test_dir)

        print("\nExecuting tests from " + test_dir)

        result = runner.run(suite)

        if not result.wasSuccessful():
            errcode = 1

    sys.exit(errcode)

if __name__ == '__main__':
    args = parse_args()
    main(args)
