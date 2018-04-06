import argparse

def argument_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size"  , type=int  , default=32)
    parser.add_argument("--epochs"      , type=int  , default=50)
    parser.add_argument("--train-data"  , type=str  , default='data/train.csv')
    parser.add_argument("--test-data"   , type=str  , default='data/test.csv')
    parser.add_argument("--outdir"      , type=str  , default='output')
    parser.add_argument("--mdl-outdir"  , type=str, default='data/test.csv')
    parser.add_argument("--outpkl", type=str, default='output')

    args = parser.parse_args()

    return args