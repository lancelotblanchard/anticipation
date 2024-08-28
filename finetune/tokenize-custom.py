import os
from argparse import ArgumentParser
from multiprocessing import Pool, RLock
from glob import glob

from tqdm import tqdm

from anticipation.config import *
from anticipation.tokenize import tokenize, tokenize_ia

def main(args):
    encoding = 'interarrival' if args.interarrival else 'arrival'
    print('Tokenizing Custom MIDI Dataset')
    print(f'  encoding type: {encoding}')

    # Specific splits for my dataset structure
    split_names = ['Train', 'Test', 'Validation']
    print(f'  train split: {split_names[0]}')
    print(f'  validation split: {split_names[2]}')
    print(f'  test split: {split_names[1]}')

    print('Tokenization parameters:')
    print(f'  anticipation interval = {DELTA}s')
    print(f'  augment = {args.augment}x')
    print(f'  max track length = {MAX_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track length = {MIN_TRACK_TIME_IN_SECONDS}s')
    print(f'  min track events = {MIN_TRACK_EVENTS}')

    split_paths = [os.path.join(args.datadir, s) for s in split_names]
    files = [glob(f'{p}/*.compound.txt') for p in split_paths]
    outputs = [os.path.join(args.datadir, f'tokenized-events-{s}.txt') for s in split_names]

    # Augmentation settings
    augment = [args.augment if s == 'Train' else 1 for s in split_names]

    func = tokenize_ia if args.interarrival else tokenize
    with Pool(processes=PREPROC_WORKERS, initargs=(RLock(),), initializer=tqdm.set_lock) as pool:
        results = pool.starmap(func, zip(files, outputs, augment, range(len(split_names))))

    seq_count, rest_count, too_short, too_long, too_manyinstr, discarded_seqs, truncations \
            = (sum(x) for x in zip(*results))
    rest_ratio = round(100*float(rest_count)/(seq_count*M),2)

    trunc_type = 'interarrival' if args.interarrival else 'duration'
    trunc_ratio = round(100*float(truncations)/(seq_count*M),2)

    print('Tokenization complete.')
    print(f'  => Processed {seq_count} sequences')
    print(f'  => Inserted {rest_count} REST tokens ({rest_ratio}% of events)')
    print(f'  => Discarded {too_short+too_long+too_manyinstr} sequences for being out of bounds')
    print(f'      - {too_short} too short')
    print(f'      - {too_long} too long')
    print(f'      - {too_manyinstr} too many instruments')
    print(f'  => Discarded {discarded_seqs} sequences for other reasons')
    print(f'  => Truncated {truncations} {trunc_type} times ({trunc_ratio}% of {trunc_type}s)')
    print('Remember to shuffle the training split!')

if __name__ == '__main__':
    parser = ArgumentParser(description='Tokenizes a custom MIDI dataset')
    parser.add_argument('datadir', help='Directory containing preprocessed MIDI to tokenize')
    parser.add_argument('-k', '--augment', type=int, default=1,
                        help='Dataset augmentation factor (multiple of 10)')
    parser.add_argument('-i', '--interarrival',
                        action='store_true',
                        help='Request interarrival-time encoding (defaults to arrival-time encoding)')

    main(parser.parse_args())
