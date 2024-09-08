import traceback
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob

from tqdm import tqdm

from anticipation.convert import midi_to_compound
from anticipation.config import PREPROC_WORKERS, TIME_RESOLUTION

def addDrumToCompound(tokens):
    it = iter(tokens)

    BEAT_LENGTH = TIME_RESOLUTION // 2
    BAR_LENGTH = BEAT_LENGTH * 4

    newTokens = []
    nextBeat = 0

    for (time_in_ticks,duration,note,instrument,velocity) in zip(it,it,it,it,it):
        if time_in_ticks >= nextBeat:
            if nextBeat % BAR_LENGTH == 0:
                newTokens.append(nextBeat)
                newTokens.append(BAR_LENGTH-1)
                newTokens.append(36)
                newTokens.append(128)
                newTokens.append(100)
            newTokens.append(nextBeat)
            newTokens.append(BEAT_LENGTH-1)
            newTokens.append(42)
            newTokens.append(128)
            newTokens.append(100)
            nextBeat += BEAT_LENGTH
        newTokens.append(time_in_ticks)
        newTokens.append(duration)
        newTokens.append(note)
        newTokens.append(instrument)
        newTokens.append(velocity)

    return newTokens

def convert_midi(filename, addDrum=False, debug=False):
    try:
        tokens = midi_to_compound(filename, debug=debug)
        if addDrum:
            tokens = addDrumToCompound(tokens)
    except Exception:
        if debug:
            print('Failed to process: ', filename)
            print(traceback.format_exc())

        return 1

    with open(f"{filename}.compound.txt", 'w') as f:
        f.write(' '.join(str(tok) for tok in tokens))

    return 0


def main(args):
    filenames = glob(args.dir + '/**/*.mid', recursive=True) \
            + glob(args.dir + '/**/*.midi', recursive=True)
    
    convert_midi_partial = partial(convert_midi, addDrum=args.add_drum)

    print(f'Preprocessing {len(filenames)} files with {PREPROC_WORKERS} workers')
    with ProcessPoolExecutor(max_workers=PREPROC_WORKERS) as executor:
        results = list(tqdm(executor.map(convert_midi_partial, filenames), desc='Preprocess', total=len(filenames)))

    discards = round(100*sum(results)/float(len(filenames)),2)
    print(f'Successfully processed {len(filenames) - sum(results)} files (discarded {discards}%)')

if __name__ == '__main__':
    parser = ArgumentParser(description='prepares a MIDI dataset')
    parser.add_argument('dir', help='directory containing .mid files for training')
    parser.add_argument('--add-drum', help='Add a drum track underneath the files', action="store_true")
    main(parser.parse_args())
