# coding=utf-8
import os
import re
import argparse
import hashlib
import shutil
import glob  # 用于替代 gfile.Glob

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'


def calc_hash_value(filename: str) -> int:
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()

    hash_int = int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)
    return hash_int


def copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def split_dataset_by_count(
    src_root: str,
    dst_root: str,
    validation_num: int,
    testing_num: int,
    rl_train_num: int,
):
    if not os.path.isdir(src_root):
        raise ValueError('No dir: {}'.format(src_root))

    search_path = os.path.join(src_root, '*', '*.wav')
    wav_files = glob.glob(search_path)

    if not wav_files:
        raise RuntimeError('No wav files'.format(search_path))

    items = []
    for wav_path in wav_files:
        label_dir = os.path.basename(os.path.dirname(wav_path))
        if label_dir == BACKGROUND_NOISE_DIR_NAME:
            continue
        h = calc_hash_value(wav_path)
        items.append((h, wav_path))

    items.sort(key=lambda x: x[0])

    total_samples = len(items)

    required = validation_num + testing_num + rl_train_num
    if required > total_samples:
        raise ValueError("Not enough samples: required {}, but only {} available.".format(required, total_samples))

    v_end = validation_num
    t_end = v_end + testing_num
    r_end = t_end + rl_train_num

    counts = {
        'training': 0,
        'validation': 0,
        'testing': 0,
        'rl_train': 0,
    }

    for idx, (_, wav_path) in enumerate(items):
        label_dir = os.path.basename(os.path.dirname(wav_path))
        rel_filename = os.path.basename(wav_path)

        if idx < v_end:
            set_name = 'validation'
        elif idx < t_end:
            set_name = 'testing'
        elif idx < r_end:
            set_name = 'rl_train'
        else:
            set_name = 'training'

        dst_path = os.path.join(dst_root, set_name, label_dir, rel_filename)
        copy_file(wav_path, dst_path)
        counts[set_name] += 1

    src_bg_dir = os.path.join(src_root, BACKGROUND_NOISE_DIR_NAME)
    if os.path.isdir(src_bg_dir):
        dst_bg_dir = os.path.join(dst_root, BACKGROUND_NOISE_DIR_NAME)
        os.makedirs(dst_bg_dir, exist_ok=True)
        bg_search_path = os.path.join(src_bg_dir, '*.wav')
        for wav_path in glob.glob(bg_search_path):
            rel_filename = os.path.basename(wav_path)
            dst_path = os.path.join(dst_bg_dir, rel_filename)
            copy_file(wav_path, dst_path)

    return counts, total_samples


def parse_args():
    parser = argparse.ArgumentParser(
        description=('Split a dataset of .wav files into training, validation, testing, and rl_train sets based on specified counts. The script organizes the files into subdirectories for each set and maintains the original label directory structure. It also handles background noise files separately.')
    )
    parser.add_argument(
        '--src_dir',
        type=str,
        default='/data/gsc',
    )
    parser.add_argument(
        '--dst_dir',
        type=str,
        default='/data/gsc_splited',
    )
    parser.add_argument(
        '--validation_num',
        type=int,
        default=10000,
    )
    parser.add_argument(
        '--testing_num',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--rl_train_num',
        type=int,
        default=15000,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.isdir(args.dst_dir):
        for name in os.listdir(args.dst_dir):
            path = os.path.join(args.dst_dir, name)
            try:
                if os.path.islink(path) or os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(f"Delete {path} failed: {e}")
    else:
        os.makedirs(args.dst_dir, exist_ok=True)

    print('Source directory: {}'.format(args.src_dir))
    print('Destination directory: {}'.format(args.dst_dir))
    print(
        'Target sample counts: validation={}, testing={}, rl_train={}'.format(
            args.validation_num, args.testing_num, args.rl_train_num
        )
    )

    counts, total = split_dataset_by_count(
        src_root=args.src_dir,
        dst_root=args.dst_dir,
        validation_num=args.validation_num,
        testing_num=args.testing_num,
        rl_train_num=args.rl_train_num,
    )
    print('Dataset splitting completed. Total samples (excluding _background_noise_):', total)
    print('Sample counts for each subset:', counts)


if __name__ == '__main__':
    main()