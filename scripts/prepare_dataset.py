#!/usr/bin/env python3


from ddpm_memo.chexpert import prepare_dataset


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('chexpert_csv')
    parser.add_argument('memodata_path')
    args = parser.parse_args()

    prepare_dataset(args.chexpert_csv, args.memodata_path)


if __name__ == '__main__':
    main()
