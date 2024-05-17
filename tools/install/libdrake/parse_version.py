"""This script parses VERSION.txt and produces a CMake listfile
which specifies the variable substitutions needed for drake-config.cmake."""

import argparse


def _parse_version_txt(path):
    contents = path.readline().split()
    assert len(contents) == 2, contents
    drake_version, git_revision = contents
    version_parts = drake_version.split('.')
    assert len(version_parts) >= 3, contents
    return version_parts[:3]


def _write_version_info(dest, version_parts):
    version_full = '.'.join(version_parts)
    dest.write(f'set(DRAKE_VERSION "{version_full}")\n')
    dest.write(f'set(DRAKE_MAJOR_VERSION "{version_parts[0]}")\n')
    dest.write(f'set(DRAKE_MINOR_VERSION "{version_parts[1]}")\n')
    dest.write(f'set(DRAKE_BUILD_VERSION "{version_parts[2]}")\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', type=argparse.FileType('r'),
        help='Path to input `VERSION.TXT`.')
    parser.add_argument(
        'output', type=argparse.FileType('w'),
        help='Path to output file.')
    args = parser.parse_args()
    version_parts = _parse_version_txt(args.input)
    _write_version_info(args.output, version_parts)
    return 0


if __name__ == '__main__':
    main()
