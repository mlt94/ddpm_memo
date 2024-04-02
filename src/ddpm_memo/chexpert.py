import csv
from pathlib import Path
import re


COLUMNS = [
    'Path',
    'Sex',
    'Age',
    'Frontal/Lateral',
    'AP/PA',
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
]


NAME_RE = re.compile(
    'patient(?P<patient>[0-9]+)'
    '/'
    'study(?P<study>[0-9]+)'
    '/'
    'view(?P<view>[0-9]+)'
    '_'
    '(?P<frontal_lateral>[fl])[a-z]+'
    '.jpg$'
)


def name_from_path(path):
    match = NAME_RE.search(str(path))

    if match is None:
        raise RuntimeError(f'Unexpected path format: {path}')

    matchdict = match.groupdict()

    for k, v in matchdict.items():
        if not v:
            raise RuntimeError(f'Key "{key}" unexpectedly had no matches.')

    return '{patient}-{study}-{view}-{frontal_lateral}'.format_map(matchdict)


def prepare_dataset(dataset_csv, outdir, img_prefix=''):
    dataset_csv = Path(dataset_csv)
    outdir = Path(outdir)

    with dataset_csv.open('rt') as file:
        next(file) # Assume that header content is COLUMNS.
        csvreader = csv.DictReader(file, COLUMNS)
        for point in csvreader:
            base_filename = name_from_path(point['Path'])
