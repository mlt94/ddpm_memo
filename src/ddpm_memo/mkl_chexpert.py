import csv
import os.path
from pathlib import Path
import re
import ipdb

import yaml


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
            raise RuntimeError(f'Key "{k}" unexpectedly had no matches.')

    return '{patient}-{study}-{view}-{frontal_lateral}'.format_map(matchdict)


def make_prompt(point, codes):
    sentences = []
    no_evidence_sentences = []
    search_phrase = "no evidence"
    for key in COLUMNS:
        if key == 'Path':
            continue

        value = point[key]
        if key == 'Age':
            sentences.append(codes[key].format(value=value))
        else:            
            if search_phrase in codes[key][value].lower():
                mod_string = codes[key][value].lower().replace(search_phrase + " of", '').strip()
                no_evidence_sentences.append(mod_string.format(value=mod_string))
                continue
            sentences.append(codes[key][value].format(value=value))

    sentence = ', '.join(filter(None, sentences))
    no_evidence = ', '.join(filter(None, no_evidence_sentences))
    if len(no_evidence) != 0:
        final_string = (sentence + ' no evidence of ' + no_evidence)
        return final_string
    return sentence


def get_image_path(dataset_path, rel_path):
    # Remove initial 'CheXpert-v1.0-small' path component.
    # Then rel_path is relative to the directory of the CSV file.
    rel_path = Path(*Path(rel_path).parts[1:])

    return Path(dataset_path) / rel_path


def prepare_dataset(dataset_csv, outdir, codes_path):
    dataset_csv = Path(dataset_csv)
    outdir = Path(outdir)
    codes_path = Path(codes_path)

    with codes_path.open('rt') as file:
        codes = yaml.safe_load(file)

    with dataset_csv.open('rt') as file:
        next(file) # Assume that header content is COLUMNS.
        csvreader = csv.DictReader(file, COLUMNS)
        for point in csvreader:
            base_filename = name_from_path(point['Path'])

            # Path to the image file.
            src_path = get_image_path(dataset_csv.parent, point['Path'])

            base_path = outdir / base_filename

            # Make symlink to image file.
            
            link_target = os.path.relpath(src_path, outdir)
           
            link_target_sym = base_path.with_suffix('.jpg')
            if not link_target_sym.is_symlink():
                base_path.with_suffix('.jpg').symlink_to(link_target)

            # Produce a prompt based on the data in the CSV file.
            prompt = make_prompt(point, codes)

            # Dump prompt into text file.
            with base_path.with_suffix('.txt').open('w+') as file:
                content = file.read()             
                if len(content) > 77:
                    print(f"!!!CLIP will fail, on sentence!!!: \n {content}")
                file.write(prompt)
