import wfdb
import numpy as np
import pandas as pd

from glob import glob
import argparse
import os


def gen_reference_csv(data_dir, reference_csv):
    if not os.path.exists(reference_csv):
        # find all .hea files under data_dir
        recordpaths = glob(os.path.join(data_dir, '*.hea'))
        results = []
        for recordpath in recordpaths:
            # patient_id ← filename without directory or extension
            patient_id = os.path.splitext(os.path.basename(recordpath))[0]

            # strip off the '.hea' to pass to rdsamp
            record_name = os.path.splitext(recordpath)[0]
            _, meta_data = wfdb.rdsamp(record_name)

            sample_rate = meta_data['fs']
            signal_len = meta_data['sig_len']
            age_comment = meta_data['comments'][0]
            sex_comment = meta_data['comments'][1]
            dx_comment  = meta_data['comments'][2]

            # parse out values
            age = age_comment[5:] if age_comment.lower().startswith('age: ') else np.NaN
            sex = sex_comment[5:] if sex_comment.lower().startswith('sex: ') else 'Unknown'
            dx  = dx_comment[4:] if dx_comment.lower().startswith('dx: ') else ''

            results.append([patient_id, sample_rate, signal_len, age, sex, dx])

        df = pd.DataFrame(results, columns=[
            'patient_id', 'sample_rate', 'signal_len', 'age', 'sex', 'dx'
        ])
        df.sort_values('patient_id').to_csv(reference_csv, index=False)


def gen_label_csv(label_csv, reference_csv, dx_dict, classes):
    if not os.path.exists(label_csv):
        results = []
        df_reference = pd.read_csv(reference_csv)

        for _, row in df_reference.iterrows():
            patient_id = row['patient_id']
            # map each code to its label name (or '' if missing)
            dxs = [dx_dict.get(code, '') for code in str(row['dx']).split(',')]
            labels = [1 if cls in dxs else 0 for cls in classes]
            results.append([patient_id] + labels)

        df = pd.DataFrame(results, columns=['patient_id'] + classes)

        # assign 10-fold split
        n = len(df)
        folds = np.zeros(n, dtype=np.int8)
        for i in range(10):
            start = int(n * i / 10)
            end   = int(n * (i + 1) / 10)
            folds[start:end] = i + 1

        df['fold'] = np.random.permutation(folds)

        # keep only those with at least one positive label
        df['keep'] = df[classes].sum(axis=1)
        df = df[df['keep'] > 0]

        df[['patient_id'] + classes + ['fold']].to_csv(label_csv, index=False)


if __name__ == "__main__":
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    dx_dict = {
        '426783006': 'SNR',  # Normal sinus rhythm
        '164889003': 'AF',   # Atrial fibrillation
        '270492004': 'IAVB', # First-degree atrioventricular block
        '164909002': 'LBBB', # Left bundle branch block
        '713427006': 'RBBB', # Complete right bundle branch block
        '59118001' : 'RBBB', # Right bundle branch block
        '284470004': 'PAC',  # Premature atrial contraction
        '63593006' : 'PAC',  # Supraventricular premature beats
        '164884008': 'PVC',  # Ventricular ectopics
        '429622005': 'STD',  # ST-segment depression
        '164931005': 'STE',  # ST-segment elevation
    }
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        default=r'data\CPSC',  # you can use Windows style or POSIX style here
        help='Directory to dataset'
    )
    args = parser.parse_args()

    data_dir     = args.data_dir
    reference_csv = os.path.join(data_dir, 'reference.csv')
    label_csv     = os.path.join(data_dir, 'labels.csv')

    gen_reference_csv(data_dir, reference_csv)
    gen_label_csv(label_csv, reference_csv, dx_dict, classes)
