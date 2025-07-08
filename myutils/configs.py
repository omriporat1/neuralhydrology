from collections import defaultdict
from typing import Literal
from myutils.general import *
import os
import pandas as pd
import yaml
from neuralhydrology.utils.config import Config

def build_basins_config(src_path, dst_path):

    root_dir = os.path.join(src_path, 'timeseries', 'netcdf')
    basins = {}

    for directory in os.listdir(root_dir):
        basins[directory] = []
        for file in os.listdir(os.path.join(root_dir, directory)):
            if file.endswith(".nc"):
                basins[directory].append(file[:-3])

    save_config(dst_path, 'all_basins', basins)

def build_attr_config(src_path, dst_path):

    attr = {'ALL_KEYS': {'per_gauge': ['gauge_id', 'gauge_name', 'country'], 'per_sample': ['date']}, 'STATIC_ATTRIBUTES': {}, 'DYNAMIC_ATTRIBUTES': {}}

    dynamic_path = os.path.join(src_path, 'timeseries', 'csv')
    static_path = os.path.join(src_path, 'attributes')


    sample_path = os.path.join(static_path, 'camels')
    for file in os.listdir(sample_path):
        if file.endswith(".csv"):
            parts = file.split('_')
            attr['STATIC_ATTRIBUTES'][parts[-2]] = {'sample_path': os.path.join('attributes', 'camels', file), 'SIZE': 0, 'KEYS': [], 'names': []}

    sample_path = os.path.join(dynamic_path, 'camels')
    file = os.listdir(sample_path)[0]
    if file.endswith(".csv"):
        parts = file.split('_')
        attr['DYNAMIC_ATTRIBUTES'] = {'sample_path': os.path.join('timeseries', 'csv', 'camels', file), 'SIZE': 0, 'KEYS': [], 'names': []}

    for key, value in attr['STATIC_ATTRIBUTES'].items():

        df = pd.read_csv(os.path.join(src_path, value['sample_path']))

        names = df.columns.to_list()
        keys = []

        for key in attr['ALL_KEYS']['per_gauge']:
            try: 
                names.remove(key)
                keys.append(key)
            except ValueError:
                pass

        value['KEYS'] = keys
        value['SIZE'] = len(names)
        value['names'] = names

    key = 'DYNAMIC_ATTRIBUTES'
    value = attr[key]

    df = pd.read_csv(os.path.join(src_path, value['sample_path']))

    names = df.columns.to_list()
    keys = []

    for key in attr['ALL_KEYS']['per_sample']:
        try: 
            names.remove(key)
            keys.append(key)
        except ValueError:
            pass

    value['KEYS'] = keys
    value['SIZE'] = len(names)
    value['names'] = names


    # print(attr)
    save_config(dst_path, 'all_attributes', attr)

    return attr

class MyDumper(yaml.Dumper):

    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)
    
    def write_line_break(self, data=None):
        super().write_line_break(data)
        if (self.indent == 0):
            self.stream.write('\n')

def save_config(path, name, config, overwrite=False):
    file_name = name + '.yaml'
    if not overwrite:
        file_name = get_free_name(path, name, '.yaml')
    yaml.dump(config, Path(path, file_name).open('w'), Dumper=MyDumper, default_flow_style=False, sort_keys=False)

def build_save_config(config: Config):
    saved_conf = config._cfg.copy()
    
    for key, val in saved_conf.items():
        if isinstance(val, Path):
            saved_conf[key] = str(val).replace('\\', '/')            
        elif isinstance(val, pd.Timestamp):
            saved_conf[key] = val.strftime('%d/%m/%Y')
    
    return saved_conf


def create_run_config(config: Config, strip=False, layout_basins: Literal['full', 'organized', None] = 'organized'):
    saved_conf = build_save_config(config)

    if strip:
        saved_conf.pop('experiment_name')
        saved_conf.pop('run_dir')

    if layout_basins == 'full' or layout_basins == 'organized':
        categories = {
            'train': config.train_basin_file,
            'validation': config.validation_basin_file,
            'test': config.test_basin_file
        }
        basin_files = set([categories['train'], categories['validation'], categories['test']])
        basins = {file: [] for file in basin_files}

        for file_name, basins_in_file in basins.items():
            with open(file_name, 'r') as f:
                for line in f:
                    basins_in_file.append(line.strip())
        
        if layout_basins == 'organized':
            for file_name in basins.keys():    
                all_basins: dict = yaml.safe_load(open('configs/all_basins.yaml', 'r'))
                concise_basins = {}
                for set_name, basins_in_set in all_basins.items():
                    all_basins_set = set(basins_in_set)
                    my_basins_set = set(basins[file_name])
                    if all_basins_set.issubset(my_basins_set):
                        concise_basins[set_name] = 'all'
                    else:
                        concise_basins[set_name] = list(all_basins_set.intersection(my_basins_set))
                basins[file_name] = concise_basins

        

        indices = defaultdict(list)
        for key, value in categories.items():
            indices[value].append(key)
        
        saved_conf['basins'] = {}
        for key, value in indices.items():
            category_name = ' & '.join(value)
            saved_conf['basins'][category_name] = basins[key]

    return saved_conf
    

def create_run_dir(config: Config):
    if config.run_dir is None:
        config.run_dir = Path('runs')

    new_run_path = str(config.run_dir / config.experiment_name)
    os.makedirs(new_run_path, exist_ok=True)

    config.run_dir = Path(new_run_path)

def add_run_config(cfg: Config, mode: Literal['full', 'organized', None] = 'full'):
    # Create the base directory if it doesn't exist
    configs_path = cfg.run_dir / 'full_configs'
    os.makedirs(str(configs_path), exist_ok=True)
    save_config(str(configs_path), 'run_config', create_run_config(cfg, strip=True, layout_basins=mode), overwrite=False)

def get_last_run_config(config: Config):
    folder = get_last_run(config)
    print(Path(config.run_dir, folder, 'run_config.yaml'))
    return Config(Path(config.run_dir, folder, 'run_config.yaml'))

def generate_basins_txt(selector: dict, src_path, dst_path):
    
    src_basins = yaml.safe_load(open(src_path, 'r'))
    basins = []
    for key, value in selector.items():
        if value == 'all':
            basins.extend(src_basins[key])
        else:
            basins.extend(value)
    
    with open(dst_path, 'w') as f:
        for basin in basins:
            f.write(basin + '\n')