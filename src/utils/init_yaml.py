import os
import yaml

def init_yaml(file_name, args):

    yaml_path = f"{args.runs_dir}/metadata/{file_name}.yaml"
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

    with open(yaml_path, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False, sort_keys=False)
