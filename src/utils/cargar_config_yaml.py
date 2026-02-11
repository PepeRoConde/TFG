import yaml

def cargar_config_yaml(checkpoint_path):

    checkpoint_name = checkpoint_path.replace('data/weights/','').replace('.pth.tar','')
    yaml_path = f"data/runs/metadata/{checkpoint_name}.yaml"
    
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            print(f"Configuración cargada dende: {yaml_path}")
            return config
    except FileNotFoundError:
        print(f"VAITES, o arquivo: {yaml_path} non existe")
        sys.exit(1)
    except Exception as e:
        print(f"VAITES ao ler o YAML: {e}")
        sys.exit(1)
