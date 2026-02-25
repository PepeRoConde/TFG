import torch

def get_device():

    if torch.cuda.is_available():
        print('Usando a gráfica NVIDIA')
        return torch.device("cuda")

    elif torch.backends.mps.is_available():
        return  torch.device("mps")
        print('Usando a gráfica do portatil')

    else:
        print("Usando a CPU, isto vai ser lento (revisa que todo esta ben, non deberías usala cpu)")
        return torch.device("cpu")
