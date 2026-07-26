import torch

def get_device()->torch.device:
    """
    Returns the best available device.
    """

    if torch.cuda.is_available():
        return torch.device('cuda')

    return torch.device('cuda')


def print_device_info()->None:
    """
    Prints information about the selected device.
    """

    device = get_device()
    print('='*50)
    print(f'Using Device : {device}')

    if device.type == "cuda":
        print(f'GPU : {torch.cuda.get_device_name(0)}')

    print('='*50)