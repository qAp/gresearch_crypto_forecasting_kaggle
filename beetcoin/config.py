
from beetcoin.utils import initialize_device

DEVICE = 'TPU'

DEVICE, AUTO, REPLICAS = initialize_device(device=DEVICE)