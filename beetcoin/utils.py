
import tensorflow as tf


def initialize_device(device='TPU'):
    if device == 'TPU':
        print('Connecting to TPU...')
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            print('Running on TPU ', tpu.master())
        except ValueError:
            tpu = None

        if tpu:
            try:
                print('Initiasing TPU...')
                tf.config.experimental_connect_to_cluster(tpu)
                tf.tpu.experimental.initialize_tpu_system(tpu)
                strategy = tf.distribute.experimental.TPUStrategy(tpu)
                print('TPU initialised.')
            except:
                print('failed to initialise TPU.')
        else:
            device = 'GPU'

    if device != 'TPU':
        strategy = tf.distribute.get_strategy()

    if device == 'GPU':
        print('Number of GPUs available',
              len(tf.config.experimental.list_physical_devices('GPU'))
              )

    auto = tf.data.experimental.AUTOTUNE
    replicas = strategy.num_replicas_in_sync
    return device, auto, replicas

