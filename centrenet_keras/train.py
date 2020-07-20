"""Training the main object detection model."""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from centernet_aux.data import AVAIL_DATASETS
from centernet_aux.models import Generator, CenterNet
from keras.optimizers import Adam, SGD
import keras
import keras.backend as K
from datetime import date, timedelta
import tensorflow as tf
import os


def main(args: Namespace) -> None:
    """Run the main program.
    Arguments:
        args: The object containing the commandline arguments
    """
    
    """Once you installed the GPU version of Tensorflow, you don't have anything to do in Keras. As written in the Keras documentation. If you are running on the TensorFlow backend, your code will automatically run on GPU if any available GPU is detected. The below line is to check that the GPU is correctly detected
    """    
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


    # In keras, we need to develop generators that give augmented data on the fly
    train_generator = Generator(
        args.data_path+ "/annotations.csv",
        args.data_path + "/classes.csv",
    )
    
    num_classes = train_generator.num_classes()

    # Save each run into a directory by its timestamp. Need to see if callbacks work. Otherwise, use this to create summary
#     log_dir = setup_dirs(
#         dirs=[args.save_dir],
#         dirs_to_tstamp=[args.log_dir],
#         config=vars(args),
#         file_name=CONFIG,
#     )[0]
    
    obj = CenterNet(num_classes=num_classes, input_size=args.input_size,mode = "train")
    model, prediction_model = obj.forward() #need to check if this is working fine
    
    # load weights into the model
    print('Loading weights into the model')
    model.load_weights(args.pretrained_weights, by_name=True, skip_mismatch=True)

    
    #Need to think if there's any need to add module to freeze layers
    
    # compile model
    model.compile(optimizer=Adam(lr=1e-3), loss={'centernet_loss': lambda y_true, y_pred: y_pred})
    
    #callback definition
    callbacks = []
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=args.tensorboard_dir,
         histogram_freq=0,
         batch_size=args.batch_size,
         write_graph=True,
         write_grads=False,
         write_images=False,
         embeddings_freq=0,
         embeddings_layer_names=None,
         embeddings_metadata=None
    )
    callbacks.append(tensorboard_callback)
    
    checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                'checkpoints/{}'.format(str(date.today() + timedelta(days=0))),
                '{dataset}_{{epoch:02d}}_{{loss:.4f}}_{{val_loss:.4f}}.h5'.format(dataset=args.dataset)
            ),verbose=1
    )
    callbacks.append(checkpoint)
    model.fit_generator(# it's changed in the newer version
        generator=train_generator,
        steps_per_epoch=args.steps,
        initial_epoch=0,
        epochs=args.epochs,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        max_queue_size=args.max_queue_size
    )
#According to documentation    
# fit(
#     x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None,
#     validation_split=0.0, validation_data=None, shuffle=True, class_weight=None,
#     sample_weight=None, initial_epoch=0, steps_per_epoch=None,
#     validation_steps=None, validation_batch_size=None, validation_freq=1,
#     max_queue_size=10, workers=1, use_multiprocessing=False
# )

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Training CenterNet",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=AVAIL_DATASETS,
        default="number_plates",
        help="which dataset to use",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./datasets/number_plates",
        help="path to the dataset",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="the number of images in each batch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate for optimization",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints/",
        help="directory where to save model",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=5000,
        help="the frequency of saving the model",
    )
    parser.add_argument(
        "--record-steps",
        type=int,
        default=100,
        help="the frequency of recording summaries",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/",
        help="directory where to write event logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for all RNGs (set to -1 to disable seeding)",
    )
    parser.add_argument(
        "--backbone_resnet",
        type=str,
        default="resnet50",
        help="which version of resnet to use. Currently supported resnet50 and resnet101",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default="./checkpoints/ResNet-50-model.keras.h5",
        help="resume training from these weights",
    )
    parser.add_argument(
        '--epochs',
        help='Number of epochs to train.',
        type=int,
        default=200
    )
    parser.add_argument(
        '--steps',
        help='Number of steps per epoch.',
        type=int,
        default=10000
    )
    parser.add_argument(
        '--multiprocessing',
        help='Use multiprocessing in fit_generator.',
        action='store_true'
    )
    parser.add_argument(
        '--workers',
        help='Number of generator workers.',
        type=int,
        default=1
    )
    parser.add_argument(
        '--max-queue-size',
        help='Queue length for multiprocessing workers in fit_generator.',
        type=int,
        default=10
    )
    parser.add_argument(
        '--input-size',
        help='Rescale the image so the smallest side is min_side.',
        type=int,
        default=512
    )
    parser.add_argument(
        '--tensorboard-dir',
        help='Log directory for Tensorboard output',
        default='./logs/{}'.format(str(date.today() + timedelta(days=0))))
    main(parser.parse_args())