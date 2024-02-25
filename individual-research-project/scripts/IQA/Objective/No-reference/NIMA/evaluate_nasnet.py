import numpy as np
import argparse
from path import Path

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from utils.nasnet import NASNetMobile, preprocess_input
from utils.score_utils import mean_score, std_score

# from tensorflow.keras import backend

# def set_image_data_format(data_format):
#     """Set the image data format.

#     Parameters:
#     - data_format: Either 'channels_first' or 'channels_last'.

#     Raises:
#     - ValueError: If the specified data_format is not valid.
#     """
#     valid_data_formats = {'channels_first', 'channels_last'}

#     if data_format not in valid_data_formats:
#         raise ValueError(f"Invalid data_format: {data_format}. Use 'channels_first' or 'channels_last'.")

#     global _image_data_format
#     _image_data_format = data_format

# def get_image_data_format():
#     """Get the current image data format."""
#     return _image_data_format if hasattr(globals(), '_image_data_format') else 'channels_last'

parser = argparse.ArgumentParser(description='Evaluate NIMA(Inception ResNet v2)')
parser.add_argument('-dir', type=str, default=None,
                    help='Pass a directory to evaluate the images in it')

parser.add_argument('-img', type=str, default=[None], nargs='+',
                    help='Pass one or more image paths to evaluate them')

parser.add_argument('-rank', type=str, default='true',
                    help='Whether to tank the images after they have been scored')

args = parser.parse_args()
target_size = (224, 224)  # NASNet requires strict size set to 224x224
rank_images = args.rank.lower() in ("true", "yes", "t", "1")

# give priority to directory
if args.dir is not None:
    print("Loading images from directory : ", args.dir)
    imgs = Path(args.dir).files('*.png')
    imgs += Path(args.dir).files('*.jpg')
    imgs += Path(args.dir).files('*.jpeg')

elif args.img[0] is not None:
    print("Loading images from path(s) : ", args.img)
    imgs = args.img

else:
    raise RuntimeError('Either -dir or -img arguments must be passed as argument')

with tf.device('/CPU:0'):
    base_model = NASNetMobile((224, 224, 3), include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('weights/nasnet_weights.h5')

    score_list = []

    for img_path in imgs:
        img = load_img(img_path, target_size=target_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # Set the image data format
        # backend.set_image_data_format('channels_last')  # Or 'channels_first' depending on your preference
        # set_image_data_format('channels_last')

        x = preprocess_input(x)

        scores = model.predict(x, batch_size=1, verbose=0)[0]

        mean = mean_score(scores)
        std = std_score(scores)

        file_name = Path(img_path).name.lower()
        score_list.append((file_name, mean))

        print("Evaluating : ", img_path)
        print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
        print()

    if rank_images:
        print("*" * 40, "Ranking Images", "*" * 40)
        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)

        for i, (name, score) in enumerate(score_list):
            print("%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))


