#test
import os
import pandas as pd
import shutil
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn import preprocessing


def get_data_transform():
    # base de dados
    lfw_allnames = pd.read_csv("lfw-dataset/lfw_allnames.csv")
    # Monta um DF como caminho para cada imagem da pessoa
    # Por exemplo se há 55 imagens da mesma pessoa é montado o DF dos caminhos para as 55 imagens dela
    image_paths = lfw_allnames.loc[lfw_allnames.index.repeat(lfw_allnames['images'])]
    image_paths['image_path'] = 1 + image_paths.groupby('name').cumcount()
    image_paths['image_path'] = image_paths.image_path.apply(lambda x: '{0:0>4}'.format(x))
    image_paths['image_path'] = image_paths.name + "/" + image_paths.name + "_" + image_paths.image_path + ".jpg"
    image_paths = image_paths.drop("images", 1)
    return image_paths

def get_min_img(image_paths, min_img):
    ind_counts = image_paths.groupby('name').count().image_path
    image_list = []
    for img in ind_counts[ind_counts >= min_img].iteritems():
        image_list.append(img[0])
    return image_list

def mount_data(pd_df, min_img, sample_size):
    person_list = get_min_img(pd_df, min_img)
    total_filtered = pd_df[pd_df['name'].isin(person_list)]
    sample_list = []
    for img in person_list:
        sample_list.append(total_filtered[total_filtered.name == img].sample(sample_size))
    return pd.concat(sample_list)

def get_mounted_data(min_img, sample_size):
    image_paths = get_data_transform()
    return mount_data(image_paths, min_img, sample_size)

def directory_mover(data, dir_name):
    co = 0
    for image in data.image_path:
        # create top directory
        if not os.path.exists(os.path.join(os.getcwd(), 'new/working/', dir_name)):
            os.makedirs(os.path.join(os.path.join(os.getcwd(), 'new/working/'), dir_name))

        data_type = data[data['image_path'] == image]['name']
        data_type = str(list(data_type)[0])
        if not os.path.exists(os.path.join(os.getcwd(), 'new/working/', dir_name, data_type)):
            os.makedirs(os.path.join(os.path.join(os.getcwd(), 'new/working/', dir_name, data_type)))
        path_from = os.path.join('lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/', image)
        path_to = os.path.join('new/working/', dir_name, data_type)
        shutil.copy(path_from, path_to)
        co += 1

    print('Moved {} images to {} folder.'.format(co, dir_name))

def dataTrainAugmentation():
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True
    )

def dataHoldOutAugmentation():
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.3
    )

def dataTestAugmentation():
    return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# tirar imagem media de todas imagens antes de treinar
def transform_image_dataframe_to_matrix(dataframe, new_width, new_height, from_current_to_images_path):
    dataframe_rows = dataframe.shape[0]
    list_images = np.zeros((dataframe_rows, new_width, new_height, 3), dtype=np.float64)
    count = 0
    for image in dataframe.image_path:
        read = Image.open(os.path.join(os.getcwd(), from_current_to_images_path, image)).resize((new_width, new_height))
        data = np.asarray(read) / 255.0
        list_images[count] = data
        count += 1
    #one hot encoding
    labels = pd.get_dummies(dataframe["name"]).to_numpy()

    #sparse encoding
    #le = preprocessing.LabelEncoder()
        #group labels
    #le.fit(dataframe["name"])
        # transform label
    #labels = le.transform(dataframe["name"])

    return list_images, labels

def _parse_function(proto):
    keys_to_features = {"label": tf.FixedLenFeature([], tf.int64),                 
                        'image_raw': tf.FixedLenFeature([], tf.string)}
                        
    parsed_features = tf.parse_single_example(proto, keys_to_features)
    image = tf.decode_raw(
        parsed_features['image_raw'], tf.uint8)

    image = tf.reshape(image, (300, 300, 3))
    return image, parsed_features["label"]

def create_dataset_tfRecord(dataframe, from_current_to_images_path):
    print(dataframe.head(1)['image_path'])
    #dataset = tf.data.TFRecordDataset(os.path.join(os.getcwd(), from_current_to_images_path))
    #print(dataset)

    '''
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    image, label = iterator.get_next()
    image = tf.reshape(image, [batch_size, 300, 300, 3])
    label = tf.one_hot(label, num_classes)
    '''
    return
