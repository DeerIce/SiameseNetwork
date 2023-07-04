import os
import sys
import cv2
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from keras import layers, models

def load_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return image

def prepare_dataset(dataset_path, train_ratio=0.8):
    pairs, labels = [], []

    subfolders = sorted([folder for folder in os.listdir(dataset_path)])

    total_folders = len(subfolders)

    for i, folder in enumerate(subfolders):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        image_names = sorted(os.listdir(folder_path))
        if len(image_names) != 2:
            continue

        image_paths = [os.path.join(folder_path, image_name)
                       for image_name in image_names]

        try:
            image1 = load_image(image_paths[0])
            image2 = load_image(image_paths[1])

            pairs.append((image1, image2))
            labels.append(int(folder[0]))  # 同一个人,标记为1,否则0
        except Exception as e:
            print(f"Failed to process image: {image_paths}")
            print(e)

        print_progress_bar(i + 1, total_folders, prefix='Loading:')

    pairs = np.array(pairs)
    labels = np.array(labels)

    # 随机打乱数据集
    indices = np.random.permutation(len(pairs))
    pairs = pairs[indices]
    labels = labels[indices]

    # 拆分训练集和验证集
    split_idx = int(len(pairs) * train_ratio)
    train_pairs, val_pairs = pairs[:split_idx], pairs[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    return (train_pairs, train_labels), (val_pairs, val_labels)

def print_progress_bar(iteration, total, prefix='', suffix='', length=70, fill='█'):
    percent = "{0:.1f}".format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()

def create_siamese_network(input_shape):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights='imagenet')

    base_model.trainable = False

    inputs1 = tf.keras.Input(shape=input_shape)
    inputs2 = tf.keras.Input(shape=input_shape)

    features1 = base_model(inputs1)
    features2 = base_model(inputs2)

    flattened1 = layers.Flatten()(features1)
    flattened2 = layers.Flatten()(features2)

    # 将两个特征向量进行融合，以便进行相似度比较
    concatenated = layers.Concatenate()([flattened1, flattened2])
    dense1 = layers.Dense(512, activation='relu')(concatenated)
    dense2 = layers.Dense(512, activation='relu')(dense1)
    output = layers.Dense(1, activation='sigmoid')(dense2)

    model = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=output)
    return model

def train_model(batch_size=128, epochs=10, train_data_path=''):
    # 准备数据集
    (train_pairs, train_labels), (val_pairs,
                                  val_labels) = prepare_dataset(train_data_path)

    input_shape = train_pairs.shape[2:]

    # 创建Siamese网络模型
    model = create_siamese_network(input_shape)

    # 编译模型
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, mode='max')

    model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_labels,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
              callbacks=[early_stopping])

    model.save('model.h5')

def cal_simi_score(dataset_path):
    model = models.load_model('model1.h5')
    correct_num=0

    subfolders = sorted([folder for folder in os.listdir(dataset_path)])
    total_folders = len(subfolders)

    for i, folder in enumerate(subfolders):
        folder_path = os.path.join(dataset_path, folder)

        if not os.path.isdir(folder_path):
            continue

        image_names = sorted(os.listdir(folder_path))

        if len(image_names) != 2:
            continue

        image_paths = [os.path.join(folder_path, image_name)
                       for image_name in image_names]

        try:
            image1 = np.array(load_image(image_paths[0]))
            image2 = np.array(load_image(image_paths[1]))
            image1 = np.expand_dims(image1, axis=0)
            image2 = np.expand_dims(image2, axis=0)

            # 执行相似度检测
            similarity_score=model.predict([image1, image2])
            if (int(folder[0])==0 and similarity_score<0.6) or (int(folder[0])==1 and similarity_score>=0.6):
                correct_num+=1

        except Exception as e:
            print(f"Failed to process image: {image_paths}")
            print(e)

    print('\naccuracy=',correct_num/total_folders)


if __name__ == '__main__':
    # train_model(batch_size=128, epochs=20, train_data_path='train')
    cal_simi_score('test')
    pass
