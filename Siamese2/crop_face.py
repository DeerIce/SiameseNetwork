import os
import cv2
import shutil
from mtcnn import MTCNN
from PIL import Image


def crop_face_img(path, detector):
    image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None:
        raise IOError("Failed to load image")
    else:
        faces = detector.detect_faces(image)
        for face in faces:
            x, y, w, h = face['box']
            face_image = image[y:y+h, x:x+w]

        confidence = face['confidence']
        os.remove(path)
        if (confidence > 0.9):
            cv2.imwrite(path, face_image)


def prepare_img_pair(dataset_path):
    detector = MTCNN()  # 加载MTCNN模型

    subfolders = sorted([folder for folder in os.listdir(dataset_path)])

    for folder in subfolders:
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        image_names = sorted(os.listdir(folder_path))  # 返回文件夹包含的文件
        if len(image_names) != 2:  # 每个子文件夹应包含两张照片
            continue

        image_paths = [os.path.join(folder_path, image_name)
                       for image_name in image_names]

        try:
            crop_face_img(image_paths[0], detector)
            crop_face_img(image_paths[1], detector)

        except Exception as e:
            print(f"Failed to process image: {image_paths}")
            try:
                os.remove(image_paths[0])
                os.remove(image_paths[1])
            except Exception as ee:
                print(ee)
            print(e)


def remove_empty_subdirectories(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            if not os.listdir(dir_path):  # 检查子文件夹是否为空
                print(f"Removing empty directory: {dir_path}")
                shutil.rmtree(dir_path)  # 递归删除空文件夹


def move_single_image_folders(source_folder, destination_folder):
    for root, dirs, files in os.walk(source_folder):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            images = [file for file in os.listdir(
                dir_path) if file.endswith('.jpg') or file.endswith('.png')]
            if len(images) == 1:
                image_name = images[0]
                image_path = os.path.join(dir_path, image_name)
                destination_path = os.path.join(destination_folder, image_name)
                shutil.move(image_path, destination_path)
                print(f"Moved image: {image_name}")
                os.rmdir(dir_path)
                print(f"Removed directory: {dir_path}")


def crop_face_from_folders_by_cascade(folder_path):
    # 加载人脸检测器（使用OpenCV内置的级联分类器）
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 遍历文件夹里的所有子文件夹
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            # 获取子文件夹的路径
            subfolder_path = os.path.join(root, dir_name)

            # 遍历子文件夹中的所有图片文件
            for file_name in os.listdir(subfolder_path):
                # 获取图片文件的路径
                image_path = os.path.join(subfolder_path, file_name)

                # 读取图片
                img = cv2.imread(image_path)

                # 将图像转换为灰度图像
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # 进行人脸检测
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # 如果检测到人脸，则保存人脸图片并删除原图片
                if len(faces) > 0:
                    # 只处理第一个检测到的人脸
                    x, y, w, h = faces[0]

                    # 提取人脸区域
                    face_img = img[y:y+h, x:x+w]

                    # 保存人脸图片的路径
                    face_save_path = os.path.join(
                        subfolder_path, 'face_' + file_name)

                    # 删除原图片
                    os.remove(image_path)

                    # 保存人脸图片
                    cv2.imwrite(face_save_path, face_img)


def crop_face_from_folders_by_MTCNN(folder_path):
    # 加载MTCNN模型
    detector = MTCNN()

    # 遍历文件夹里的所有子文件夹
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            # 获取子文件夹的路径
            subfolder_path = os.path.join(root, dir_name)

            image_names = sorted(os.listdir(subfolder_path))  # 返回文件夹包含的文件
            if len(image_names) != 2:  # 每个子文件夹应包含两张照片
                shutil.rmtree(subfolder_path)
                continue

            # 遍历子文件夹中的所有图片文件
            for file_name in os.listdir(subfolder_path):
                # 获取图片文件的路径
                image_path = os.path.join(subfolder_path, file_name)

                # 读取图片
                img = cv2.imread(image_path)

                try:
                    # 进行人脸检测
                    faces = detector.detect_faces(img)

                    for face in faces:
                        x, y, w, h = face['box']
                        face_image = img[y:y+h, x:x+w]

                        confidence = face['confidence']
                        os.remove(image_path)
                        # 保存人脸图片的路径
                        face_save_path = os.path.join(
                            subfolder_path, file_name)
                        if (confidence > 0.9):
                            cv2.imwrite(face_save_path, face_image)

                except Exception as e:
                    print(e)


if __name__ == '__main__':
    # prepare_pair_face('train3')

    # remove_empty_subdirectories('train2')

    # move_single_image_folders('train2', 'train3')

    crop_face_from_folders_by_MTCNN('train2')

    pass
