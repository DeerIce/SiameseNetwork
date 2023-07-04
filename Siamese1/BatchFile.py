import os
import cv2
import shutil
from mtcnn import MTCNN

def combine_different_person(folder_path):
    subdirectories = [os.path.join(folder_path, name) for name in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, name))]

    output_directory = os.path.join(folder_path, "Combined")
    os.makedirs(output_directory, exist_ok=True)

    index = 1

    for i in range(len(subdirectories)):
        for j in range(i + 1, len(subdirectories)):
            folder1 = subdirectories[i]
            folder2 = subdirectories[j]

            files1 = [os.path.join(folder1, name) for name in os.listdir(folder1)
                      if os.path.isfile(os.path.join(folder1, name))
                      and not name.lower().endswith(".db")]  # 排除以 ".db" 结尾的文件
            files2 = [os.path.join(folder2, name) for name in os.listdir(folder2)
                      if os.path.isfile(os.path.join(folder2, name))
                      and not name.lower().endswith(".db")]
            for file1 in files1:
                for file2 in files2:
                    combined_folder = os.path.join(
                        output_directory, f"0_{index}")
                    os.makedirs(combined_folder, exist_ok=True)

                    shutil.copy2(file1, combined_folder)
                    shutil.copy2(file2, combined_folder)

                    index += 1

def combine_same_person(folder_path):
    # 获取文件夹下的所有文件夹路径
    subdirectories = [os.path.join(folder_path, name) for name in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, name))]

    # 创建一个新的文件夹来存储组合后的图片
    output_directory = os.path.join(folder_path, "Combined")
    os.makedirs(output_directory, exist_ok=True)

    index = 1

    # 遍历每个文件夹
    for subdir in subdirectories:
        # 获取当前文件夹下的所有图片文件路径
        image_files = [os.path.join(subdir, name) for name in os.listdir(subdir)
                       if os.path.isfile(os.path.join(subdir, name))
                       and not name.lower().endswith(".db")]

        # 将每两张图片进行组合
        for i in range(len(image_files)):
            for j in range(i + 1, len(image_files)):
                image1 = image_files[i]
                image2 = image_files[j]

                # 创建新的文件夹来存储当前组合的图片
                combined_folder = os.path.join(output_directory, f"1_{index}")
                os.makedirs(combined_folder, exist_ok=True)

                # 复制图片到新的文件夹
                shutil.copy2(image1, combined_folder)
                shutil.copy2(image2, combined_folder)

                index += 1

def crop_face_from_folders_by_MTCNN(folder_path):
    # 加载MTCNN模型
    detector = MTCNN()

    # 遍历文件夹里的所有子文件夹
    for root, dirs, files in os.walk(folder_path):
        for dir_name in dirs:
            subfolder_path = os.path.join(root, dir_name)

            image_names = sorted(os.listdir(subfolder_path))
            if len(image_names) != 2:
                shutil.rmtree(subfolder_path)
                continue

            for file_name in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, file_name)
                img = cv2.imread(image_path)
                try:
                    faces = detector.detect_faces(img)

                    for face in faces:
                        x, y, w, h = face['box']
                        face_image = img[y:y+h, x:x+w]
                        confidence = face['confidence']
                        os.remove(image_path)
                        face_save_path = os.path.join(
                            subfolder_path, file_name)
                        if (confidence > 0.9):
                            cv2.imwrite(face_save_path, face_image)

                except Exception as e:
                    print(e)

if __name__ == '__main__':
    # combine_different_person('datapath')
    # combine_same_person('datapath')
    # crop_face_from_folders_by_MTCNN('datapth')
    pass