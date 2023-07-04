import os
import shutil
import itertools

'''************************************************************************
把文件夹train里所有图片两两组合
combine_any_tow_images('train')'''


def combine_any_tow_images(source_folder):
    # 获取源文件夹中的所有图片文件
    image_files = [f for f in os.listdir(source_folder) if f.endswith(".jpg")]

    # 获取所有图片文件的两两组合
    combinations = list(itertools.combinations(image_files, 2))

    # 在源文件夹中创建子文件夹
    subfolders = ['1'+f"{i+2609}" for i in range(len(combinations))]
    for subfolder in subfolders:
        os.makedirs(os.path.join(source_folder, subfolder))

    # 复制图片组合到各个子文件夹中
    for i, combo in enumerate(combinations):
        for image_file in combo:
            source_path = os.path.join(source_folder, image_file)
            destination_folder = os.path.join(source_folder, subfolders[i])
            destination_path = os.path.join(destination_folder, image_file)
            shutil.copy(source_path, destination_path)

    print("批量移动图片完成！")


'''************************************************************************
把folder_path中的target_image_name分别与剩余照片两两结合并放到新创建的子文件夹里
combine_images_of_A_with_others('train3', '1.jpg')'''


def combine_images_of_A_with_others(folder_path, target_image_name):
    image_paths = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name != target_image_name:
            image_paths.append(file_path)

    for i, image_path in enumerate(image_paths):
        new_folder = os.path.join(folder_path, '0'+f"{i+153}")
        os.makedirs(new_folder, exist_ok=True)
        new_image_path = os.path.join(new_folder, target_image_name)
        shutil.copyfile(os.path.join(
            folder_path, target_image_name), new_image_path)
        shutil.copyfile(image_path, os.path.join(
            new_folder, os.path.basename(image_path)))
        print(
            f"Combined images: {target_image_name}, {os.path.basename(image_path)}")


'''************************************************************************
把folder_path中的文件夹两两组合,再把各组合里位于不同文件夹的文件两两结合
combine_two_different_person('train')'''


def combine_two_different_person(folder_path):
    subdirectories = [os.path.join(folder_path, name) for name in os.listdir(folder_path)
                      if os.path.isdir(os.path.join(folder_path, name))]

    output_directory = os.path.join(folder_path, "Combined")
    os.makedirs(output_directory, exist_ok=True)

    index = 4925

    for i in range(len(subdirectories)):
        for j in range(i + 1, len(subdirectories)):
            folder1 = subdirectories[i]
            folder2 = subdirectories[j]

            files1 = [os.path.join(folder1, name) for name in os.listdir(folder1)
                      if os.path.isfile(os.path.join(folder1, name))
                      and not name.lower().endswith(".db")]  # 排除以 ".db" 结尾的文件
            files2 = [os.path.join(folder2, name) for name in os.listdir(folder2)
                      if os.path.isfile(os.path.join(folder2, name))
                      and not name.lower().endswith(".db")]  # 排除以 ".db" 结尾的文件

            for file1 in files1:
                for file2 in files2:
                    combined_folder = os.path.join(
                        output_directory, f"0_{index}")
                    os.makedirs(combined_folder, exist_ok=True)

                    shutil.copy2(file1, combined_folder)
                    shutil.copy2(file2, combined_folder)

                    index += 1


'''************************************************************************
把folder_path中的文件夹里同一个人的图片两两组合
combine_same_person('train')'''


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
                       and not name.lower().endswith(".db")]  # 排除以 ".db" 结尾的文件

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


combine_same_person('train2')
# combine_two_different_person('train2') #上次组合用了pic里的000~038
