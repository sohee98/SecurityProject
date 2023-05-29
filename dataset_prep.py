import cv2
import os
import glob
import shutil

save_folder = '/SSD2/Security_Project/Dataset_All/'
os.makedirs(save_folder, exist_ok=True)

img_shape_list = []

def move_image_folder_1():      # 1~2530
    save_name = 1       # 저장할 이미지 이름
    image_root_1 = '/SSD2/Security_Project/signature'
    img_path_list = glob.glob(image_root_1+'/*.png')
    for img_path in img_path_list:
        shutil.copyfile(img_path, save_folder+f'{save_name}.png')
        save_name += 1

def move_image_folder_2():      # 300개
    save_name = 2531       # 저장할 이미지 이름
    image_root_2 = '/SSD2/Security_Project/sample_signature/sample_Signature'       # /forged   or   /genuine
    img_path_list = glob.glob(image_root_2+'/*/*')
    for img_path in img_path_list:
        shutil.copyfile(img_path, save_folder+f'{save_name}.png')
        save_name += 1

def move_image_folder_3():  
    save_name = 2831       # 저장할 이미지 이름
    image_root_3 = '/SSD2/Security_Project/dataset_signature_final/Dataset'         # /dataset1 ~ 4  +  /forge or real1
    img_path_list = glob.glob(image_root_3+'/*/*/*')
    for img_path in img_path_list:
        shutil.copyfile(img_path, save_folder+f'{save_name}.png')
        save_name += 1

# move_image_folder_1()       # 1~2530
# move_image_folder_2()       # 2531~2831 (300개)
# move_image_folder_3()       # 2831~3550 (720개)


##########################################################################################

from PIL import  Image, ImageDraw, ImageFont
import glob

file_path = '/SSD2/Security_Project/Dataset_All/'
list_images = glob.glob(file_path + '*.png')

min_width, min_height = 10000, 10000
max_width, max_height = 0, 0

for image in list_images:
    img = Image.open(image)
    width, height = img.size

    if width < min_width:
        min_image_size_w = (width, height)
        min_width = width
    if height < min_height:
        min_image_size_h = (width, height)
        min_height = height

    if width > max_width:
        max_image_size_w = (width, height)
        max_width = width
    if height > max_height:
        max_image_size_h = (width, height)
        max_height = height

    if width == 72 and height == 36:    # '/SSD2/Security_Project/Dataset_All/3060.png'
        print(image)
    if width == 3786  and height == 2222:    # '/SSD2/Security_Project/Dataset_All/3130.png'
        print(image)
        


print('min_image_size_w, min_image_size_h, max_image_size_w, max_image_size_h')
print(min_image_size_w, min_image_size_h, max_image_size_w, max_image_size_h)
'''(72, 36) (72, 36) (3786, 2222) (3342, 2691)'''

# def get_mean_and_std(dataloader):
#     channels_sum, channels_squared_sum, num_batches = 0, 0, 0
#     for data, _ in dataloader:
#         # Mean over batch, height and width, but not over the channels
#         channels_sum += torch.mean(data, dim=[0,2,3])
#         channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
#         num_batches += 1
    
#     mean = channels_sum / num_batches

#     # std = sqrt(E[X^2] - (E[X])^2)
#     std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

#     return mean, std

# meanT, stdT = get_mean_and_std(trainloader)
# meanV, stdV = get_mean_and_std(validloader)
# print(f'Trainset - Mean : {meanT}, std : {stdT}')
# print(f'Validset - Mean : {meanV}, std : {stdV}')
