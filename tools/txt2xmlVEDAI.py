import os, sys
import glob

import cv2
from PIL import Image
# dict = {'0': "People",  # 字典对类型进行转换
#         '1': "Car",
#         '2': "Bus",
#         '3': "Lamp",
#         '4': "Motorcycle",
#         '5': "Truck",
#         }
dict = {'0': "car",  # 字典对类型进行转换
        '1': "pickup",
        '2': "camping",
        '3': "other",
        '4': "other",
        '5': "tractor",
        '6':"boat",
        '7':"van",
        }
# ['car', 'pickup', 'camping','truck', 'other', 'tractor', 'boat', 'van']
# classes = ['People', 'Car', 'Bus', 'Lamp', 'Motorcycle', 'Truck']
# VEDAI 图像存储位置
src_img_dir = "/home/zjq/dataset/VEDAI/vi"
# VEDAI 图像的 ground truth 的 txt 文件存放位置
src_txt_dir = "/home/zjq/dataset/VEDAI/labels"
src_xml_dir = "./datasets/VEDAI/Annotations"
os.makedirs(src_xml_dir, exist_ok=True)

img_names = os.listdir(src_img_dir)

for img in img_names:
    print(img)
    if img.split('.')[-1] != 'png':
        continue
    print(img)
    im = cv2.imread(src_img_dir + '/' + img)
    Pheight, Pwidth, Pdepth = im.shape

    gt = open(src_txt_dir + '/' + img.replace('.png', '.txt')).read().splitlines()
    name = img.split('.')[0]
    xml_file = open((src_xml_dir + '/' + name + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>' + str(img) + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(Pwidth) + '</width>\n')
    xml_file.write('        <height>' + str(Pheight) + '</height>\n')
    xml_file.write('        <depth>' + str(Pdepth) + '</depth>\n')
    xml_file.write('    </size>\n')

    # write the region of image on xml file
    for img_each_label in gt:
        spt = img_each_label.split(' ')  # 这里如果txt里面是以逗号‘，’隔开的，那么就改为spt = img_each_label.split(',')。
        print(spt)
        xml_file.write('    <object>\n')

        xml_file.write('        <name>' + dict[str(spt[0])] + '</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(int(((float(spt[1]))*Pwidth+1)-(float(spt[3]))*0.5*Pwidth)) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(int(((float(spt[2]))*Pheight+1)-(float(spt[4]))*0.5*Pheight)) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(int(((float(spt[1]))*Pwidth+1)+(float(spt[3]))*0.5*Pwidth)) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(int(((float(spt[2]))*Pheight+1)+(float(spt[4]))*0.5*Pheight)) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')

    xml_file.write('</annotation>')