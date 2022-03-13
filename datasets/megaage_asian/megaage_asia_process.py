import numpy as np
import cv2
import scipy.io
import argparse
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import sys
from PIL import Image
import dlib
from moviepy.editor import *

detector = dlib.get_frontal_face_detector()
def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str,
                        help="path to output database mat file")
    parser.add_argument("--img_size", type=int, default=120,
                        help="output image size")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # output_path = '/disks/disk2/data/megaage_asian_pro/test'
    output_path = '/disks/disk2/data/megaage_asian_pro/train'
    img_size = args.img_size

    # mypath = '/disks/disk2/data/megaage_asian/test'
    mypath = '/disks/disk2/data/megaage_asian/train'
    isPlot = False

    # age_file = np.loadtxt('/disks/disk2/data/megaage_asian/list/test_age.txt')
    age_file = np.loadtxt('/disks/disk2/data/megaage_asian/list/train_age.txt')
    # dis_file = np.loadtxt('/disks/disk2/data/megaage_asian/list/test_dis.txt', dtype='float32')
    dis_file = np.loadtxt('/disks/disk2/data/megaage_asian/list/train_dis.txt', dtype='float32')
    # img_name_file = np.genfromtxt('/disks/disk2/data/megaage_asian/list/test_name.txt', dtype='str')
    img_name_file = np.genfromtxt('/disks/disk2/data/megaage_asian/list/train_name.txt',dtype='str')
    img_files=[]
    dis_files=[]
    age_files=[]
    for i in tqdm(range(len(img_name_file))):
        img = cv2.imread(mypath + '/' + img_name_file[i])
        detected = detector(img, 1)
        age = int(float(age_file[i]))
        if len(detected) != 1:  # skip if there are 0 or more than 1 face
            # print("skip if there are 0 or more than 1 face:",len(detected))
            img = img[20:-20, :, :]
            try:

                tmp = np.array(Image.fromarray(np.uint8(img)).resize((120, 120), Image.ANTIALIAS))
                cv2.imwrite(os.path.join(output_path, img_name_file[i]), tmp)
                img_files.append(img_name_file[i])
                dis_files.append(dis_file[i])
                age_files.append(age)
            except ValueError:
                print(img_name_file[i])


        else:
            for idx, face in enumerate(detected):
                width = face.right() - face.left()
                height = face.bottom() - face.top()
                tol = 15
                up_down = 5
                diff = height - width
                # print(width,height)
                if (diff > 0):
                    if not diff % 2:  # symmetric
                        tmp = img[(face.top() - tol - up_down):(face.bottom() + tol - up_down),
                              (face.left() - tol - int(diff / 2)):(face.right() + tol + int(diff / 2)),
                              :]
                    else:
                        tmp = img[(face.top() - tol - up_down):(face.bottom() + tol - up_down),
                              (face.left() - tol - int((diff - 1) / 2)):(face.right() + tol + int((diff + 1) / 2)),
                              :]
                if (diff <= 0):
                    if not diff % 2:  # symmetric
                        tmp = img[(face.top() - tol - int(diff / 2) - up_down):(
                                    face.bottom() + tol + int(diff / 2) - up_down),
                              (face.left() - tol):(face.right() + tol),
                              :]
                    else:
                        tmp = img[(face.top() - tol - int((diff - 1) / 2) - up_down):(
                                    face.bottom() + tol + int((diff + 1) / 2) - up_down),
                              (face.left() - tol):(face.right() + tol),
                              :]
                try:
                    tmp = np.array(Image.fromarray(np.uint8(tmp)).resize((120, 120), Image.ANTIALIAS))
                    cv2.imwrite(os.path.join(output_path, img_name_file[i]), tmp)
                    img_files.append(img_name_file[i])
                    dis_files.append(dis_file[i])
                    age_files.append(age)
                except ValueError:
                    print(img_name_file[i])

    print(len(img_files),len(age_files),len(dis_files))
    np.savetxt('train_name.txt', np.array(img_files),fmt="%s",delimiter=' ')
    np.savetxt('train_age.txt', np.array(age_files),fmt="%d",delimiter=' ')
    np.savetxt('train_dis.txt', np.array(dis_files),fmt="%g",  delimiter=' ')

            # try:
            #     tmp = np.array(Image.fromarray(np.uint8(tmp)).resize((120, 120), Image.ANTIALIAS))
            #     cv2.imwrite(os.path.join(output_path, img_name_file[i]), tmp)
            #
            # except ValueError:
            #     print(f'Failed wrote')
            #     pass






if __name__ == '__main__':
    main()