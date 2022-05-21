'''
preprocess celebA
'''

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import os
import os.path as path
import csv
from tqdm import tqdm
import cv2


class CelebA_Preprocess():
    def __init__(self):
        self.celeba_PATH = '/root/datasets/CelebA/img_celeba/img_celeba'
        self.celeba_bbox_PATH = '/root/datasets/CelebA/celeba_crop_bbox.txt'


        #make dirs
        self.file_dir = path.dirname(__file__)
        os.makedirs(path.join(self.file_dir,'train'), exist_ok = True)
        os.makedirs(path.join(self.file_dir,'test'), exist_ok = True)
        os.makedirs(path.join(self.file_dir,'val'), exist_ok = True)

        self.bbox_inf = []
        self.file_list = []
        self.type_list = []
        n_train = 0
        n_test = 0
        n_val = 0
        with open(self.celeba_bbox_PATH) as tsv:
            for line in tqdm(csv.reader(tsv, delimiter = ' ')):
                self.file_list.append(line[0])
                self.bbox_inf.append(line[2:])
                self.type_list.append(int(line[1]))

                if int(line[1]) == 0:
                    n_train += 1
                elif int(line[1]) == 1:
                    n_val += 1
                else:
                    n_test +=1

                # check whether all bbox is square or not.

                if line[4] != line[5]:
                    print(line)


        print("# train: ",n_train, ", # test: ", n_test, ", # val:", n_val)
        print("# of files : ", len(self.file_list))

    def crop_all(self):
        '''
        it crops all image following bounding box information.
        Also, it divides the datasets into train, val, test.
        (05/09)
        '''
        for ind, img_name in tqdm(enumerate(self.file_list)):
            img = cv2.imread(path.join(self.celeba_PATH, img_name))
            x = int(self.bbox_inf[ind][0])
            y = int(self.bbox_inf[ind][1])
            w = int(self.bbox_inf[ind][2])
            h = int(self.bbox_inf[ind][3])

            '''pad imgs when bbox is out of img'''
            x_front = 0   # offset for the case when we padded in front of the img.
            y_front = 0
            x_back = 0
            y_back = 0
            
            if x<0:
                x_front = -x
            if y<0:
                y_front = -y
            if x+w>= img.shape[0]:
                x_back = x+w-img.shape[0]+1
            if y+h>=img.shape[1]:
                y_back = y+w-img.shape[1]+1

            if x_front+y_front+x_back+y_back > 0:
                img = cv2.copyMakeBorder(img, x_front, x_back, y_front, y_back, cv2.BORDER_REPLICATE)
                x = x + x_front
                y = y + y_front

            crop_img = img[y:(y+h),x:(x+w),:]
            re_crop_img = cv2.resize(crop_img, dsize=(128,128), interpolation = cv2.INTER_LINEAR)
            
            save_path = None
            if self.type_list[ind] == 0:
                save_path = path.join(self.file_dir,'train')
            elif self.type_list[ind] == 1:
                save_path = path.join(self.file_dir,'val')
            else:
                save_path = path.join(self.file_dir,'test')
            
            save_name = path.join(save_path, img_name)

            
            cv2.imwrite(save_name, re_crop_img)

        
if __name__ == '__main__':
    preprocessor = CelebA_Preprocess()
    preprocessor.crop_all()

    

