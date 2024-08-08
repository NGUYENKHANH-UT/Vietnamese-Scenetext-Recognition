import numpy as np
from PIL import Image
import cv2
import argparse
import os

# import handle code
from src.utils.four_points_transform import four_points_transform
from src.utils.encode_base64 import encode_base64

from src.craft.craft_predict import predict_craft, str2bool
from src.vietocr.vietocr_predict import predict_vietocr

from src.craft.load_model import load_model_craft
from src.vietocr.load_model import load_model_vietocr

from src.utils.group_text_box import group_text_box
from src.craft import file_utils

parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='src/craft/weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='data', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='src/craft/weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

args = parser.parse_args()

""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)
result_folder = 'result'

if __name__ == '__main__':
    # ======================  Load model  ================
    net, refine_net = load_model_craft(args=args)
    detector = load_model_vietocr(args=args)
    
    for k, image_path in enumerate(image_list):
        txt_content = ''
        main_image = cv2.imread(image_path)
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        boxes = predict_craft(net, refine_net, image_path=image_path, args=args)
        boxes = group_text_box(boxes)

        # rec phase
        for count, box in enumerate(boxes): 
            # cat anh ra va convert to PIL
            sub_img = four_points_transform(main_image, np.array(box, dtype='float32'))
            # write content
            box = [list(map(int, i)) for i in box]
            box = [list(map(lambda x: max(x, 0) ,i)) for i in box]
            txt_content += f"{box[0][0]},{box[0][1]},{box[1][0]},{box[1][1]},{box[2][0]},{box[2][1]},{box[3][0]},{box[3][1]}\t"
                    

            # ap dung can bang sang
            # sub_img = histogram_equalzed_rgb(sub_img)
            sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
            sub_img = Image.fromarray(sub_img)

            # predict with vietocr
            (pred, prob) = predict_vietocr(detector, sub_img)
        
            # write to txt
            txt_content += f"{pred}, {prob}\n"   
        # Tạo tên file dựa trên tên ảnh, thay đổi phần mở rộng thành .txt
        txt_filename = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
        txt_filepath = os.path.join(result_folder, txt_filename)

        # Lưu nội dung vào file .txt
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(txt_content)

    
            