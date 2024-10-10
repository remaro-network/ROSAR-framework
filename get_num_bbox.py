import csv
import os
import random
import cv2
import argparse
import torch
import torchvision
import numpy
import onnx
import onnxruntime
import shutil
import ast
from pathlib import Path

from util import preprocess, decode_outputs, postprocess, show_pytorch_img, vis_box, visual
from model_util import check_model, predict

from config import *

def run(img):
    im_fold_path = 'data'

    # Perform prediction
    model_path = f'onnx/{MODEL_NAME}'
    print('Model: ' + model_path)
    check_model(model_path)
    print('Class names:', C_NAMES)

    preproc_img, img_info = preprocess(img)
    # print('image info:', img_info)
    # print(preproc_img)
    # print(preproc_img.shape)
    # cv2.imshow('preprocessed image', preproc_img[0].detach().cpu().numpy().transpose(1, 2, 0).astype(numpy.uint8))
    # cv2.imshow('original image', cv2.imread(img))
    # cv2.waitKey()

    raw_pred = predict(preproc_img, model_path)     # until this point the prediction seems to be good
    raw_pred = raw_pred[None, :, :]
    raw_pred_backup = raw_pred.clone()
    # print(raw_pred)

    decoded_pred = decode_outputs(raw_pred, torch.FloatTensor)          # until this point the decoded output tensor seems to be equivalent to the one that I get in the YOLOX code
    # print(decoded_pred)

    final_pred, indices_map = postprocess(decoded_pred, len(C_NAMES), CONF_THRESHOLD, NMS_THRESHOLD, CLASS_AGNOSTIC)
    final_pred, indices_map = final_pred[0], indices_map[0]
    # print(final_pred)
    # print(indices_map)

    count = 0
    for idx, prediction in enumerate(final_pred):

        # print(idx, 'prediction:', prediction)
        # print('raw prediction:', raw_pred[0][indices_map[idx]])
        # print('decoded prediction:', decoded_pred[0][indices_map[idx]])
        if prediction.numpy()[-1] >= 0.1:
            continue

        # for delta in sorted([0.01, 0.05, 0.10, 0.25], reverse=True):
        _allmost_preproc_img = preproc_img[0].detach().cpu().numpy().transpose(1, 2, 0).astype(numpy.uint8).copy()
        pred_img = vis_box(_allmost_preproc_img, prediction[0:4], prediction[4] * prediction[5], int(prediction[-1]))
        # cv2.imshow(f'pred_img', pred_img)
        # cv2.waitKey()

        count = count + 1

    # img = visual(final_pred, img_info, CONF_THRESHOLD, img_info['file_name'])
    # cv2.imshow('predictions', img)
    # cv2.waitKey()
    print(count)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', type=str, help='input image name')
    opt = parser.parse_args()
    print('CMD Arguments:', opt)
    return opt


def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)