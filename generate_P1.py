import csv
import os
import cv2
import argparse
import torch
import numpy
import shutil
from pathlib import Path

from model_util import check_model, get_model_interface, predict
from util import preprocess, decode_outputs, postprocess, vis_box

from config import *

def process_single_image(img_path, bbox_ind, model_path, delta):
    preproc_img, img_info = preprocess(img_path)
    # print('image info:', img_info)
    # print(preproc_img)
    # print(preproc_img.shape)
    # cv2.imshow('preprocessed image', preproc_img[0].detach().cpu().numpy().transpose(1, 2, 0).astype(numpy.uint8))
    # cv2.imshow('original image', cv2.imread(img))
    # cv2.waitKey()

    raw_pred = predict(preproc_img, model_path) 
    raw_pred = raw_pred[None, :, :]
    raw_pred_backup = raw_pred.clone()
    # print(raw_pred)

    decoded_pred = decode_outputs(raw_pred, torch.FloatTensor)       
    # print(decoded_pred)

    final_pred, indices_map = postprocess(decoded_pred, len(C_NAMES), CONF_THRESHOLD, NMS_THRESHOLD, CLASS_AGNOSTIC)
    final_pred, indices_map = final_pred[0], indices_map[0]
    # print(final_pred)
    # print(indices_map)

    idx = bbox_ind
    # print('final_pred before:', final_pred)
    prediction_filtered = [pred for pred in final_pred if pred[-1] < 0.1]
    # print('final_pred after:', prediction_filtered)
    prediction = prediction_filtered[idx]

    # print(idx, 'prediction:', prediction)
    # print('raw prediction:', raw_pred[0][indices_map[idx]])
    # print('decoded prediction:', decoded_pred[0][indices_map[idx]])

    # for delta in sorted([0.01, 0.05, 0.10, 0.25], reverse=True):
    _allmost_preproc_img = preproc_img[0].detach().cpu().numpy().transpose(1, 2, 0).astype(numpy.uint8).copy()
    img_minus, img_plus = add_delta_noise_to_bbox(_allmost_preproc_img, prediction[0:4], delta)

    # pred_img = vis_box(_allmost_preproc_img, prediction[0:4], prediction[4] * prediction[5], int(prediction[-1]))
    # cv2.imshow(f'pred_img_{delta}', pred_img)
    # cv2.imshow(f'img_minus_{delta}', img_minus)
    # cv2.imshow(f'img_plus_{delta}', img_plus)
    # cv2.waitKey()

    prop_fn = f'{Path(img_info["file_name"]).stem}_perturbed_bbox_{idx}_delta_{delta}.vnnlib'
    prop_path = str(Path('./vnnlib').joinpath(Path(prop_fn)))

    my_serialize_property(prop_path, model_path, img_minus, img_plus, raw_pred, indices_map[idx], 0.1, None)

    with open('instances.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'onnx/{MODEL_NAME}', prop_path, '125'])

def my_serialize_property(prop_path, model_path, im_minus, im_plus, raw_pred, raw_bbox_ind, eps, img_size):
    h, w, n_bboxes, n_data = get_model_interface(model_path)
    n_channels = 3
    # print(h, w, n_channels, n_bboxes, n_data,)

    # Input variables declaration
    with open(prop_path, 'w') as f:
        # Input variables declaration
        n_inp = 0
        f.write(';Input variables:\n')
        for y in range(h):
            for x in range(w):
                for k in range(n_channels):
                    f.write('(declare-const X_' + str(n_inp) + ' Real)' + '\n')
                    n_inp += 1

        # Output variables declaration
        n_out = 0
        f.write('\n;Output variables:\n')
        for b in range(n_bboxes):
            for d in range(n_data):
                f.write('(declare-const Y_' + str(n_out) + ' Real)' + '\n')
                n_out += 1

        # Input constraints definition
        upper = im_plus.transpose((2, 0, 1))
        lower = im_minus.transpose((2, 0, 1))
        n_inp = 0
        f.write('\n;Input constraints:\n')
        for k in range(n_channels):
            for y in range(h):
                for x in range(w):
                    ub = upper[k][y][x]
                    lb = lower[k][y][x]
                    f.write('(assert (<= X_' + str(n_inp) + ' ' + str(ub) + '))' + '\n')
                    f.write('(assert (>= X_' + str(n_inp) + ' ' + str(lb) + '))' + '\n')
                    n_inp += 1

        # Output constraints definition
        pred = raw_pred.numpy()[0]
        n_out = 0
        f.write('\n;Output constraints:\n')
        f.write('(assert (or \n')
        
        for b in range(n_bboxes):
            bbox = pred[b]

            wall_pred = bbox[5] > bbox[6]
            close_enough = (abs(bbox[0] - pred[raw_bbox_ind][0]) < 0.35 and
                            abs(bbox[1] - pred[raw_bbox_ind][1]) < 0.35 and
                            abs(bbox[2] - pred[raw_bbox_ind][2]) < 0.35 and
                            abs(bbox[3] - pred[raw_bbox_ind][3]) < 0.35)
            obj_det = bbox[4] > 0.5

            for d in range(n_data): # Constrain upper and lower bounds of each bounding box element
                data = bbox[d]
                ub = data
                lb = data
                if b == raw_bbox_ind:
                    if d == 4: # Object existence probability does not change  below 50 or above 100 percent  ...  more than eps
                        assert(data >= 0 and data <= 1)
                        # ub = min(data*(1+eps), 1.0)
                        # lb = max(data*(1-eps), 0.0)
                        # constr = '\t(and (>= Y_' + str(n_out) + ' ' + str(ub) + ')) (and (<= Y_' + str(n_out) + ' ' + str(lb) + '))' + '\n'
                        lb = 0.5
                        ub = 1.0
                        constr = '\t(and (>= Y_' + str(n_out) + ' ' + str(ub) + ')) (and (<= Y_' + str(n_out) + ' ' + str(lb) + '))' + '\n'
                        f.write(constr)
                    elif d in [5,6]:  # Class conditional probability allowed to fluctuate (all between 0 and 1)
                        assert(data >= 0 and data <= 1)
                        ub = 1.0
                        lb = 0.0
                        constr = '\t(and (>= Y_' + str(n_out) + ' ' + str(ub) + ')) (and (<= Y_' + str(n_out) + ' ' + str(lb) + '))' + '\n'
                        f.write(constr)

                n_out += 1

            if b == raw_bbox_ind: # Highest class conditional probability remains the highest despite of perturbation (negated property)
                max_class_prob_ind = numpy.argmax(bbox[5:7])
                n_out_max_class_prob = n_out - 2 + max_class_prob_ind
                n_out_class_probs = [n for n in range(n_out - 2, n_out) if n != n_out_max_class_prob]

                # f.write(f'HERE: Y_{n_out}')
                for n in n_out_class_probs:
                    constr = '\t(and (>= Y_' + str(n) + ' Y_' + str(n_out_max_class_prob) + '))' + '\n'
                    f.write(constr)

        f.write('))\n')

def run(img_path, bbox_ind, delta):
    prop_fold_path = 'vnnlib'
    instances_fname = 'instances.csv'
    if os.path.exists(prop_fold_path) and os.path.isdir(prop_fold_path):
        shutil.rmtree(prop_fold_path)
    if os.path.exists(instances_fname):
        os.remove(instances_fname)
    os.mkdir(prop_fold_path)

    # Perform prediction
    model_path = f'onnx/{MODEL_NAME}'
    print('Model: ' + model_path)
    check_model(model_path)
    print('Class names:', C_NAMES)

    process_single_image(img_path, bbox_ind, model_path, delta)

def add_delta_noise_to_bbox(im, bbox, d):
    im_plus = im.copy()
    im_minus = im.copy()
    # for y in range(int(bbox[1])+1, int(bbox[3])):
        # for x in range(int(bbox[0])+1, int(bbox[2])):
    for y in range(0, 416):
        for x in range(0, 416):
            color = im[y][x]
            color_plus = [min(int(color[0]*(1+d)),255), min(int(color[1]*(1+d)),255), min(int(color[2]*(1+d)),255)]
            color_minus = [max(int(color[0]*(1-d)),0), max(int(color[1]*(1-d)),0), max(int(color[2]*(1-d)),0)]
            im_plus[y][x] = color_plus
            im_minus[y][x] = color_minus
    return im_minus, im_plus

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, help='path of input image')
    parser.add_argument('bbox_ind', type=int, help='index of the bounding box')
    parser.add_argument('delta', type=float, help='perturbation upper bound')
    opt = parser.parse_args()
    print('CMD Arguments:', opt)
    return opt

def main(opt):
    run(**vars(opt))

def generate_dummy_prop_yolox():
    prop_path = './vnnlib/yolox_dummy_property.vnnlib'
    model_path = './onnx/yolox_nano.onnx'

    h, w, n_bboxes, n_data = get_model_interface(model_path)
    n_channels = 3
    with open(prop_path, 'a') as f:
        # Input variables declaration
        n_inp = 0
        f.write(';Input variables:\n')
        for y in range(h):
            for x in range(w):
                for k in range(n_channels):
                    f.write('(declare-const X_' + str(n_inp) + ' Real)' + '\n')
                    n_inp += 1
        # Output variables declaration
        n_out = 0
        f.write('\n;Output variables:\n')
        for b in range(n_bboxes):
            for d in range(n_data):
                f.write('(declare-const Y_' + str(n_out) + ' Real)' + '\n')
                n_out += 1
        # Input constraints definition
        n_inp = 0
        f.write('\n;Input constraints:\n')
        for y in range(h):
            for x in range(w):
                for k in range(n_channels):
                    ub = +0.1
                    lb = -0.1
                    f.write('(assert (<= X_' + str(n_inp) + ' ' + str(ub) + '))' + '\n')
                    f.write('(assert (>= X_' + str(n_inp) + ' ' + str(lb) + '))' + '\n')
                    n_inp += 1

def read_predict_draw(img_path):
    preproc_img, img_info = preprocess(img_path)
    # print(preproc_img)
    # print(preproc_img.shape)
    # cv2.imshow(img_path, preproc_img[0].detach().cpu().numpy().transpose(1, 2, 0).astype(numpy.uint8))
    # cv2.waitKey()

    raw_pred = predict(preproc_img, f'onnx/{MODEL_NAME}')     # until this point the prediction seems to be good
    raw_pred = raw_pred[None, :, :]
    # print(raw_pred)
    # print(raw_pred.shape)

    decoded_pred = decode_outputs(raw_pred, torch.FloatTensor)
    final_pred, indices_map = postprocess(decoded_pred, len(C_NAMES), CONF_THRESHOLD, NMS_THRESHOLD, CLASS_AGNOSTIC)
    final_pred, indices_map = final_pred[0], indices_map[0]
    # print(final_pred)
    # print(final_pred.shape)
    print(indices_map)

    pred_img = preproc_img[0].clone().detach().cpu().numpy().transpose(1, 2, 0).astype(numpy.uint8)

    if final_pred is None:
        return pred_img
    
    print('The originally predicted bounding box:', [value.item() for value in raw_pred[0][96]])

    for idx, prediction in enumerate(final_pred):
        # print("Original:", [value.item() for value in raw_pred[0][indices_map[idx]]])
        # print("New pred:", [value.item() for value in prediction])
        # prediction[4] = raw_pred[0][indices_map[idx]][4]

        pred_img = vis_box(pred_img.copy(), prediction[0:4], prediction[4], int(prediction[-1]))
        # pred_img = vis_box(pred_img.copy(), prediction[0:4], prediction[4] * prediction[5], int(prediction[-1]))

    return pred_img

def test_predictions():
    ori_img_path = 'data/Compressed_490.png'
    pred_img = read_predict_draw(ori_img_path)
    cv2.imshow(ori_img_path, pred_img)

    for adv_image in os.listdir('./adv_attacks/'):
        filename = os.fsdecode(adv_image)
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            filepath = os.path.join('./adv_attacks/', filename)
            print('Reading image from:', filepath)

            adv_pred_img = read_predict_draw(filepath)
            cv2.imshow(filepath[-20:], adv_pred_img)

    cv2.waitKey()
    input('asd')

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
