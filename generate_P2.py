import argparse
import csv
import cv2
import numpy
import os
import shutil
import torch

from pathlib import Path

from config import C_NAMES, CLASS_AGNOSTIC, CONF_THRESHOLD, IMG_SIZE, MODEL_NAME, NMS_THRESHOLD
from model_util import check_model, get_model_interface, predict
from util import decode_outputs, draw_all_bboxes_and_plot, postprocess, preprocess, vis_box

def create_black_lines(img, min_delta=0.15):
    im_minus = img.copy()
    im_plus = img.copy()


    with open('line_indices.cfg', 'r') as f:
        lines = f.readlines()
        line_ids = [int(line_id.strip()) for line_id in lines]

        for y in line_ids:
            for x in range(0, IMG_SIZE[1]):
                color = img[y][x]
                color_minus = [max(int(color[0]*min_delta),0), max(int(color[1]*min_delta),0), max(int(color[2]*min_delta),0)]
                color_plus = [min(int(color[0]),255), min(int(color[1]),255), min(int(color[2]),255)]
                im_minus[y][x] = color_minus
                im_plus[y][x] = color_plus

    return im_minus, im_plus

def serialize_prop2(prop_path, model_path, im_minus, im_plus, raw_pred, raw_bbox_ind):
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

def process_single_image(img, bbox_ind, model_path, min_delta):
    # reading and preprocessing input image
    preproc_img, img_info = preprocess(img)
    # print('image info:', img_info)
    # print(preproc_img)
    # print(preproc_img.shape)
    # cv2.imshow('preprocessed image', preproc_img[0].detach().cpu().numpy().transpose(1, 2, 0).astype(numpy.uint8))
    # cv2.imshow('original image', cv2.imread(img))
    # cv2.waitKey()

    # doing predictions with the model for the input image
    raw_pred = predict(preproc_img, model_path)     # until this point the prediction seems to be good
    raw_pred = raw_pred[None, :, :]
    # print('raw_pred:', raw_pred)
    # print('raw_pred.shape:', raw_pred.shape)

    # decoding (upscaling) bounding boxes
    decoded_pred = decode_outputs(raw_pred, torch.FloatTensor)          # until this point the decoded output tensor seems to be equivalent to the one that I get in the YOLOX code
    # print('decoded_pred:', decoded_pred)
    # print('decoded_pred.shape:', decoded_pred.shape)

    # filtering bboxes and computing conf/obj scores
    final_pred, indices_map = postprocess(decoded_pred, len(C_NAMES), CONF_THRESHOLD, NMS_THRESHOLD, CLASS_AGNOSTIC)
    final_pred, indices_map = final_pred[0], indices_map[0]
    print('final_pred:', final_pred)
    print('final_pred.shape:', final_pred.shape)
    print('indices_map:', list(zip(range(len(indices_map)), indices_map)))

    # draw_all_bboxes_and_plot(preproc_img, final_pred, 'all predictions')

    idx = bbox_ind
    print('final_pred before:', final_pred)
    prediction_filtered = [pred for pred in final_pred if pred[-1] < 0.1]
    print('final_pred after:', prediction_filtered)
    prediction = prediction_filtered[idx]
    print(idx, 'prediction:', prediction)

    # draw_all_bboxes_and_plot(preproc_img, [prediction], 'attacked prediction')

    # print('raw prediction:', raw_pred[0][indices_map[idx]])
    # print('decoded prediction:', decoded_pred[0][indices_map[idx]])

    # for delta in sorted([0.01, 0.05, 0.10, 0.25], reverse=True):
    _allmost_preproc_img = preproc_img[0].detach().cpu().numpy().transpose(1, 2, 0).astype(numpy.uint8).copy()
    img_minus, img_plus = create_black_lines(_allmost_preproc_img, min_delta)

    # pred_img = vis_box(_allmost_preproc_img, prediction[0:4], prediction[4] * prediction[5], int(prediction[-1]))
    # cv2.imshow(f'pred_img_{min_delta}', pred_img)
    # cv2.imshow(f'img_minus_{min_delta}', img_minus)
    # cv2.imshow(f'img_plus_{min_delta}', img_plus)
    # cv2.waitKey()

    #### attack
    # preproc_img, img_info = preprocess('./img_Compressed_23-06-2023_SurveyDataset-RealTimeSSS528_black_lines_0_delta_0.45.png')
    # raw_pred = predict(preproc_img, model_path)     # until this point the prediction seems to be good
    # raw_pred = raw_pred[None, :, :]
    # decoded_pred = decode_outputs(raw_pred, torch.FloatTensor)
    # final_pred, indices_map = postprocess(decoded_pred, len(C_NAMES), CONF_THRESHOLD, NMS_THRESHOLD, CLASS_AGNOSTIC)
    # final_pred, indices_map = final_pred[0], indices_map[0]
    # draw_all_bboxes_and_plot(preproc_img, final_pred, 'adversarial attack')
    #### attack

    prop_fn = f'img_{Path(img_info["file_name"]).stem}_black_lines_{idx}_min_delta_{min_delta}.vnnlib'
    prop_path = str(Path('./vnnlib').joinpath(Path(prop_fn)))

    serialize_prop2(prop_path, model_path, img_minus, img_plus, raw_pred, indices_map[idx])

    with open('instances.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'onnx/{MODEL_NAME}', prop_path, '125'])

    # cv2.waitKey()

def run(img, bbox_ind, min_delta):
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

    process_single_image(img, bbox_ind, model_path, min_delta)

    return

def main(opt):
    run(**vars(opt))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', type=str, help='input image name')
    parser.add_argument('bbox_ind', type=int, help='index of the bounding box')
    parser.add_argument('min_delta', type=float, help='perturbation upper bound')
    opt = parser.parse_args()
    print('CMD Arguments:', opt)
    return opt

if __name__=='__main__':
    opt = parse_opt()
    main(opt)