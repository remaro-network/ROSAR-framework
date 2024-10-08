import cv2
import numpy as np
import os
import torch
import torchvision

from config import *

_TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]
__all__ = ["meshgrid"]

HWs = [torch.Size([52, 52]), torch.Size([26, 26]), torch.Size([13, 13])]
STRIDES = [8, 16, 32]


def show_pytorch_img(im_scaled):
    # ------------------------ DEBUG ------------------------
    im_tmp = im_scaled.cpu().detach().numpy()[0]
    im_tmp = (im_tmp.transpose(1, 2, 0) * 255).astype(np.uint8)
    im_tmp = cv2.cvtColor(im_tmp, cv2.COLOR_RGB2BGR)
    # print('im_tmp:', im_tmp)
    # print(im_tmp.shape)
    # print(type(im_tmp))
    cv2.imshow('im_tmp', im_tmp)
    cv2.waitKey()
    # -------------------------------------------------------

def draw_all_bboxes_and_plot(preproc_img, final_pred, name='Final predictions'):
    pred_img = preproc_img[0].clone().detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    for idx, prediction in enumerate(final_pred):
        # print("Original:", [value.item() for value in raw_pred[0][indices_map[idx]]])
        # print("New pred:", [value.item() for value in prediction])
        # prediction[4] = raw_pred[0][indices_map[idx]][4]

        pred_img = vis_box(pred_img.copy(), prediction[0:4], prediction[4], int(prediction[-1]))

    cv2.imshow(name, pred_img)
    # cv2.waitKey()

def torch_to_numpy(im_scaled):
    im_tmp = im_scaled.cpu().detach().numpy()[0]
    im_tmp = (im_tmp.transpose(1, 2, 0) * 255).astype(np.uint8)
    im_tmp = cv2.cvtColor(im_tmp, cv2.COLOR_RGB2BGR)
    return im_tmp

def _preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    # cv2.imshow('padded_img', padded_img)

    # cv2.imshow('img', padded_img)
    # cv2.waitKey()


    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def preproc(img, input_size):
    img, _ = _preproc(img, input_size, SWAP)
    if LEGACY:
        img = img[::-1, :, :].copy()
        img /= 255.0
        img -= np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        img /= np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    return img, np.zeros((1, 5))

def preprocess(img):
    img_info = {"id": 0}
    if isinstance(img, str):
        img_info["file_name"] = os.path.basename(img)
        img = cv2.imread(img)
    else:
        img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img

    # cv2.imshow('raw_img', img)
    # cv2.waitKey()

    ratio = min(TEST_SIZE[0] / img.shape[0], TEST_SIZE[1] / img.shape[1])
    img_info["ratio"] = ratio

    img, _ = preproc(img, TEST_SIZE)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.float()
    # print(img)
    # print(img.shape)
    # print(type(img))
    if DEVICE == "gpu":
        img = img.cuda()
        if FP16:
            img = img.half()  # to FP16
    
    return img, img_info

def meshgrid(*tensors):
    if _TORCH_VER >= [1, 10]:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)


def decode_outputs(outputs, dtype):
    # print('outputs:', outputs)

    if not DECODE_IN_INFERENCE:
        return outputs
    else:
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(HWs, STRIDES):
            # print('hsize, wsize, stide:', hsize, wsize, stride)
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        # print('outputs.shape:', outputs.shape)
        # print(grids.shape)
        # print(strides.shape)

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)
        # print('new outputs:', outputs)

        return outputs
    
def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output, indices_map = [None for _ in range(len(prediction))], [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # print(image_pred[:, 5:5+num_classes])

        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        # print(class_conf)
        # print(class_pred)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        original_indices = np.arange(len(detections))
        # print('original shape:', detections.shape)
        # print('first mask:', conf_mask)

        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        original_indices = original_indices[conf_mask]
        # print(original_indices)

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        original_indices = original_indices[nms_out_index]
        # print('final_indices:', original_indices)
        if output[i] is None:
            output[i] = detections
            indices_map[i] = original_indices
        else:
            output[i] = torch.cat((output[i], detections))
            indices_map[i] = torch.cat((indices_map[i], original_indices))

    return output, indices_map

def visual(output, img_info, cls_conf=0.35, image_name="", frame=0):
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    if output is None:
        return img
    output = output.cpu()

    bboxes = output[:, 0:4]

    # preprocessing: resize
    bboxes /= ratio

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]

    vis_res = vis(img, bboxes, scores, cls, cls_conf, C_NAMES, image_name, frame)
    return vis_res

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None, name_image=None, frame=0):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()

        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

def vis_box(img, box, score, class_id):
    x0 = int(box[0])
    y0 = int(box[1])
    x1 = int(box[2])
    y1 = int(box[3])

    color = (_COLORS[class_id] * 255).astype(np.uint8).tolist()

    text = '{}:{:.1f}%'.format(C_NAMES[class_id], score * 100)
    txt_color = (0, 0, 0) if np.mean(_COLORS[class_id]) > 0.5 else (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

    txt_bk_color = (_COLORS[class_id] * 255 * 0.7).astype(np.uint8).tolist()

    cv2.rectangle(
        img,
        (x0, y0 + 1),
        (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
        txt_bk_color,
        -1
    )
    
    cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)