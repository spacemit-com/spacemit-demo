import cv2
import numpy as np
import onnxruntime as ort
import spacemit_ort
import time


class Detection:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):        
        self.conf_thres = conf_thres
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(
        model_path,        
        providers=["SpaceMITExecutionProvider"],
        sess_options=session_options
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.input_shape = self.sess.get_inputs()[0].shape[2:4]


    def infer(self, img_src):
        img = img_src.copy()
        img = self.process_img(img)
        model_infer = self.sess.run(None, {self.input_name: img})
        pred = self.post_process(model_infer)
        pred = self.non_max_suppression_face(pred, self.conf_thres, self.iou_thres)
        
        for det in(pred[0]):
            det[:4] = scale_coords(img.shape[2:], det[:4], img_src.shape).round()
            
        return pred



    def process_img(self, img):
        img = letterbox(img,self.input_shape)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).copy()

        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        img = img[None]
        return img

    def post_process(self, output):
        init_anchors = np.array([4.,5.,8.,10.,13.,16.,23.,29.,
                                43.,55.,73.,105.,146.,217.,
                                231.,300.,335.,433.]).astype(np.float32).reshape(3,1,3,1,1,2)
        strides = [8, 16, 32]
        pred_list = []
        for idx, pred in enumerate(output):
            y = np.full_like(pred[..., :6], 0)
            bs, _, ny, nx, _ = pred.shape
            grid = _make_grid(nx, ny)
    
            y_tmp = sigmoid(np.concatenate([pred[..., :5], pred[..., 15:]], axis=-1))
            y[..., :2] = (y_tmp[..., :2] * 2.-0.5+grid) * strides[idx]
            y[..., 2:4] = (y_tmp[..., 2:4] * 2) ** 2 * init_anchors[idx]
            y[..., 4:] = y_tmp[..., 4:]
            pred_list.append(y.reshape(bs, -1, 6))
        pred_result = np.concatenate(pred_list, 1)
        return pred_result
    

    @staticmethod
    def non_max_suppression_face(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        time_limit = 10.0  # seconds to quit after

        t = time.time()
        output = [np.zeros((0, 6))] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                l = labels[xi]
                v = np.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = np.concatenate((x, v), 0)
            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            box = xywh2xyxy(x[:, :4])
            conf = x[:, 5:].max(1).reshape(-1, 1)
            class_idx = np.argmax(x[:, 5:], axis=1).reshape(-1, 1).astype(np.float32)
            x = np.concatenate((box, conf, class_idx), 1)[conf.reshape(-1) > conf_thres]

            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = nms(boxes, scores, iou_thres)  # NMS

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break  # time limit exceeded

            return output
        
        

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[0] = np.clip(boxes[0], 0, img_shape[1])  # x1
    boxes[1] = np.clip(boxes[1], 0, img_shape[0])  # y1
    boxes[2] = np.clip(boxes[2], 0, img_shape[1])  # x2
    boxes[3] = np.clip(boxes[3], 0, img_shape[0])  # y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[[0, 2]] -= pad[0]  # x padding
    coords[[1, 3]] -= pad[1]  # y padding
    coords[:4] /= gain
    clip_coords(coords, img0_shape)
    return coords
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def letterbox(img, new_shape=(320, 320), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def _make_grid(nx=20, ny=20):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

def nms(bboxes,scores,threshold=0.5):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order)>0:
        i = order[0]
        keep.append(i)
        if len(order)==1:
            break

        xx1 = x1[order[1:]].clip(min=x1[i])
        yy1 = y1[order[1:]].clip(min=y1[i])
        xx2 = x2[order[1:]].clip(max=x2[i])
        yy2 = y2[order[1:]].clip(max=y2[i])

        w = (xx2-xx1).clip(min=0)
        h = (yy2-yy1).clip(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero()[0]
        if len(ids) == 0:
            break
        order = order[ids+1]
    return np.array(keep)

