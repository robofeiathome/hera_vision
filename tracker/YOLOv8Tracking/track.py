#Teste tracker v1 --ainda nao funciona--
#import hydra
import torch
import cv2
from random import randint
from SORT import *
import numpy as np
#from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

#tracking (necessita do arquivo SORT.py)
#-------------------------------------------------------------------------------------------
tracker = None
def init_tracker():
    global tracker
    
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    tracker = Sort(max_age=sort_max_age,min_hits=sort_min_hits,iou_threshold=sort_iou_thresh)

rand_color_list = []
    

def random_color_list():
    global rand_color_list
    rand_color_list = []
    for i in range(0,5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)
    #......................................
        

def draw_boxes(img, bbox, identities=None, categories=None, names=None,offset=(0, 0)):
    
    for i, box in enumerate(bbox):
        
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        
        data = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = names[cat]
        
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,20), 2)
        
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, [255, 255, 255], 1)
        
        cv2.circle(img, data, 3, (255,255,255),-1)   #centroid of box
        
    return img
#----------------------------------------------------------------------------------------

#Detector padrao da yolo com algumas modificacoes para rodar com o tracker personalizado
#-----------------------------------------------------------------------------------------------------
class DetectionPredictor(BasePredictor):
    
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = img[..., ::-1]  # BGR para RGB
        img = img.copy()  #copia a imagem para evitar ``negative stride error``
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 para fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression (preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det) 
                                        
        
        #results = []
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            #results.append(Results(boxes=pred, orig_shape=shape[:2]))
        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.source_type.webcam or self.source_type.from_img: # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        # tracker
        self.data_path = p
    
        #save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)
        
        det = preds[idx]
        #self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detecta por classe
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
    
    
        # #..................Usa a func de track....................
        dets_to_sort = np.empty((0,6))
        
        for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
            dets_to_sort = np.vstack((dets_to_sort, 
                        np.array([x1, y1, x2, y2, conf, detclass])))
        
        tracked_dets = tracker.update(dets_to_sort)
        tracks =tracker.getTrackers()
        
        for track in tracks:
            [cv2.line(im0, (int(track.centroidarr[i][0]),
                        int(track.centroidarr[i][1])), 
                        (int(track.centroidarr[i+1][0]),
                        int(track.centroidarr[i+1][1])),
                        rand_color_list[track.id], thickness=3) 
                        for i,_ in  enumerate(track.centroidarr) 
                            if i < len(track.centroidarr)-1 ] 
        

        if len(tracked_dets)>0:
            bbox_xyxy = tracked_dets[:,:4]
            identities = tracked_dets[:, 8]
            categories = tracked_dets[:, 4]
            draw_boxes(im0, bbox_xyxy, identities, categories, self.model.names)

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        
        return log_string
#----------------------------------------------------------------------------------------------------

#@hydra.main(version_base=None, config_path=str(DEFAULT_CFG.parent), config_name=DEFAULT_CFG.name)

#tracker+detection
#----------------------------------------------------------------------------------------------------
def predict(cfg):
    init_tracker()
    model = cfg.model or "yolov8n.pt"
    #cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    stream=True
    #nao funcoiona se usar source=0 e depois o video capture (source)
    source = cfg.source if cfg.source is not None else ROOT / "assets"
    cap = cv2.VideoCapture(source)
    random_color_list()

    while cap.isOpened():
    # pega o frame da camera
        ret, frame = cap.read()
        if not ret:
            break

        predictor = DetectionPredictor(cfg)
        results=predictor(frame)
        
        #Outro metodo para mostrar as boxes e masks
        #--------------------------------------------------------------------------
        #for r in results:
        
            #boxes = draw_boxes (frame, log_string )
            #masks = r.masks

        # Desenha as boxes
        #for box in list(boxes()):
        #    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy.view(-1).tolist()]
        #    label = f"{model.names[box.cls.int()]}: {box.conf:.2f}"
        #    color = (255, 0, 0)  # blue
        #    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        #    cv2.putText(frame, label, (x1, y1 - 10),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Desenha as mascaras de segmentacao
        #if masks is not None:
        #    for mask, box in zip(masks, boxes):
        #        mask = mask.byte().cpu().numpy()
        #        color = colors[box.cls.int()]
        #        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #        cv2.drawContours(frame, contours, -1, color, 2)

        #cv2.imshow("Detection results", frame)
        #----------------------------------------------------------------------
        
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    cfg=DEFAULT_CFG
    predict(cfg)