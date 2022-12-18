import copy
import os
import cv2
import json
import numpy as np


class plot_result():
    def __init__(self, gt, predict, predict_1, id):
        self.predict = predict
        self.predict_1 = predict_1
        self.name = gt[id]['name']
        self.task = gt[id]['task']
        gt_path = np.array([np.array(gt[id]['X']), np.array(gt[id]['Y'])])
        img_path = 'data/TP/COCOSearch18-images-TP/images/' + self.task
        for img_name in os.listdir(img_path):
            if img_name == self.name:
                self.img = cv2.imread(img_path+'/'+img_name)
                self.img = cv2.resize(self.img, (512, 320))
                img_copy = copy.deepcopy(self.img)
                break

        for i in range(gt_path.shape[1]):
            cv2.circle(self.img, (int(gt_path[0][i]), int(gt_path[1][i])), 5, (100,100,0), 1)
            if i < gt_path.shape[1]-1:
                cv2.line(self.img,  (int(gt_path[0][i]), int(gt_path[1][i])),  (int(gt_path[0][i+1]), int(gt_path[1][i+1])), (255,255,0), 2)
            #cv2.imwrite('result/gt_path.jpg', self.img)


    def plot_path(self, predict, output_file, color_ratio):
        for p in predict:
            if p['name'] == self.name and p['task'] == self.task:
                predict_path = np.array([np.array(p['X']), np.array(p['Y'])])
                break

        for i in range(predict_path.shape[1]):
            cv2.circle(self.img, (int(predict_path[0][i]), int(predict_path[1][i])), 5, (100, 0, 100), 1)
            if i < predict_path.shape[1]-1:
                cv2.line(self.img,  (int(predict_path[0][i]), int(predict_path[1][i])),
                         (int(predict_path[0][i+1]), int(predict_path[1][i+1])), (255*color_ratio, 0, 255*color_ratio), 2)
            if color_ratio != 1:
                cv2.putText(self.img, str(i), (int(predict_path[0][i]), int(predict_path[1][i])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        if color_ratio != 1:
            cv2.imwrite(output_file, self.img)


if __name__ == '__main__':
    with open('data/human_scanpaths_TP_trainval_valid.json') as f1:
        gt = json.load(f1)

    with open('assets\log_20221130_1722/predicted_scanpaths.json') as f2:
        predict = json.load(f2)

    with open('assets\log_20221208_1725/predicted_scanpaths.json') as f3:
        predict_1 = json.load(f3)

    with open('assets\log_20221208_2334/predicted_scanpaths.json') as f4:
        predict_2 = json.load(f4)

    ids = [x for x in range(len(predict))]
    for id in ids:
        plot_r = plot_result(gt, predict, predict_1, id)
        plot_r.plot_path(predict, 'result/original/'+plot_r.task+'/'+plot_r.name, 0.5)
        #plot_r.plot_path(predict_1, 'result/1/'+plot_r.task+'/'+plot_r.name, 0.5)
        #plot_r.plot_path(predict_2, 'result/2/' + plot_r.task + '/' + plot_r.name, 0.2)
        #plot_r.plot_path(predict_2, 'test.jpg', 0.2)


