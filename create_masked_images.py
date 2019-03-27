import cv2
import os

if __name__ == '__main__':

    path = './data'
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))

    for f in files:
        path_trimap = './data/annotations/trimaps/' + f[f.rfind('/') + 1:-3] + 'png'
        save_path = './data/masked_images/' + f[2: f.rfind('/')]
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        imgCut = cv2.imread(f)
        imgCut = cv2.cvtColor(imgCut, cv2.COLOR_BGR2RGB)
        maskimg = cv2.imread(path_trimap)
        maskimg = cv2.cvtColor(maskimg, cv2.COLOR_BGR2RGB)

        [rows, columns, channels] = imgCut.shape
        for c in range(channels):
           for row in range(rows):
               for column in range(columns):
                   if(maskimg[row,column, c] == 2):
                       imgCut[row,column, c] = 0
        cv2.imwrite(save_path + '/' + f[f.rfind('/') + 1:], cv2.cvtColor(imgCut, cv2.COLOR_RGB2BGR))
        print ('saving image ' + '/' + save_path + f[f.rfind('/') + 1:])
