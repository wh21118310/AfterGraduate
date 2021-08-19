import torch,cv2
import numpy as np
import torch.nn as nn
img = cv2.imread('data/cartoon.jpg', cv2.IMREAD_GRAYSCALE)
img = np.array(img,dtype='float32')
img = torch.from_numpy(img.reshape(1,1,img.shape[0],img.shape[1]))
avgPool = nn.AvgPool2d((4,4),stride=(4,4))  #4*4的窗口，步长为4的平均池化

img = avgPool(img)

img = torch.squeeze(img)  #去掉1的维度

img = img.numpy().astype('uint8')  #转换格式，准备输出

cv2.imwrite("data/out.jpg", img)

cv2.imshow("result", img)

cv2.waitKey(0)

cv2.destroyAllWindows()