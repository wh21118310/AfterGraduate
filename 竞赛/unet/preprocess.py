from PIL import Image
import torchvision.transforms as transforms
import glob
tests_path = glob.glob('sardata/train/label/*.png')
# 遍历素有图片
for test_path in tests_path:
 save_res_path = test_path.split('\\')[1]
 a = 'sardata/train/preprocess\\' + save_res_path
 image=Image.open(test_path).convert('RGB')
 t=transforms.Compose([
 transforms.Resize((512, 512)),])
 image=t(image)
 Image_copy = Image.Image.copy(image)

 Image.Image.save(Image_copy,a)
