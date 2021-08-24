# encoding:utf-8
# @Author: DorisFawkes
# @File:
# @Date: 2021/08/21 21:21
import glob

tests_path = glob.glob('./sardata/test/image/*.png')
for path in tests_path:
    save_res_path = path.split('/')[-1]
    a = 'sardata/test/predict\\'+save_res_path
    print(save_res_path)