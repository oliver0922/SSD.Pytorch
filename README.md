# KITT Dataset을 이용해 SSD 학습하기  (Google Coab 환경)

## 1. Data Preparation and processing 

(1) Download KITTI Dataset and Upload on Gdirve  

link= http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d  

1. VOC2007 폴더를 생성한 후 내부에 JPEGImages, ImageSets, Annotations 3개의 폴더를 생성한다.  

![image](https://user-images.githubusercontent.com/69920975/122263210-2362c000-cf11-11eb-9ae1-d26932e05be6.png)

2. ImageSets 폴더 내부에 Main 폴더를 생성한다.

3. 위 링크에 접속하여 left color images of objec data set(12GB)와 training labels of object data set(5MB)를 다운로드한다.  

4. gdrive/VOC2007 내부에 압축파일을 업로드한다. 

6. JPEGImages 폴더 내부에 image 압축파일을 넣고 해제하고, data_object_label_2 압축파일을 해제한다. 

![image](https://user-images.githubusercontent.com/69920975/122277916-6036b300-cf21-11eb-8cc2-9f23529d8510.png)

7. repository를 clone 한다.

(2) Data Preprocessing  

1. (Optional) KITTI Dataset의 class ('Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' or 'DontCare') 의 개수는 총 9개이다.  
class의 개수를 줄이고 싶다면 Fine tuning을 위해 classnamechange.py 파일 내부의 class를 변경해주고 실행해준다.  

![image](https://user-images.githubusercontent.com/69920975/122277987-73e21980-cf21-11eb-8ccd-f91759339c24.png)

2. SSD에 사용될 dataset은 Pascal VOC dataset 형식(xml 확장자파일)이므로 KITTI Dataset label(txt 확장자파일)을 변경해줄 필요가 있다.
따라서 txt_to_xml.py 파일을 실행해준다. 

3. trainval, test dataset으로 나누기 위해 ceat_train_test_txt.py 파일을 실행시켜준다. 


## 2. Setting IP and Port for Using visdom  

visdom library(훈련 과정을 관찰하게 해주는 library)를 사용하기 위해 visdom을 설치해준다.
train.py 내부 viz=visdom.Visdom('ip번호','port번호')를 설정해준다.

## 4. Train(with pretrained weights)  

1. clone 한 디렉토리 내부에 weights 폴더를 만들고 https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth에서 pretrained model을 다운 받는다  
![image](https://user-images.githubusercontent.com/69920975/122281004-c1ac5100-cf24-11eb-84cc-32f0f7756bf0.png)

2.  train.py 내부의 --dataset_root parsing 하는 부분을 VOC2007 폴더 경로로 바꿔준다.  
data/config 파일 내부에서 fine tunning을 위해 SSD512의 num_classes 부분을 class 개수로 바꿔준다. 마찬가지로 HOME 부분을 Clone 한 디렉토리의 제일 바깥의 폴더 경로를 입력해준다. 
data/voc0712.py 내부의 VOC_CLASSES 부분에 검출하고자 하는 class 이름을 적어준다  
                       VOC_ROOT 부분의 HOME 뒷부분에 dataset의 최상 디렉토리 이름을 적어준다. 
                       




  




Licensed under MIT, see the LICENSE for more detail.


