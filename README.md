# KITT Dataset을 이용해 SSD 학습하기  (Google Coab 환경)

## 1. Data Preparation and processing 

(1) Download KITTI Dataset and Upload on Gdirve  

1. VOC2007 폴더를 생성한 후 내부에 JPEGImages, ImageSets, Annotations 3개의 폴더를 생성한다.  

![image](https://user-images.githubusercontent.com/69920975/122263210-2362c000-cf11-11eb-9ae1-d26932e05be6.png)

2. ImageSets 폴더 내부에 Main 폴더를 생성한다.

3. 링크(link= http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)에 접속하여 left color images of objec data set(12GB)와 training labels of object data set(5MB)를 다운로드한다.  

4. gdrive/VOC2007 내부에 압축파일을 업로드한다. 

5. VOC2007 폴더 내부에 image, label 압축파일을 넣고 해제한다. 

![image](https://user-images.githubusercontent.com/69920975/122277916-6036b300-cf21-11eb-8cc2-9f23529d8510.png)

6. repository를 clone 한다.  

![image](https://user-images.githubusercontent.com/69920975/122329186-ee378b80-cf6b-11eb-9a00-480fe6ef5e33.png)


(2) Data Preprocessing  

1. (Optional) KITTI Dataset의 class ('Car', 'Van', 'Truck','Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' or 'DontCare') 의 개수는 총 9개이다.  
class의 개수를 줄이고 싶다면 Fine tuning을 위해 classnamechange.py 파일 내부의 class를 변경해주고 실행해준다. 본 code에서는 car,pedestrian,truck,van,tram 5가지 class 사용

![image](https://user-images.githubusercontent.com/69920975/122329296-2212b100-cf6c-11eb-839b-9dbf36cee4f0.png)

2.SSD에 사용될 dataset은 Pascal VOC dataset 형식(xml 확장자파일)이므로 KITTI Dataset label(txt 확장자파일)을 변경해줄 필요가 있다.
따라서 txt_to_xml.py 파일을 실행해준다.  
![image](https://user-images.githubusercontent.com/69920975/122329308-28089200-cf6c-11eb-99cb-e585eefb9973.png)


3. trainval, test dataset으로 나누기 위해 create_train_test_txt.py 파일을 실행시켜준다.  

![image](https://user-images.githubusercontent.com/69920975/122329321-2f2fa000-cf6c-11eb-81c5-bf10a48760ab.png)


실행 시 ImageSets/Main 폴더 내부에 아래와 같은 파일들이 생기는데, train_val.txt 파일에는 훈련시킬 사진들의 파일명, test.txt 파일에는 시험할 사진들의 파일명이 기록되어있다.  

## 2. Import visdom

학습 과정을 관찰할 수 있는 library visdom을 import한다. 

![image](https://user-images.githubusercontent.com/69920975/122329437-600fd500-cf6c-11eb-8d25-4939f32fe675.png)

## 3. Setting IP and Port for Using visdom  

visdom library(훈련 과정을 관찰하게 해주는 library)를 사용하기 위해 train.py 내부에서  Visualizer class 내부의 생성자 함수에서 visdom.Visdom('자신의 IP주소', port 번호)로 변경해준다.

## 4. Train  

pretrained 된 vgg16모델을 사용하기 위해 내부에 weights 폴더를 만들고 다음과 같은 명령을 실행한다.  

![image](https://user-images.githubusercontent.com/69920975/122329681-cac11080-cf6c-11eb-9155-796aced87296.png)

train.py 파일을 실행시킨다. (train.py 파일 내부에서 하이퍼파라미터를 변경할 수 있다..)  
본인이 설정한 ip주소:port 번호로 접속을 하면 아래와 같이 훈련과정을 살펴보고 그 때의 mAP값을 측정할 수 있다.   

#### Loss  
![image](https://user-images.githubusercontent.com/69920975/122328502-c267d600-cf6a-11eb-87ee-e91bd99e85da.png)


## 5. Eval  

모델 성능을 평가하기 위해 eval.py 파일을 실행한다.  

![image](https://user-images.githubusercontent.com/69920975/122329734-df050d80-cf6c-11eb-9384-9a63d52179c9.png)
  
  
![image](https://user-images.githubusercontent.com/69920975/122329767-ee845680-cf6c-11eb-9896-a81a4b5d3a7f.png)



  


Licensed under MIT, see the LICENSE for more detail.


