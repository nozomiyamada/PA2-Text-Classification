# PA2-Text-Classification
## Maximum Entropy

* Accuracy of train data: 
* Accuracy of development data: 

## Deep Averaging Network 1
* word embedding - Thairath Corpus (500,000articles, skip-gram, 20epoch, 300dim)
* ฝึกโมเดลโดยใช้ keras
* 3 hidden layers: 300 > 200 > 100 > 50 > 12
* optimizer = adam, loss function = cross entropy

![image](https://user-images.githubusercontent.com/44984892/53429385-232c1700-3a1f-11e9-8421-fd54f5a7a193.png)
![image](https://user-images.githubusercontent.com/44984892/53429393-28896180-3a1f-11e9-9d9f-7d2c921cde8a.png)

![image](https://user-images.githubusercontent.com/44984892/53429716-be24f100-3a1f-11e9-984f-5432de5d8d47.png)
![image](https://user-images.githubusercontent.com/44984892/53429724-c2510e80-3a1f-11e9-84c4-dab668d2d793.png)

![image](https://user-images.githubusercontent.com/44984892/53430199-a13ced80-3a20-11e9-9284-7e34fb09bab1.png)
![image](https://user-images.githubusercontent.com/44984892/53430207-a732ce80-3a20-11e9-9da1-22414fdd28a4.png)

![image](https://user-images.githubusercontent.com/44984892/53431039-52905300-3a22-11e9-8df4-0b8807892ac7.png)
![image](https://user-images.githubusercontent.com/44984892/53431048-57ed9d80-3a22-11e9-86a7-2ef7f487741b.png)

|epoch | 10 | 30 | 50 | 100 |
|:-:|:-:|:-:|:-:|:-:|
|train accuracy| 0.6808 | 0.7157 | 0.7579 | 0.8830 |
|validate accuracy|0.6925 | 0.6872 | 0.6877 | 0.6507 |

ในกรณี 100 epoch มันน่าจะ overfitting เพราะ validation loss ขึ้นเยอะ เพราะฉะนั้นจะใช้โมเดล 40 epoch 

## Deep Averaging Network 2
* ฝึกโมเดลโดยใช้ keras (ใช้ Tesla K80 GPU บน Google Colab)
* Layers: 127813 > 300 > 100 > 30 > 12
* optimizer = adam, loss function = cross entropy

![unknown](https://user-images.githubusercontent.com/44984892/53440129-6940a500-3a36-11e9-9909-71849efe3460.png)
![unknown1](https://user-images.githubusercontent.com/44984892/53440136-6ba2ff00-3a36-11e9-8d03-fb9757530ab0.png)

แค่ 5 epoch ก็ได้ accuracy สูงกว่า DAN1 (Accuracy 0.7857, Validation Accuracy 0.6923)

![1](https://user-images.githubusercontent.com/44984892/53441004-6c3c9500-3a38-11e9-8346-9be31058fdc5.png)
![2](https://user-images.githubusercontent.com/44984892/53441005-6c3c9500-3a38-11e9-9c43-2d06e22865a1.png)

ส่วนกรณี 10 epoch เกิด overfitting เข่นเดียวกันกับ DAN1
