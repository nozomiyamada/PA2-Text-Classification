# PA2-Text-Classification
<pre>
MaxEnt model train and test accuracies
0.869949522780876
0.6356272180219102
DAN model train and test accuracies
0.7137566953953314
0.6219719179139022
DAN model2 train and test accuracies
0.7750126744109154
0.6161857738003395
</pre>

## Deep Averaging Network 1
* word embedding - Thairath Corpus (500,000articles, skip-gram, 20epoch, 300dim)
* ฝึกโมเดลโดยใช้ keras
* 3 hidden layers: 300 > 200 > 100 > 50 > 12
* optimizer = adam, loss function = cross entropy

![image](https://user-images.githubusercontent.com/44984892/53429716-be24f100-3a1f-11e9-984f-5432de5d8d47.png)
![image](https://user-images.githubusercontent.com/44984892/53429724-c2510e80-3a1f-11e9-84c4-dab668d2d793.png)

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

## Deep Averaging Network 2 with Dropout
* Layers: 127813 > 300 > 200 > 100 > 30 > 12
* dropout rate = 0.2
* optimizer = adam, loss function = cross entropy

![dropout1](https://user-images.githubusercontent.com/44984892/53444671-626b5f80-3a41-11e9-8ad8-8427e7f343fb.png)
![dropout2](https://user-images.githubusercontent.com/44984892/53444672-626b5f80-3a41-11e9-8ebc-5ecdf7a77b2b.png)
