# PA2-Text-Classification
## Maximum Entropy

* Accuracy of train data: 
* Accuracy of development data: 

## Deep Averaging Network 1
* word embedding - Thairath Corpus (skip-gram, 20epoch, 300dim)
* ฝึกโมเดลโดยใช้ keras
* 3 hidden layers: 300 > 200 > 100 > 50 > 12
* optimizer = adam, loss function = cross entropy

![image](https://user-images.githubusercontent.com/44984892/53429385-232c1700-3a1f-11e9-8421-fd54f5a7a193.png)
![image](https://user-images.githubusercontent.com/44984892/53429393-28896180-3a1f-11e9-9d9f-7d2c921cde8a.png)

![image](https://user-images.githubusercontent.com/44984892/53429716-be24f100-3a1f-11e9-984f-5432de5d8d47.png)
![image](https://user-images.githubusercontent.com/44984892/53429724-c2510e80-3a1f-11e9-84c4-dab668d2d793.png)



![image](https://user-images.githubusercontent.com/44984892/53431039-52905300-3a22-11e9-8df4-0b8807892ac7.png)
![image](https://user-images.githubusercontent.com/44984892/53431048-57ed9d80-3a22-11e9-86a7-2ef7f487741b.png)

|epoch | 10 | 30 | 50 | 100 |
|:-:|:-:|:-:|:-:|:-:|
|train accuracy| 0.6808 | 0.7157 | 0.7579 | 0.8830 |
|validate accuracy|0.6925 | 0.6872 | 0.6877 | 0.6507 |

ในกรณี 100 epoch มันน่าจะ overfitting เพราะ validation loss ขึ้นเยอะ เพราะฉะนั้นจะใช้โมเดล 40 epoch 
