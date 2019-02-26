# PA2-Text-Classification

## DAN 1
* word embedding - Thairath Corpus (skip-gram, 20epoch, 300dim)
* ฝึกโมเดลโดยใช้ keras
* 3 hidden layers: 300 > 200 > 100 > 50 > 12
* optimizer = adam, loss function = cross entropy

![image](https://user-images.githubusercontent.com/44984892/53392539-3eba0200-39cc-11e9-9ec4-28b026a4765d.png)
![image](https://user-images.githubusercontent.com/44984892/53391016-a15ccf00-39c7-11e9-99aa-8f80d953b635.png)
![image](https://user-images.githubusercontent.com/44984892/53391470-0664f480-39c9-11e9-8f7d-3d8aef7ee7e4.png)
![image](https://user-images.githubusercontent.com/44984892/53392050-cd2d8400-39ca-11e9-92d1-727b1676c0f8.png)

|epoch | 10 | 30 | 50 | 100 |
|:-:|:-:|:-:|:-:|:-:|
|train accuracy| 0.6820 | 0.7134 | 0.7580 | 0.8855 |
|validate accuracy| 0.6324 | 0.6877 | 0.6784 | 0.6341 |

![image](https://user-images.githubusercontent.com/44984892/53393603-6363a900-39cf-11e9-84dc-c9768686df1c.png)

ในกรณี 100 epoch มันน่าจะ overfitting เพราะ validation loss ขึ้นเยอะ เพราะฉะนั้นจะใช้โมเดล epoch=50 
