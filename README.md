# PA2-Text-Classification
<pre>
MaxEnt model train and test accuracies
0.869949522780876
0.6356272180219102
DAN model train and test accuracies (150epoch, dropout 0.55)
0.7415301871404325
0.6528313531862366
DAN model2 train and test accuracies
0.7750126744109154
0.6161857738003395
</pre>

## Deep Averaging Network 1
* word embedding - Thairath Corpus (500,000articles, skip-gram, 20epoch, 300dim)
* ฝึกโมเดลโดยใช้ keras
* 3 hidden layers: 300 > 200 > 100 > 50 > 12
* optimizer = adam, loss function = cross entropy

### Sigmoid, no dropout
![image](https://user-images.githubusercontent.com/44984892/53571861-7a0b2b00-3b9c-11e9-8bfd-42567065d863.png)
![image](https://user-images.githubusercontent.com/44984892/53571890-87c0b080-3b9c-11e9-86f5-1fa11161679b.png)

### Relu, no dropout
![image](https://user-images.githubusercontent.com/44984892/53571256-03b9f900-3b9b-11e9-83be-8e91ef676316.png)
![image](https://user-images.githubusercontent.com/44984892/53571259-07e61680-3b9b-11e9-9d64-c5c73fc07d57.png)

## Deep Averaging Network with Dropout

* 3 hidden layers: 300 > 200 > 100 > 50 > 12
* optimizer = adam, loss function = cross entropy

### Sigmoid, 150 epoch, dropout 0.55

![image](https://user-images.githubusercontent.com/44984892/53562029-e37e4000-3b82-11e9-8edc-589983f4009e.png)
![image](https://user-images.githubusercontent.com/44984892/53562043-eb3de480-3b82-11e9-89ee-29c254d4e215.png)

### Relu, 150 epoch, dropout 0.55

![image](https://user-images.githubusercontent.com/44984892/53570801-f8b29900-3b99-11e9-950b-7548e8fe8be2.png)
![image](https://user-images.githubusercontent.com/44984892/53569835-a2dcf180-3b97-11e9-8bc5-4feb0392c166.png)



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
