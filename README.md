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

### Sigmoid, no dropout, 50 epoch
![image](https://user-images.githubusercontent.com/44984892/53577939-23581e00-3ba9-11e9-8378-8ca665211b15.png)
![image](https://user-images.githubusercontent.com/44984892/53577947-27843b80-3ba9-11e9-97f6-180375413fc4.png)

### Relu, no dropout, 50epoch
![image](https://user-images.githubusercontent.com/44984892/53577177-c9a32400-3ba7-11e9-9a5a-d51b0a16f68d.png)
![image](https://user-images.githubusercontent.com/44984892/53577133-b1330980-3ba7-11e9-8dfd-8d4ee3d55429.png)

### Sigmoid, dropout 0.2, 100 epoch
![image](https://user-images.githubusercontent.com/44984892/53574649-9316da80-3ba2-11e9-9e93-27b9b20a0f80.png)
![image](https://user-images.githubusercontent.com/44984892/53574656-98742500-3ba2-11e9-9036-e11aaed1556c.png)

### Relu, dropout 0.2, 100 epoch
![image](https://user-images.githubusercontent.com/44984892/53576373-138b0a80-3ba6-11e9-8803-600a26a88b7e.png)
![image](https://user-images.githubusercontent.com/44984892/53576396-1be34580-3ba6-11e9-9983-241b7faae22c.png)

### Sigmoid, dropout 0.55, 150 epoch

![image](https://user-images.githubusercontent.com/44984892/53562029-e37e4000-3b82-11e9-8edc-589983f4009e.png)
![image](https://user-images.githubusercontent.com/44984892/53562043-eb3de480-3b82-11e9-89ee-29c254d4e215.png)

### Relu, dropout 0.55, 150 epoch

![image](https://user-images.githubusercontent.com/44984892/53570801-f8b29900-3b99-11e9-950b-7548e8fe8be2.png)
![image](https://user-images.githubusercontent.com/44984892/53569835-a2dcf180-3b97-11e9-8bc5-4feb0392c166.png)



## Deep Averaging Network 2
* ฝึกโมเดลโดยใช้ keras (ใช้ Tesla K80 GPU บน Google Colab)
* Layers: 127813 > 300 > 100 > 50 > 12
* optimizer = adam, loss function = cross entropy

## Sigmoid, dropout 0.5, 30epoch


### Relu, dropout 0.5, 30 epoch
![drop5relu1](https://user-images.githubusercontent.com/44984892/53571672-108b1c80-3b9c-11e9-8c54-b7f124af80aa.png)
![drop5relu2](https://user-images.githubusercontent.com/44984892/53571673-1123b300-3b9c-11e9-9723-53eea1851dd5.png)

