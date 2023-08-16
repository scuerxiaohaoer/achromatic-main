# achromatic-main
![image](https://github.com/scuerxiaohaoer/achromatic-main/assets/112793866/930487bd-3e90-4286-824d-c566a84b2a49)

# Deep learned virtual lens conjugated with real singlet lens for broadband achromatic imaging

This repository contains the official implemention of the paper *Deep learned virtual lens conjugated with real singlet lens for broadband achromatic imaging* .

### Testing
Please download the pre-trained model ckpt at
https://www.kaggle.com/datasets/tianyuehe/all-results

Please download the test dataset at
https://www.kaggle.com/datasets/tianyuehe/test-pkl

You can run the testing script with
> python response_recovery.py test_external

where you can change the parameters and the image path to test your own example.

### Training

run
> python response_recovery.py train

You are now all set to train the model! 
