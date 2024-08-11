# achromatic-main
![微信图片_20240811134741](https://github.com/user-attachments/assets/2cdc138b-6a7e-42e5-9603-512b4069cdd3)

# Wholly differentiable Virtual Lens for ultrathin-plate broadband achromatic imaging

This repository contains the official implemention of the paper *Wholly differentiable Virtual Lens for ultrathin-plate broadband achromatic imaging* .


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
