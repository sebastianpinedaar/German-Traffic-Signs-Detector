# German-Traffic-Signs-Detector

These scripts implement three different models to classify images of the data set: German-Traffic-Signs-Detector.

Model1: Logistic regression in scikit-learn
Model2: Logistic regression in Tensorflow
Model3: LeNet5 in Tensorflow

File description:

- app.py: main file to execute the different commands
- download.py: download data
- imshow_labels.py: implement function to show images with predicted label
- model1.py: implement functions to train and test model1
- model2.py: implement functions to train and test model2
- model3.py: implement functions to train and test model3

We expect that the images are in folders depending on the class to which they belong.

Example of execution (assuming we are on the same folder as this README)

To execute download command:

	>> python app.py download

To execute models inference:

- Model1:
	>> python app.py infer --m model1 --d images/user
- Molde2:
	>> python app.py infer --m model2 --d images/user
- Molde3:
	>> python app.py infer --m model3 --d images/user


To execute training:

- Model1:
	>> python app.py train --m model1 --d images/train
- Model2:
	>> python app.py train --m model2 --d images/train
- Model3:
	>> python app.py train --m model3 --d images/train

To execute test:

- Model1:
	>> python app.py test --m model1 --d images/test
- Model2:
	>> python app.py test --m model2 --d images/test
- Model3:
	>> python app.py test --m model3 --d images/test