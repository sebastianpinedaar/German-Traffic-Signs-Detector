# German-Traffic-Signs-Detector

Example of execution (assuming we are on the same folder as this README)

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