# image_recognation_project :octocat:

This library aims to build a training, testing ans validation sets to recognize between two classes.
Using
  - [Tesnsorflow]  - to train and test model
  - [Numpy] - to load data 
  - [matplotlib] - to visualize the validation result
  
# demo :rocket:
We have two classs: ```rose``` and ```sunflower```  and the images of then are stored in the folder. There are about 1500 rose pictures and about 900 sunflower pictures in the two folders. Then I use the model selection from sklearn to ramdomly split the training data(70%) and testing data(30%). And I use keras ```CNN``` models to train the date. 
Then, run the follwing code.
```python 
python flower_model.py
```
It builds a CNN deep learning model to ```train``` the images, and 5 epoches to improve the accuracy. The ```test``` accuracy will be calculated by evaluate the testing data. For these class, the test accuracy is 0.8968166850270084. 


Then, we predict the test data to do the ```validation```. the library will show the plots of top 300 test images. The test labes, as well as the confidence index and prediction labes from the models, are listed below each image. The plot images are attached to this repository. Wrong prediction will marked in red color.

# user api  :+1:
just chage pictures in the folder! run the model, and it will automatically generate models and split 30% of the data to be the test data.

If you want to use the model to predict on a single image, add the following codes.
```python 
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
#prediction
predictions_single = model.predict(img)
#show the result
print("The prediction of the image is" + class_names[np.argmax(predictions_single[0])])

```
The predicton result will be printed.

# referance
[Tensiflow Tutorial](https://www.tensorflow.org/tutorials/keras/)
