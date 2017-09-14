# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder
 
 ## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

## Project stuff
When building my FCN I ran into some difficulty with tensor shapes, partially because of my silly mistakes and not using the tensors that we took out of vgg for keep_prob and image_input.
Then I came to find out I set the number of channels on my `correct_label` tensor to 1 too many. There are only 2 channels on the `correct_label` tensor, because we're doing "road" and "not road". There could be more if there were more classifications. I chose to stick with two for now.
Once that was sorted out it was a bit simpler to figure out similar errors. On my very first run, I set my `batch_size` to 10, and left the `correct_label` set to 10 too. Unfortunately when you get through the first run, we end up with only 9 images on the last batch. I figured out that 289 was divisble by 17 and just decided to use a `batch_size` of 17. I know that I could probably use some sort of TF variable here to make this adjustable, but I found that training on a g3.8xlarge goes pretty fast still.
Now, with my first run completed my image came out looking like this: <uu_000098.png>.
`Epoch: 0010 |  cost = 0.725917161`
This definitely isn't correct, and was actually what I was attempting to avoid by using the kernel regularizer that was suggested in the video. Clearly having the green for road everywhere is not what I want.
I double checked that this isn't working by putting my epochs up to 20 (from the original 10).
Definitely still wasn't working, image came out like this: <uu_000003.png>
`Epoch: 0020 |  cost = 0.701197147`