# Computer Vision for Classification of Defects in Silicon Nanopillar Arrays
## Abstract:
This project centers around development of a computer vision approach for classification of defects in arrays of silicon nanopillars that have been fabricated on a silicon wafer.

## Project Background:
During my time at UT Austin as a PhD student, I researched metrology solutions for nano-manufacturing. My work centered around using spectral imaging system to charactize devices that we called "large area nanostructure arrays" which includes things like nanopillar arrays, grating structures, and mesh structures which have been fabricated on flat large area substrates such as silicon wafers, glass sheets, roll-to-roll webs, etc. People are interested in making these kinds of structures because they have many important applications including ones in displays, memory storage devices, electronics, etc. In order to manufacture these kinds of devices succesfully, metrology systems (systems that can measure and characterize these devices) need to be in place to measure their quality as they are being made. It's important to have metrology so that defects can be detected as they appear in the devices. Furthermore, the defects can to be *classified* so that the manufacturing facility knows specifically what went wrong and people can hopefully fix things so that devices can continue to be made correctly. 

Silicon nanopillar arrays were of particular interest to me, and much of my work was centered around them. They have many interesting applications, although what really got me hooked on them initially was the fact that they create these really amazing colors (see images below) due to a phenomenon called structural coloration (see my publication [reference 2] for more info). The colors arise because of the nanoscale size and shape of the pillars, and subtle changes in the geometry - changes on the order of a few nanometers, for instance - can completely change the color that they exhibit. As it turns out, this is an extremely useful characteristic from a metrology perspective, because characterizing this color can give insight into what's happening on the nanoscale, which otherwise can't be seen without the use of tools like electron micrscopes. Electron microscopes are extremely slow, and so being able to do an optical characterization that can image a full wafer *in seconds* is highly preferred. 

<img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Wafers.jfif" title="Images of Silicon Nanopillar Array Wafers">

One of my primary focuses of my PhD was developing computer vision algorithms for detecting and classifying defects in these Si nanopillar arrays. I actually published a paper on this work [1] which shows how more-traditional image processing approaches can be used. This work was fairly rudimentary from a computer vision perspective, and wouldn't have been accepted to a computer vision journal, but in the context of metrology for silicon nanopillar manufacturing, the work offers insight at the interface of computer vision and the manfuacturing itself and demonstrates beginnings for more advanced algorithms. After I graduated, I wanted to re-explore this problem in the context of machine learning based computer vision, and thus this project was born.

## Description of the Problem:
I fabricated a wafer containing arrays of silicon nanopillars of which I recorded an RGB image of using my imaging system. If fabrication was 100% succesful, the wafer would look like the left side of the image below. The pattern is only intended for certain areas on the wafer. Of course, the wafer I fabricated (shown on the right) is imperfect - actually, it is plagued by many defects. This image needs to be processed in such a way as to automatically detect and classify the defects that are present in the wafer. 

<p float="left">
    <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Ideal Wafer.jpg" height="300" title="Rendering of an Ideal Wafer"/>
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/RGB_jpg.jpg" height="300" title="RGB Image"/>
</p>

### What's on the wafer?
The wafer, which is 100 mm in diameter, contains nearly 2000 1x1 mm square arrays of silicon nanopillars. Each nanopillar in the arrays has a diameter of ~100 nm and 200 nm pitch. The arrays give off a vibrant green color due to structural coloration. This green color signifies succesful fabrication of the arrays, because only the target geometry would produce this color. The color is *really* sensitive to the exact geomety of the arrays, so any other geometry (what we call a defect) will produce a different color. These defects can arise from a set of different root causes in the manufacturing process including contimination, non-optimized fabrication processes, faulty tools, etc. For the most part, the defects on this wafer fall into one of seven categories:
1. particle void
2. non-fill void
3. etch delay
4. edge etch delay
5. edge non-etch
6. edge non-fill
7. scratch

Understanding these defect types requires knowledge of various nanofabrication processes that were involved in making the wafer. If you want, you can read reference [1] to learn about these defect modes. Otherwise, all you really need to understand for this project is that there are different types of defects which we are trying to classify.

## Methods:

### Pre-Processing:
  The RGB image is converted to a color-indexed (also could be called color-quantized) image to reduce the size of the color space to just a handful of colors (red, black, gray, green, faded green). These colors are by far the most popular colors on the wafer and are useful for the subsequent feature engineering.
 
<p float="left">
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/mask_sqrs.png" width="30%" title="Squares Mask Image"/>
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/RGBi_jpg.jpg"  width="30%" title="Indexed RGB Image"/>
</p>

Next, the indexed RGB image is masked to isolate the 1x1 mm square device regions on the wafer using the image mask "mask_sqrs.png". Each image of each device is then considered as an example in this machine learning problem. The devices and their associated information (pixel coordinates, bounding box, etc.) are arranged in a pandas dataframe which is utilized for the machine learning.

### Feature Engineering:
Most of the effort that went into this project went into feature engineering to provide relevant features to the classifier. Features include: 
* the fraction of device that is defective
* the fractions of each device that are of each of one of the following indexed colors (red, black, gray, green, faded green)
* if the device is at an edge of the pattern area (a boolean feature 0/1)
* the fraction of the perimeter pixels of gray areas that touch each of the following colors (black, faded green, green)

### Labelling the Training Data:
The script "labeller_updatable.py" was used to manually label the training data. This script (at least) works in the IDE Spyder and basically shows the image of a device from the training set one at a time and asks for input as to what defects are present in the device. The input options are abbreviations I came up with for the various defect types ('p','nf','ed','eed','ene','enf','s'). I would enter a comma-separated list of these abbreviations and press enter. The script allows for 'quit' to be entered as well, which allowed me to end the program and take a break from labelling. It would save the current progress and I could then come back later and continue labelling. Hence, the *updatable* aspect of the labeller. The label lists are then automatically parsed and transformed into a sparse boolean array (one-hot encoding).

>*As mundane as this portion of the project sounds, the ability to label quickly and efficiently was crucial.* 
  
### Model Training and Evaluation:
This approach utilizes the random forest classifier from the Scikit-Learn library. The classification is multi-output, meaning that each device on the wafer can have any number of the 7 defect types. In fact, many of the devices are plagued by multiple types of defects which may or may not be overlapping each other, so the multi-output style classifier is most effective. Originally, I tried to distill the problem to a single-output problem, in which the most prominent defect type in each device would be the output, but this had undesirable results, including that it made the labelling process very subjective.

The training/evaluation set is formed by randomly sampling 100 devices from the ~2000 devices on the wafer. The training and evaulation sets are then split 67% and 33% (respectively) at random. The classifier is trained on the training set and then evaluated on the evalution set (go figure). 

## Results and Discussion:
### Evaluation Scores:

|   Defect Type   | # of Training Examples | Precision | Recall |
|:--------------- |:------------------:    | :--------:| :----: |
| particle void   |     17                 | 100       | 83     |
| non-fill void   |     42                 | 100       | 58     |
| etch delay      |     40                 |  92       | 86     |
| edge etch delay |     45                 | 100       |100     |
| edge non etch   |      2                 | NaN       |  0     |
| edge non-fill   |      0                 | NaN       |NaN     |
| scratch         |     29                 |  50       | 20     |

The high overall recall value suggests that the model is quite effective at minimizing false negatives. So, the model is generally good at classifying defects *when they are present*. The precision is somewhat lower, which is because the model has a slightly higher percentage of false positives. What this means is that the model is generally effective at correctly classifying defects when they are present, but also tends to classify defects when they are not present - in other words, "crying wolf".

The number of the different defect classes vary dramatically, so the precision and recall for each defect type are listed...

### Test Results Visualization:
The images below show the classification predictions made by the model for 3 of the 7 defect types for the whole wafer. The results are visualized by gray'ing out the entire image of the wafer except for the device regions in which that particular defect was detected, which are given their normal RGB values. 

(left-to-right: particle voids, edge edge delay, and non-fill voids) (hover to see title). 

<p float="left">
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/classification_image_p_predict.jpg" width="30%" title="devices with classified particle defects"/> 
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/classification_image_eed_predict.jpg" width="30%" title="devices with classified edge etch delay defects"/> 
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/classification_image_nf_predict.jpg"  width="30%" title="devices with classified non-fill defects"/>
</p>

## Outstanding Issues:
1. Certain types of defects were either over- or under-represented in the training dataset, causing clear biases in the classification. Edge non-fill, for instance, has only a few instances on the entire wafer and happened to not be randomly sampled for the training dataset and therefore is not represented at all. Manually, devices impacted by edge non-fill can be forced to be a part of the training dataset, although the training will still suffer from having so few instances. On the other hand, edge etch delay seems over-represented. The training set could be forced to have a uniform amount of training examples of each type.
2. 

## References:
[1] ASME Paper.

[2] https://www.osapublishing.org/oe/abstract.cfm?uri=oe-26-23-30952
