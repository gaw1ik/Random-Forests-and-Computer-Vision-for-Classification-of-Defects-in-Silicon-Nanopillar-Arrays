# Computer Vision for Classification of Defects in Silicon Nanopillar Arrays
## Abstract:
This project centers around development of a machine learning computer vision approach for classification of defects in arrays of silicon nanopillars that have been fabricated on a silicon wafer.

## Project Background:
During my time at UT Austin as a PhD student, I researched metrology solutions for nano-manufacturing. My work centered around using spectral imaging system to charactize devices that we called "large area nanostructure arrays" which includes things like nanopillar arrays, grating structures, and mesh structures which have been fabricated on flat large area substrates such as silicon wafers, glass sheets, roll-to-roll webs, etc. People are interested in making these kinds of structures because they have many important applications including ones in displays, memory storage devices, electronics, etc. In order to manufacture these kinds of devices succesfully, metrology systems (systems that can measure and characterize these devices) need to be in place to measure their quality as they are being made. It's important to have metrology so that defects can be detected as they appear in the devices. Furthermore, the defects can to be *classified* so that the manufacturing facility knows specifically what went wrong and people can hopefully fix things so that devices can continue to be made correctly. 

Silicon nanopillar arrays were of particular interest to me, and much of my work was centered around them. They have many interesting applications, although what really got me hooked on them initially was the fact that they create these really amazing colors (see images below) due to a phenomenon called structural coloration (see my publication [reference 2] for more info). The colors arise because of the nanoscale size and shape of the pillars, and subtle changes in the geometry - changes on the order of a few nanometers, for instance - can completely change the color that they exhibit. As it turns out, this is an extremely useful characteristic from a metrology perspective, because characterizing this color can give insight into what's happening on the nanoscale, which otherwise can't be seen without the use of tools like electron micrscopes. Electron microscopes are extremely slow, and so being able to do an optical characterization that can image a full wafer *in seconds* is highly preferred. 

<img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/Wafers.jfif" width="500" title="Images of Silicon Nanopillar Array Wafers">

One of my primary focuses of my PhD was developing computer vision algorithms for detecting and classifying defects in these Si nanopillar arrays. I actually published a paper on this work [1] which shows how more-traditional image processing approaches can be used. This work was fairly rudimentary from a computer vision perspective, and wouldn't have been accepted to a computer vision journal, but in the context of metrology for silicon nanopillar manufacturing, the work offers insight at the interface of computer vision and the manfuacturing itself and demonstrates beginnings for more advanced algorithms. After I graduated, I wanted to re-explore this problem in the context of machine learning based computer vision, and thus this project was born.

## Description of the Problem:
A wafer containing arrays of silicon nanopillars has been fabricated of which I recorded an RGB image of using an imaging system. If fabrication was 100% succesful, the wafer would look like what the left side of the image below shows. Note that the pattern is only intended for certain areas on the wafer which is why only some areas are green. Of course, the wafer I fabricated (shown on the right) is imperfect - actually, it is plagued by many defects. These defects need to be automatically detected and classified, and since they are readily visible in the image data, a computer vision approach makes sense. 

<p float="left">
    <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/Ideal Wafer.jpg" height="300" title="Rendering of an Ideal Wafer"/>
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Images/RGB.jpg" height="300" title="RGB Image"/>
</p>

### What's on the wafer?
<img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/A1 x-sect30.jpg" width="300" title="Cross-sectional Scanning Electron Micrscopy Image of Silicon Nanopillar Array">

The wafer, which is 100 mm in diameter, has nearly 2000 1x1 mm square device regions which contain arrays of silicon nanopillars which look like the ones shown in the image above. The nanopillars have a diameter of ~100 nm and pitch of 200 nm. The arrays give off a vibrant green color due to structural coloration. This green color signifies succesful fabrication of the arrays, because only the target geometry would produce this color. The color is *really* sensitive to the exact geomety of the arrays, so any other geometry (what we call a defect) will produce a different color. These defects can arise from a set of different root causes in the manufacturing process including contamination, non-optimized fabrication processes, faulty tools, etc. For the most part, the defects on this wafer fall into one of seven categories:
1. particle void
2. non-fill void
3. etch delay
4. edge etch delay
5. edge non-etch
6. edge non-fill
7. scratch

Each of these defects manifests on the wafer with a different appearance - with different colors, spatial orientation, location, etc. Understanding these defect types in context requires knowledge of the various nanofabrication processes that were involved in making the wafer. For the purposes of this repository, I mainly want to focus on the machine learning and computer vision aspect of this problem, and so I'll leave the context here, having simply stated that there are different unique types of defects which need to be classified.

## Methods:

### Pre-Processing:
  The code "Pre-Processing.py" is used to pre-process the raw RGB data to prepare it for the subsequent computer vision tasks. First, the RGB image is masked to isolate the 1x1 mm square device regions on the wafer using the image mask "mask_sqrs.png". The RGB device images are then color-indexed (or color quantized) to reduce the size of the color space to just a handful of colors (red, black, si (silicon-colored), green, faded green). These colors are chosen ad hoc based on the fact that they are observed to be the most popular colors on the wafer and different colors are associated with different defect types. As mentioned, when the pillars are fabricated succesfully, they produce a particular shade of green. Thus, the color green corresponds to the so-called "yield" condition for the devices and is defined specifically as HSV colors meeting the conditions 60<H<115, S>140, V>140. The device images are converted to the HSV color space for this operation. The color black is then defined as V<75 and si (silicon) as V>75 & S<50. The remaining pixels are then quantized to either red or faded green depending on which color centroid they are closet to (in terms of euclidean distance) where red centroid = (120,119,55) and faded green centroid = (50,90,50).
 
<p float="left">
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Images/mask_sqrs.png" width="30%" title="Squares Mask Image"/>
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Images/RGBi.png"  width="30%" title="Indexed RGB Image"/>
</p>

Each image of each device is then considered as an example in this problem. For defect classification, only devices containing a significant percentage of defectivity (pixels with a color other than green) are considered. The threshold for considering a device defective is arbitraily set at 10% and the Devices_Info dataframe is filtered for fraction of defectivity being greater than or equal to 10%. This creates a new DataFrame (Data\Devices_Dfct_Info) containing only the defective devices and their information (square row, square column, bounding box, and pixel coordinates) as shown in the image below.

  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/Devices_Dfct_Info screenshot.JPG"  width="500" title="Screenshot of Devices_Info DataFrame">

### Feature Engineering:
Various features are calculated for each defective device to be used as input features to the machine learning classifier. Features include: 
* the fraction of device that is defective
* the fractions of each device that are of each of one of the following indexed colors (black, Si, red, faded green)
* if the device is at an edge of the pattern area (a boolean feature 0/1)
* the fraction of the perimeter pixels of Si areas that touch each of the following colors (black, green, red, faded green)

This forms a new DataFrame (Data\Devices_Dfct_Features) as shown below.

  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/Devices_Dfct_Features screenshot.JPG"  width="800" title="Screenshot of Devices_Info DataFrame">

### Labelling the Training Data:
200 of the defective devices are randomly sampled forming Data\Devices_Training_Info and Data\Devices_Training_Features. Next the training examples must be labeled which is done with the script "labeller_updatable.py". This script (at least) works in the IDE Spyder by displaying the image of a device from the training set one at a time and asking for input as to what defects are present in the device. The input options are abbreviations I came up with for the various defect types ('p','nf','ed','eed','ene','enf','s'). I would enter a comma-separated list of these abbreviations for each device. The script allows for 'quit' to be entered as well, which allowed me to end the program and take a break from the rather time-consuming labeling process. It would save the current progress and I could then come back later and continue labelling. Hence, the *updatable* aspect of the labeller. The label lists are then automatically parsed and transformed into a sparse boolean array (one-hot encoding).
  
### Model Training and Evaluation:
The machine learning is done using the random forest classifier from the Scikit-Learn library. The classification is multi-output, meaning that each device on the wafer can have any combination of the 7 defect types. In fact, many of the devices are plagued by multiple types of defects, so the multi-output style classifier is most effective. Originally, I tried to distill the problem to a single-output problem, in which the most prominent defect type in each device would be the output, but this had undesirable results, including that it made the labelling process very subjective and difficult.

<img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/gray_train.jpg" width="30%" title="training dataset"/> 

The training and evaulation sets are split 67% and 33% (respectively) at random. The classifier is fit to the training data and then used to make predictions on the evaluation dataset. The accuracies of those predictions for each defect type are then scored in terms of precision and recall.

## Results and Discussion:
### Evaluation Scores:

|   Defect Type   | # of Training Examples | Precision | Recall |
|:--------------- |:------------------:    | :--------:| :----: |
| particle void   |     36                 | 100       | 62     |
| non-fill void   |     87                 | 81        | 79     |
| etch delay      |     72                 |  79       | 88     |
| edge etch delay |     87                 | 96        |96      |
| edge non etch   |      8                 | 100       |  100   |
| edge non-fill   |      1                 | NaN       |0       |
| scratch         |     44                 |  54       | 47     |

### Test Results Visualization:
The images below show the classification predictions made by the model for 3 of the 7 defect types for the whole wafer. The results are visualized by gray'ing out the entire image of the wafer except for the device regions in which that particular defect was detected, which are given their normal RGB values. 

(left-to-right: particle voids, edge edge delay, and non-fill voids) (hover to see title). 

<p float="left">
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/classification_image_p_predict.jpg" width="30%" title="devices with classified particle defects"/> 
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/classification_image_eed_predict.jpg" width="30%" title="devices with classified edge etch delay defects"/> 
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/classification_image_nf_predict.jpg"  width="30%" title="devices with classified non-fill defects"/>
</p>

## Future Work:
The primary goal of this project wasn't to make a perfect classifier, it was to teach myself machine learning. I think the best way to continue learning is to move on to a new project instead of grinding for better model performance. So, I'm likely going to leave this project where it currently stands. That being said, I have thought about what could be done to improve this classifier, and I would like to discuss that here.

The first thing I would do would be...

## References:
[1] ASME Paper.

[2] https://www.osapublishing.org/oe/abstract.cfm?uri=oe-26-23-30952
