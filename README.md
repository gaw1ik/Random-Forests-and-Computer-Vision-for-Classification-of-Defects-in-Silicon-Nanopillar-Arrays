# Random Forests and Image Processing for Classification of Defects in Silicon Nanopillar Arrays
## Description:
This project centers around the development of a method for classifying defects on a silicon wafer containing nanopillar arrays using random forests and image processing techniques.

## Project Background:
During my time at UT Austin as a PhD student, I researched metrology solutions for nano-manufacturing. My work centered around using spectral imaging system to charactize devices that we called "large area nanostructure arrays" which includes things like nanopillar arrays, grating structures, and mesh structures which have been fabricated on flat large area substrates such as silicon wafers, glass sheets, roll-to-roll webs, etc. People are interested in making these kinds of structures because they have many important applications including ones in displays, memory storage devices, electronics, etc. In order to manufacture these kinds of devices succesfully, metrology systems (systems that can measure and characterize these devices) need to be in place to measure their quality as they are being made. It's important to have metrology so that defects can be detected as they appear in the devices. Furthermore, the defects can to be *classified* so that the manufacturing facility knows specifically what went wrong and people can hopefully fix things so that devices can continue to be made correctly. 

Silicon nanopillar arrays were of particular interest to me, and much of my work was centered around them. They have many interesting applications, although what really got me hooked on them initially was the fact that they create these really amazing colors (see images below) due to a phenomenon called structural coloration (see my publication [reference 1] for more info). The colors arise because of the nanoscale size and shape of the pillars, and subtle changes in the geometry - changes on the order of a few nanometers, for instance - can completely change the color that they exhibit. As it turns out, this is an extremely useful characteristic from a metrology perspective, because characterizing this color can give insight into what's happening on the nanoscale, which otherwise can't be seen without the use of tools like electron micrscopes. Electron microscopes are extremely slow, and so being able to do an optical characterization that can image a full wafer *in seconds* is highly preferred. 

<img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/Wafers.jfif" width="500" title="Images of Silicon Nanopillar Array Wafers">

One of my primary focuses of my PhD was developing computer vision algorithms for detecting and classifying defects in these Si nanopillar arrays. I've actually submitted a paper on this work which shows how more-traditional image processing approaches can be used. This work was fairly rudimentary from a computer vision perspective, and wouldn't have been accepted to a computer vision journal, but in the context of metrology for silicon nanopillar manufacturing, the work offers insight at the interface of computer vision and the manfuacturing itself and demonstrates beginnings for more advanced algorithms. After I graduated, I wanted to re-explore this problem in the context of machine learning based computer vision, and thus this project was born.

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

### Quick note on the structuring of the repo:
The main folder contains the code files, the saved classifier model, and four sub-folders: "Data\", "Figures\", "Images\", and "Raw_Data\". "Data\" contains the various DataFrames, "Figures\" contains figures that were created specifically for use in the README, and "Images\" contains images that were created as part of the image processing. The "Raw_Data\" folder is meant to containg the raw dataset (a large RGB bitmap image) which is not included here, partly because the raw data itself was too large to be uplaoded. Nonetheless, I've included the folder to dictate the structure as the Pre-Processing code calls for "Raw_Data\RGB.bmp".

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

## Results and Discussion

### Evaluation Scores:

|   Defect Type   | # of Training Examples | Precision | Recall |
|:--------------- |:------------------:    | :--------:| :----: |
| edge non etch   |      8                 | 100       |  100   |
| edge etch delay |     87                 | 96        |96      |
| particle void   |     36                 | 100       | 62     |
| non-fill void   |     87                 | 81        | 79     |
| etch delay      |     72                 |  79       | 88     |
| scratch         |     44                 |  54       | 47     |
| edge non-fill   |      1                 | NaN       |0       |

### Discussion of Performance:
#### Edge Non Etch
The approach handles a few of the defect categories fairly well. Scores in precision and recall for edge non-etch are perfect. This is likely to instill an overly optimistic level of performance. Edge non etch is so common in this particular wafer, that the edge square feature is probably over-predictive for this type of defect. Basically, if there is a defect in an edge square, the algorithm is very likely to classify it as a edge non etch, which happens to be correct very often in this wafer, but wouldn't necessarily be the case in other wafers or even in parts of this wafer that weren't included in the training data. 

This can be helped by adding additional features that check for the local spatial location and orientation of the defect, for instance, to make sure that the defect actually exists on the outer edge of the edge square before it is classified as an edge defect. This feature was actually included in my rule-based model which I discuss in my publication (currently in review), but I did not have time to code that into this project. 

#### Edge Etch Delay
The story is fairly similar with predictions of edge etch delay. The model technically performs very well on this sample, but I am aware of potential failure modes similar to the ones for edge non etch if the model were to be applied to future samples.

#### Particle Voids
Particle void classification scores perfectly on precision, meaning that of the devices that had been classified as having particle void defects, that prediction was correct 100% of the time. However, recall for particle voids is of sub-par performance, meaning that there were devices affected by particle defects that were not identified as being positive for this type of defect. In other words: When flags were raised, the flags were accurate, but many flags weren't raised where they should have been.

#### Non-fill Voids and Etch Delay
Performance for non-fill voids is decent, but not great. Similar is the case for etch delay.

#### Scratches
Performance for scratches is generally quite poor. This suggests that features offering predictive capabilities for scratches were not present.

#### Edge Non-Fills
Performance for edge non-fills was abysmal. This is most likely to be due to the fact that there was only 1 training example for this type of defect. The model simply was not given enough examples of this defect to form a sense of it. Precision is infinitely low, meaning that there were no positive identifications whatsoever, and recall is 0, meaning that there was at least one false negative. 

The first thing I would do improve the performance here would be to increase the number of training examples. Training examples were selected purely at random, and so it makes sense that such a rare defect type (only ~10 devices on the wafer have this defect) would get underrepresented. Some manual forcing of devices containing this defect would need to be fed into the training dataset, although one would need to be careful not to include too many of the few that exist, because then the model performance could not be trustably evaulated.

### Test Results Visualization:
The images below show the classification predictions made by the model for 6 of the 7 defect types *for the whole wafer* (not just the training/eval set). Note that edge non-fill classification is not shown due to its low performance. The results are visualized by gray'ing out the entire image of the wafer except for the device regions in which that particular defect was detected, which are given their normal RGB values. 

(hover to see titles). 

<p float="left">
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/classification_image_ene_predict.jpg" width="30%" title="devices with classified edge non etch defects"/> 
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/classification_image_eed_predict.jpg" width="30%" title="devices with classified edge etch delay defects"/> 
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/classification_image_p_predict.jpg"  width="30%" title="devices with classified particle defects"/>
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/classification_image_nf_predict.jpg"  width="30%" title="devices with classified non-fill defects"/>
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/classification_image_ed_predict.jpg"  width="30%" title="devices with classified etch delay defects"/>
  <img src="https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/Figures/classification_image_s_predict.jpg"  width="30%" title="devices with classified scratch defects"/>
</p>

## Conclusions and Future Work:
The primary goal of this project wasn't to make a perfect classifier. Instead, I simply wanted to build a machine learning pipeline to attack this old problem from my research days. I kept the features and approach to training fairly simple. Nonetheless, the model shows a lot of potential, and for certain defect types demonstrates a high degree of predictive power. 

Additional features could be added and the training set could be massaged a lot more to represent rarer defect types (like edge non-fills) to make the performance of this model better. However, I think the best way to continue learning is to move on to a new project instead of grinding for better model performance. So, I'm likely going to leave this project where it currently stands.

## References:
[1] https://www.osapublishing.org/oe/abstract.cfm?uri=oe-26-23-30952
