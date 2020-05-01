# nanopillar-computer-vision
## ONE-LINER:
This project centers around development of a computer vision approach for classification of defects in arrays of silicon nanopillars that have been fabricated on a silicon wafer.

## PROJECT BACKGROUND:
During my time at UT Austin as a PhD student, I researched metrology solutions for nano-manufacturing. My work centered around image characterization of what we called "large area nanostructure arrays" which includes things like nanopillar arrays, grating structures, and mesh structures which have been fabricated over large areas such as silicon wafers, glass sheets, roll-to-roll webs, etc. These kinds of structure have applications in displays, memory storage devices, electronics, etc. 

Silicon nanopillar arrays were of particular interest to me, and much of my work was centered around them. One of my primary focuses, was development of computer vision algorithms for detecting and classifying defects in these arrays. I actually published a paper on this work [1] which shows how traditional image processing approaches can be used. Post-graduation, I wanted to re-explore this problem in the context of machine learning based computer vision, and thus this project was born.

### MORE CONTEXT:
A high resolution RGB image of a wafer was recorded using an imaging system. The wafer, which is 100 mm in diameter, contains nearly 2000 1x1 mm square arrays of silicon nanopillars. Each nanopillar has a diameter of ~100 nm and 200 nm pitch. The arrays give off a vibrant green color due to a phenomenon called structural coloration (see [2] for a similar account). This green color signifies succesful fabrication of the arrays, and any deviation from this green color signifies some sort of fabrication error or defect. These defects can arise from a set of different root causes. In manufacturing, it is important to be able to detect defects and also classify them so that specific problems in the manufacturing facility can be addressed. In high volume manufacturing, this classification process would need to be done in an automated fashion, and thus computer vision methods are necessary.

## METHOD:

### PRE_PROCESSING:
  The RGB image is converted to a color-index image to reduce the size of the color space to just a handful of colors (red, black, gray, green, black, and faded green). These colors are by far the most popular colors on the wafer and are useful for the subsequent feature engineering.  

### FEATURE ENGINEERING:
Most of the effort that went into this project went into feature engineering to provide relevant features to the classifier. Features include: 
  -fraction of device that is defective
  -fractions of device that are of each of the following colors (red, black, gray, green, faded green)
  -if the device is at an edge of the pattern area (a boolean feature 0/1)
  -the fraction of the perimeter pixels that touch each of the following colors (black, gaded green, green)
  
### MACHINE LEARNING:
This approach utilizes a random forest classifier from the Scikit Learn Library. The classification is multi-output, meaning that each device on the wafer can have any number of defect types including: particle, non-fill, etch delay, edge etch delay, edge non-etch, edge non-fill, or scratch. Understanding these defect types requires knowledge of various nanofabrication processes that were involved in making the wafer. You most likely don't have this knowledge, so just understand that they are different types of defects which we are trying to classify.

## RESULTS:

![test image](https://github.com/gaw1ik/nanopillar-computer-vision/blob/master/classification_image_p_predict.jpg)

## OUTSTANDING ISSUES:
1. Certain types of defects were either over- or under-represented in the training dataset, causing clear biases in the classification. Edge non-fill, for instance, has only a few instances on the entire wafer and happened to not be randomly sampled for the training dataset and therefore is not represented at all. Manually, devices impacted by edge non-fill can be forced to be a part of the training dataset, although the training will still suffer from having so few instances. On the other hand, edge etch delay seems over-represented. The training set could be forced to have a uniform amount of training examples of each type.
2. 

## REFERENCES:
[1] ASME Paper.
[2] https://www.osapublishing.org/oe/abstract.cfm?uri=oe-26-23-30952
