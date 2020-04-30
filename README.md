# nanopillar-computer-vision
ONE-LINER:
This project centers around development of a computer vision approach for classification of defects in arrays of silicon nanopillars that have been fabricated on a silicon wafer.

PROJECT BACKGROUND:
During my time at UT Austin as a PhD student, I researched metrology solutions for nano-manufacturing. My work centered around image characterization of what we called "large area nanostructure arrays" which includes things like nanopillar arrays, grating structures, and mesh structures which have been fabricated over large areas such as silicon wafers, glass sheets, roll-to-roll webs, etc. These kinds of structure have applications in displays, memory storage devices, electronics, etc. 

Silicon nanopillar arrays were of particular interest to me, and much of my work was centered around them. One of my primary focuses, was development of computer vision algorithms for detecting and classifying defects in these arrays. I actually published a paper on this work [1] which shows how traditional image processing approaches can be used. Post-graduation, I wanted to re-explore this problem in the context of machine learning based computer vision, and thus this project was born.

MORE CONTEXT:
A high resolution RGB image of a wafer that I constructed early on in my research was recorded using an imaging system. The wafer, which is 100 mm in diameter, contains nearly 2000 1x1 mm square arrays of silicon nanopillars. Each nanopillar has a diameter of ~100 nm and 200 nm pitch. The arrays give off a vibrant green color due to a phenomenon called structural coloration (see [2] for a similar account). This green color signifies succesful fabrication of the arrays, and any deviation from this green color signifies some sort of fabrication error or defect. These defects can arise from a set of different root causes. In manufacturing, it is important to be able to detect defects and also classify them so that specific problems in the manufacturing facility can be addressed. In high volume manufacturing, this classification process would need to be done in an automated fashion. Thus, this project sets out to develop a computer vision approach for doing this automated defect classification.

METHOD:
This approach utilizes a random forest classifier from the Scikit Learn Library. Most of the effort that went into this project went into feature engineering to provide relevant features to the classifier. 

REFERENCES:
[1] ASME Paper.
[2] https://www.osapublishing.org/oe/abstract.cfm?uri=oe-26-23-30952
