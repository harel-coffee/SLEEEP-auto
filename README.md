*code, data and documentation behind the publication:*

# Robustness Of Radiomics To Variations In Segmentation Methods In Multimodal Brain MRI

*By Maarten Gijsbert Poirot, Matthan Caan, Henricus Gerardus Ruhe, Atle Bjørnerud, Inge Groote, Liesbeth Reneman and Henk Marquering.*

Published: 06 October 2022 at https://www.nature.com/articles/s41598-022-20703-9 

[RiS citation](./cite.ris)

## Abstract

**Background:** Radiomics in neuroimaging uses fully automatic segmentation to delineate the anatomical areas for which radiomic features are computed. However, differences among these segmentation methods affect radiomic features to an unknown extent.

**Method:** A scan-rescan dataset(n=46) of T1-weighted and diffusion tensor images was used. Subjects were split into a sleep-deprivation and a control group. Scans were segmented using four segmentation methods from which radiomic features were computed. First, we measured segmentation agreement using the Dice-coefficient. Second, robustness and reproducibility of radiomic features were measured using the intraclass correlation coefficient (ICC). Last, difference in predictive power was assessed using the Friedman-test on performance in a radiomics-based sleep deprivation classification application.

**Results:** Segmentation agreement was generally high (interquartile range=0.77-0.90) and median feature robustness to segmentation method variation was higher (ICC>0.7) than scan-rescan reproducibility (ICC 0.3-0.8). However, classification performance differed significantly among segmentation methods (p<0.001) ranging from 77% to 84%. Accuracy was higher for more recent deep learning-based segmentation methods.

**Conclusion:** Despite high agreement among segmentation methods, subtle differences significantly affected radiomic features and their predictive power. Consequently, the effect of differences in segmentation methods should be taken into account when designing and evaluating radiomics-based research methods.

![brains](images/brains.jpg)

## This repository

This repository contains three things:

1. In the `/data` directory you will find the data used to train our models to classify sleep deprived subjects from normal sleep-wake-cycle subjects. Howver, due to data limitations of GitHub repositories, the largest portion of data has been omitted and can be accessed upon request. These files have been replaced by `.gitkeep` files. These should provide those interested in this repository insight into the structuring and of data available.
2. In the `/scripts` directory you will find all scripts used in (1) preprocessing T1 and DTI MRI scans, (2) software used for segmentation, (3, 4) standardizing the directory structure, (5) analyzing the resulting segmentations, (6) calculating radiomic features, concatenating radiomic features into dataframes, feature selection and finally (7) modeling. Since this is a largely chronological process every step has been numbered and put into their own directory
3. Lastly, you will also find the `README *.pptx` to guide you through the use of code in combination with the data available. We will now quickly glance over each of these steps and the corresponding data.

Here is a nice picture from the documentation slide deck. Please get in touch if you have any questions.

Best,



Maarten Poirot

![network](images/shallowNN.jpg)



