# Final-Project-Group2

# Table of contents
- Scope of project
- Overview
- Dataset description
- Running the code

# Scope of project
The rapid growth of remote sensing technologies using satellite images has helped in building surveillance and security systems for water bodies.
Maritime monitoring is essential for many government bodies to catch hold of any criminal activities happening in international waters. 
Many illegal activities like unlawful fishing, hijacking of ships, encroachment of sea borders, illicit exchange of sea cargo, accidents, and military attacks.
The traditional manual approach to ship classification is time-consuming, expensive, and error-prone, which limits its effectiveness in real-time applications.
Therefore, the development of an automated ship classification system using deep learning techniques has the potential to revolutionize the way we monitor and manage maritime activities.

# Overview
The link to download the dataset is here:- https://www.kaggle.com/datasets/gasgallo/masati-shipdetection. Refer the kaggle dataset download guide for more details.
Download the dataset first from the above link.
Once you have downloaded the dataset, clone this github repository and make sure you have the folders:-
1. Code - Python scripts
2. Excel - Dataset description and class distribution
3. Final-Group-Presentation - Powerpoint presentation
4. Final-Group-Project-Report - PDF Report
5. Group Proposal - PDF
6. Individual-Final-Project-Report

# Dataset description
Once you have downloaded the dataset ensure that you place it in a Data folder and the path for it is the same as that of all the folders mentioned above. 
This means that you should have 7 folders in total (including Data).
The Data folder should consist of these seven folders namely:-
1. coast
2. coast-ship
3. detail
4. land
5. multi
6. ship
7. water
This is how the sample images look like:-
![](/Images/dataset_images.png)
# Running the code
Once your project structure is ready, use the below command for running the script:-
<br>
> cd Code/
> 
> python3 train.py --model {model-name}
<br>
Replace model-name with "VGG16", "VGG19", "Inception", "Resnet", "Xception", "'CNNmodel'" to use the train script. The environment in which the code is running
should have the following packages installed:- <br>
1. tensorflow == 2.11.0 <br>
2. keras == 2.11.0 <br>
3. sklearn == 1.2.0 <br>
4. cv2 == 4.7.0 <br>
5. numpy == 1.24.1 <br>

