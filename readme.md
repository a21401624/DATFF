# Dual Attention Transformer Fusion (DATFF) Module 

This is the code for our paper 'Dual Attention Feature Fusion for Visible-Infrared Object Detection' that has been accepted by ICANN2023.

Note that we only include the code for DroneVehicle dataset. The config file of FLIR dataset will be released soon.

![DATFF](DATFF.png)

## About the code

Our code is build on mmrotate==0.3.4. We only present the modifications we have made to the mmrotate codebase. So you should put the python files under right locations and
register them in '\_\_init\_\_.py'.

## About the dataset

You can find the annotations(Folders: trainlabelrtxt, vallabelrtxt, testlabelrtxt) and image lists(train.txt, val.txt, test.txt) from this Google Disk link: 
https://drive.google.com/drive/folders/1ZXOgmTk5_Tvz464vzclHJyZ1tv41GEYu?usp=drive_link

You can find the images(Folders: trainimgr, trainimg, valimgr, valimg, testimgr, testimg) from the DroneVehicle dataset Github page(https://github.com/VisDrone/DroneVehicle).