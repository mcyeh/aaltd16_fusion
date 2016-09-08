# AALTD'16 Submission

by [Michael Yeh](http://www.cs.ucr.edu/~myeh003/) and [Eamonn Keogh](http://www.cs.ucr.edu/~eamonn/)

This package contains the code necessary for reproducing our submission for the "bofSC + randShape" method which won the [AALTD'16 Challenge](https://aaltd16.irisa.fr/challenge/).

The submitted files are included in the folder 'output\_submit', and the file names are 'dn1024\_\_sl25\_\_sn150\_\_task1.txt' for task 1 and 'dn1024\_\_sl25\_\_sn150\_\_task2.txt' for task 2.

Two 3rd party toolbox are used in this package, and they are [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) and [SPAMS](http://spams-devel.gforge.inria.fr/). Both toolbox's source code (the version used in this package) is included in the folder '\_toolbox\_source'. Please refer to the toolbox's corresponding websites for the most updated version.

## 1. System Requirement
The code has only been tested on Windows 64-bit. The code does not work on other operating system as the provided MEX files for the 3rd party toolbox are compiled specifically for 64-bit Windows. Please refer to the website of the 3rd party toolbox for instruction on compiling MEX files for your operating system (see [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) and [SPAMS](http://spams-devel.gforge.inria.fr/)).

## 2. Train and Test with AALTD'16 Dataset
Please follow the steps below to train and test using the AALTD'16 Challenge dataset. Note, the result text files could be slightly different from the submitted result. To reproduce the result exactly, please follow the instruction in Section 3.

1. Download 'train.h5', 'test\_task1.h5', and 'test\_task2.h5' from the website of [AALTD'16](https://aaltd16.irisa.fr/challenge/).
2. Move the downloaded files into the folder 'dataset' in this directory.
3. In MATLAB, set the current working directory to the folder contains this file.
4. In MATLAB, use the command 'RunSystem' to execute the 'RunSystem.m' script.

## 3. Reproduce the Submitted Result
Because part of the training system uses randomized algorithm, the predicted labels obtain from each independently trained models could differ slightly. For verification purposes, the model that used to generated the submitted outputs is included in this package. Please follow the steps listed below to reproduce the submitted text files.

1. Download 'train.h5', 'test\_task1.h5', and 'test\_task2.h5' from the website of [AALTD'16](https://aaltd16.irisa.fr/challenge/).
2. Move the downloaded files into the folder 'dataset' in this directory.
3. From the folder 'output\_submit', move the file 'dn1024\_\_sl25\_\_sn150.mat' to the folder 'output'
4. In MATLAB, set the current working directory to the folder contains this file.
5. In MATLAB, use the command 'RunSystem' to execute the 'RunSystem.m' script.