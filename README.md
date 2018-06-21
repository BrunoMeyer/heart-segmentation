# Left Ventricle segmentation
This project is an model to segmentation of endocardium and epicardium in medical images. The code include an pipeline that use many methods like circles identification by Hough transformation.


## Database
This project uses an public database that is avaliable in: http://laurentnajman.org/heart/H_data.html
This was build by the follow reference: 


NAJMAN, Laurent et al. An open, clinically validated database of 3d+ t cine-MR images of the left ventricle with associated manual and automated segmentations. Insight Journal, 2007.


The creators of database provides an binary file to extract the planes of the images (that were 3D). This file was be added to the project as `extractplane` and can be accessed in http://laurentnajman.org/heart/H_programs.html.

# Requirements
- python >= 3.65
  - opencv-python >= 3.4.0.12
  - numpy >= 1.14.2

# How to use
To download the database and run the algorithm, run the follow command:

```bash
./run.sh
```


The default metrics was the *expert1* especialist. To change it, change the `main.py` file.
