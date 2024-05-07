# Ovarian Cancer Subtype Classification
A final project for CSCI 567. <br>
Data from https://www.kaggle.com/competitions/UBC-OCEAN/overview <br>
<br>

## Lunit
* All codes needed for training Lunit baseline model is in lunit/train_base_lunit.py <br>
* All codes needed for training Lunit CLIP model is in lunit/train_clip_lunit.py <br>
* All codes needed for testing Lunit baseline model is in lunit/Test_base_lunit.py <br>
* All codes needed for testing Lunit CLIP model is in lunit/Test_clip_lunit.py <br>

## Phikon
* All codes needed for training and testing Phikon baseline model is in Phikon_based/train_base_model.ipynb <br>
* All codes needed for training and testing Phikon-based CLIP model is in Phikon_based/train_clip_model.ipynb <br>

## CTransPath
* All codes needed for training CTransPath baseline model is in TransPath/cTransPath_model_train.py <br>
* All codes needed for training CTransPath CLIP model is in TransPath/clip_train.py <br>
* All codes needed for testing CTransPath baseline model is in TransPath/cTransPath_test.py <br>
* All codes needed for testing CTransPath CLIP model is in TransPath/clip_test.py <br>

## Other files
* patchGenerate.py is for naive preprocessing <br>
* patch_mask.py is for Otsu's thresholding matter detection preprocessing (This file might need to be rerun multiple times to get 100 patches for all images) <br>
* text_similarity.ipynb contains codes for analyzing text similarities <br>


