# WristBased-EE-Estimation
Codebase for paper 'Developing and comparing a new BMI inclusive energy expenditure algorithm on wrist worn wearables'.

### Content
- [wrist_preprocessing.py][1]: preprocessing raw wrist-based sensor data (e.g., accelerometer, gyroscope) collected from wearable devices.
- [in-lab.py][2]: processes and analyzes data collected in a controlled lab environment (training and inferencing).
- [in_wild.py][3]: processes and analyzes data collected in free-living conditions (inferencing only).
- [Result Visualization.ipynb][4]: Jupyter Notebook for visualizing and analyzing the results of the energy expenditure estimation (demographics, Bland-Altman plots).
- [helper_preprocess.py][5]: helper script containing utility functions for preprocessing sensor data.
- [helper_extraction.py][6]: helper script for feature extraction from preprocessed sensor data.
- [helper_model.py][7]: helper script for training, evaluating, and using predefined machine learning models.
- [helper_visualization.py][8]: helper script for generating visualizations.
- [sort_resample_organize.py][9]: helper script for organizing, sorting, and resampling raw sensor data into a structured format.

[1]: https://github.com/HAbitsLab/WristBased-EE-Estimation/blob/main/helper_preprocess.py
[2]: https://github.com/HAbitsLab/WristBased-EE-Estimation/blob/main/in-lab.py
[3]: https://github.com/HAbitsLab/WristBased-EE-Estimation/blob/main/in_wild.py
[4]: https://github.com/HAbitsLab/WristBased-EE-Estimation/blob/main/in_wild.py
[4]: https://github.com/HAbitsLab/WristBased-EE-Estimation/blob/main/Result%20Visualization.ipynb
[5]: https://github.com/HAbitsLab/WristBased-EE-Estimation/blob/main/helper_preprocess.py
[6]: https://github.com/HAbitsLab/WristBased-EE-Estimation/blob/main/helper_extraction.py
[7]: https://github.com/HAbitsLab/WristBased-EE-Estimation/blob/main/helper_model.py]
[8]: https://github.com/HAbitsLab/WristBased-EE-Estimation/blob/main/helper_visualization.py
[9]: https://github.com/HAbitsLab/WristBased-EE-Estimation/blob/main/sort_resample_organize.py

# Web API to compare Calorie Estimates between a Smartwatch and Actigraphy
An open-access tool developed to estimate energy expenditure from wrist-worn wearable sensor data. Using accelerometer and gyroscope signals from commercial smartwatches, this calculator applies a machine learning model to generate minute-by-minute metabolic equivalent of task (MET) estimates. 

Link: https://wristmetcalculator.fsm.northwestern.edu/

Sample data is provide in this repo: [/data](data)
