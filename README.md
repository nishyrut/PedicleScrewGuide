[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Educational%20%2F%20Experimental-orange.svg)]()

## Introduction
An End-to-End pipeline that predicts optimal screw trajectory from CT scans, converting these predictions to patient specific 3D-printed anatomical guides, enhancing surgical precision compared to freehand techniques, while also minimizing radiation exposure.

> [!WARNING] 
> This script is not intended for clinical usage and should only be used for educational purposes.

> [!NOTE]
> This script has not been tested widely and the testing is an ongoing process. The base algorithm and the JIG mold may vary in the future updates.

## Outputs
**Predicted Trajectory**

https://github.com/user-attachments/assets/e371713f-5787-4e9b-adeb-a91ab70f62c9

**Generated JIG**

https://github.com/user-attachments/assets/3a239a59-43b5-4698-8a2e-7ac0ffc464a0

**JIG STL File**

<img width="536" height="364" alt="jig_stl" src="https://github.com/user-attachments/assets/a4e3d2ac-e91d-4c74-b623-4d6fd7aa29b3" />

## Installation
* This script requires python, along with the modules listed in requirements.txt
* Install python for your OS and ensure python and pip is in PATH env
* Install necessary python modules using the command ```pip install -r requirements.txt```

## Features
* Finds the optimal trajectory for screw placement (See cautions below for more information)
* Avoids Superior Articular Process on Lumbar vertebrae
* Generates a JIG so that it avoids the central ligament in spoinous process, at the same time, reducing the material
* Parameters of JIG and Screw can be modified
* Automatic conversion to STL file, making it 3D printable
* End-to-End pipeline with no manual intervention. Point to the CT scan with necessary inputs, and the printable output is generated.

## Usage
* The script should be run from the root directory (the directory where the script resides)
* It can be run using the command
 ``` python main.py <ct_scan> <vertebrae_name> <screw_length> <screw_dia> <inner_dia_jig> <outer_dia_jig>```
   |parameters|Definition|
   |---|---|
   |<ct_scan>|The CT scan|
   |<vertebrae_name>|The vertebrae for which the screw should be inserted. Each vertebrae should be separated by "+" (Refer the example below)|
   |<screw_length>|The length of the screw in mm|
   |<screw_dia>|The diameter of the screw in mm|
   |<inner_dia_jig>|The inner diameter of the jig in mm (Refer the attached JIG image)|
   |<outer_dia_jig>|The outer diameter of the jig in mm (Refer the attached JIG image)|
* Ex: ```python main.py ct_scan_23.nii.gz vertebrae_T12+vertebrae_L1+vertebrae_L2 30 6 1.5 6```
* This example reads the CT scan file, attempts to find the optimal screw placement with screw length and diameter of 30mm x 6mm, and attempts to create a 3D printable jig where the thickness of the mold will be 6 mm, and the path along the trajectory will be 1.5 mm for vertebrae T12, L1, L2.
* The outputs are saved to ***data/downloads/generated_vertebraes.zip***

## System Requirements
If your system meets the system requirements of [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), it can run this script.

## Cautions:
* **As stated above, This script is not intended for clinical usage. Use this only for educational purpose**
* This script may not generate JIG for some vertebrae, in this case, it usually means the length of the screw far exceeds any seating position inside vertebrae. Try to reduce the screw length of the vertebrae and run the script again
* The shape of the JIG and Screw trajectory prediction algorithm is under development and may go through heavy modification
* The current algorithm, which effectively finds the trajectory that is away from vertebral walls, only takes the shape of the vertebrae into account.
 * It does not detect and avoid fracture (TODO)
 * It does not differentiate medial and lateral wall in vertebral foramen (TODO)

## Acknowledgments & Citations
This pipeline relies on **TotalSegmentator** for automated CT image segmentation and **SimpleITK** for CT image manipulation. If you use this repository for research or academic work, please ensure you cite the original papers:
* **TotalSegmentator**
 > Wasserthal, J., Breit, H.-C., Meyer, M.T., Pradella, M., Hinck, D., Sauter, A.W., Heye, T., Boll, D., Cyriac, J., Yang, S., Bach, M., Segeroth, M., 2023. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence. https://doi.org/10.1148/ryai.230024
* **nnU-Net**
 > Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
Nature Methods, 18(2), 203-211.
* **ITK / SimpleITK**
 > R. Beare, B. C. Lowekamp, Z. Yaniv, "Image Segmentation, Registration and Characterization in R with SimpleITK", J Stat Software, 86(8), https://doi.org/10.18637/jss.v086.i08, 2018.
