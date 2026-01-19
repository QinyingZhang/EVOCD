# An Explainable Urban Flooding Severity Assessment Method Based on Vehicle Orientation and Component Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14581613.svg)](https://doi.org/10.5281/zenodo.14581613)

## Abstract
Decision-making processes in urban flooding assessment models often lack transparency, thereby limiting their credibility in critical early warning scenarios. To address this challenge, we propose an explainable visual assessment framework based on conditional hierarchical perception. Building upon refined flooding severity standards and multi-view datasets, the framework decomposes the assessment task into three cascading, traceable subtasks: vehicle orientation classification, orientation-condition-guided key component detection, and flooding severity assessment through physical mapping. Orientation classification provides structured prior information that dynamically activates corresponding component detectors for each perspective, thus enhancing cross-view robustness. An explicit physical mapping between component submersion states and flooding severity levels enables the visualization of the decision-making process. Furthermore, to address the scarcity of extreme-level samples, a collaborative strategy combining focal loss and inverse-frequency class weights is introduced to effectively mitigate the impact of class imbalance. Experiments demonstrate the robust performance of the front-end modules (orientation classification accuracy: 92.31%; component detection mAP@0.5: 96.73%). The final assessment model achieves an accuracy of 77.83% and a macro F1-score of 74.25% on the test set, while maintaining high recall for high-severity levels despite data imbalance. This framework strikes a favorable balance among performance, robustness, and explainability, thereby providing novel insights for refined urban flooding monitoring and the development of reliable early warning systems.

## Project Structure

- `stage1_ori_cla`: Vehicle orientation classification
- `stage2_com_loc`: Key component detection
- `stage3_flood_assess`: Flooding severity assessment
- `All_ablation_experiments`: Ablation studies

## Installation & Environment
- **OS**: Windows 11
- **Python**: 3.8+
- **PyTorch**: 2.5.1
- **CUDA**: 12.1
- **Other**: Ultralytics YOLOv11 codebase, matplotlib, seaborn, opencv-python, scikit-learn, numpy

```bash
pip install torch==2.5.1 matplotlib seaborn opencv-python scikit-learn numpy
# And Ultralytics YOLOv11 requirements as needed
```

## Usage
- Each module contains independent scripts for training, inference, and evaluation.
- Ablation experiments are under `All_ablation_experiments`.
- Dataset and weights are not publicly available; please configure your own paths as needed.

## Results
- Orientation classification accuracy: **92.31%**
- Component detection mAP@0.5: **96.73%**
- Severity assessment accuracy: **77.83%**
- Macro F1-score: **74.25%**
- High recall for severe levels despite data imbalance

## Code and Data Availability

The code is publicly available in this repository. The dataset and pre-trained weights are not publicly available due to privacy and licensing constraints. Requests for access can be directed to the corresponding author.

## License
MIT
