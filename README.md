# Population-Public-Health-Research
This project developed an AI model (EfficientNetB0) to detect breast cancer in ultrasound images, achieving 95% accuracy with 97% specificity. The system outperforms radiologists and could enable accessible screening in low-resource areas. Future work focuses on clinical validation and implementation.


# Project Title
Exploring Deep Learning AI Ultrasound as a Primary Breast Cancer Screening Test: An Informatics Approach to Cancer Prevention

# Authors
•	Frederick Damptey
•	Benjamin Odoom Asomaning

# Project Overview
This project explores the potential of deep learning-based AI models to enhance breast cancer detection using ultrasound imaging, particularly in resource-constrained regions where access to mammography is limited. The study focuses on developing and evaluating a Convolutional Neural Network (CNN) model to classify breast ultrasound images into normal, benign, and malignant categories with high accuracy.

# Objectives
1.	Develop a deep learning model to classify breast ultrasound images accurately.
2.	Evaluate the model's performance against established benchmarks for mammography and human radiologists.
3.	Assess the feasibility of using AI-enhanced ultrasound as a primary screening tool in low-resource settings.

# Methodology
Dataset
•	Source: The dataset, Breast Ultrasound Images Dataset (Dataset_BUSI_with_GT), was sourced from Kaggle and consists of 780 ultrasound images labeled as normal, benign, or malignant. Labels were confirmed by radiologists and histopathology.
•	Secondary Dataset: Additional data from The Cancer Imaging Archive (TCIA) was used for validation.
Preprocessing
•	Images were resized, normalized, and augmented (rotation, flipping, zooming, etc.) to enhance model robustness.
•	The dataset was split into training and test sets using stratified K-Fold Cross-Validation (K=5) to handle class imbalance.
Model Architecture
•	Models Evaluated: ResNet50, EfficientNetB0, InceptionV3, and VGG16, pre-trained on ImageNet.
•	Fine-Tuning: The last two layers of each model were unfrozen for training, while the rest were kept frozen to leverage transfer learning.
•	Hyperparameters: Consistent across models (50 epochs, L1 regularization, dropout, learning rate).
Evaluation Metrics
•	Accuracy, AUC-ROC, Sensitivity (Recall), and Specificity were used to assess performance.
•	Results were compared against radiologist benchmarks and prior studies.

# Results
The EfficientNetB0 model achieved the highest performance:
•	Accuracy: 95%
•	AUC-ROC: 0.99
•	Sensitivity: 93%
•	Specificity: 97%
These results surpassed:
•	Radiologist benchmarks (80% sensitivity, 85% specificity).
•	Meta-analytic findings for USS AI systems (e.g., Li et al., 2024: AUC 0.732, sensitivity 0.93, specificity 0.90).
Comparative Performance Table
Model	Test Accuracy	AUC-ROC	Sensitivity	Specificity
ResNet50	0.92	0.964	0.881	0.945
EfficientNetB0	0.95	0.99	0.933	0.97
InceptionV3	0.941	0.982	0.934	0.962
VGG16	0.933	0.982	0.909	0.951

# Discussion
Key Findings
•	The model demonstrated superior performance in distinguishing malignant from non-malignant lesions, reducing false positives (high specificity).
•	AI-enhanced ultrasound could serve as a scalable, cost-effective screening tool in low-resource settings, addressing accessibility gaps.
Implications
•	Population Health: Early detection in underserved regions could reduce breast cancer mortality.
•	Clinical Workflow: Integration of AI could alleviate the burden on radiologists and improve diagnostic consistency.

# Limitations
1.	Dataset Size: Limited to 780 images; larger, diverse datasets are needed for broader validation.
2.	Computational Resources: Training required significant GPU power, limiting hyperparameter optimization.
3.	Time Constraints: Limited scope for exhaustive literature review.

# Future Work
1.	Expand the dataset and incorporate segmentation tasks.
2.	Implement explainable AI techniques for clinical transparency.
3.	Conduct meta-analyses to validate findings against recent studies.
4.	Advocate for policy changes to integrate AI ultrasound into screening guidelines.

# Personal Contributions
•	Frederick Damptey: Led model development, fine-tuning, and performance evaluation. Contributed to the manuscript and comparative analysis.
•	Benjamin Odoom Asomaning: Managed dataset preprocessing, augmentation, and validation. Assisted in literature review and results interpretation.

# Data Sources
•	Primary Dataset: Kaggle - Breast Ultrasound Images Dataset
•	Secondary Dataset: TCIA - Breast Lesions USG

# References
Key references are listed in the project document. Full citations are available in the manuscript.

