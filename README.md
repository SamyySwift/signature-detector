# Custom Signature Detector

### **🎯Overview:**

This project aims to create a custom object detection model to recognize signatures within the global fund pdf documents for the phase 2 of the PDF Redaction Tool Service.Within the Tensorflow folder contains the following sub-folders.

> **Tensorflow:**

  - `models` - This folder in used to clone Tensorflow's model gerden.
  - `inference-results` - This folders holds the ouput of the model's predictions
  - `scripts` - This folder holds all the python scripts.
  - `workspace` - This folder holds sub folders needed for training custom object detection. Within the workspcae contains the training_demo sub directory.
  
    > **training_demo:** Within the training_demo contains the following sub directories.
      
      - `annotations:` This folder will be used to store all *.csv files and the respective TensorFlow *.record files, which contain the list of annotations for our      dataset images.

       - `exported-models:` This folder will be used to store exported versions of our trained model(s).

       - `images:` This folder contains a copy of all the images in our dataset, as well as the respective *.xml files produced for each one, once labelImg is used to annotate objects.

      - `images/train:` This folder contains a copy of all images, and the respective *.xml files, which will be used to train our model.

      - `images/test:` This folder contains a copy of all images, and the respective *.xml files, which will be used to test our model.

      - `models:` This folder will contain a sub-folder for each of training job. Each subfolder will contain the training pipeline configuration file *.config, as well as all files generated during the training and evaluation of our model.

      - `pre-trained-models:` This folder will contain the downloaded pre-trained models, which shall be used as a starting checkpoint for our training jobs.
  
  ### **🛠️ Tools and Libraries:**
  - Tensorflow
  - Tensorflow's object detection API
  - Python
  - Google Colab
  - Pandas
  - [img2pdf](https://pypi.org/project/img2pdf/)
  - [pdf2image](https://pypi.org/project/pdf2image/)
  - OpenCv
  - LabelImg
  - Numpy
  
  To install the above requirements successfully, follow this [Tensorflow's documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)

### **⚙️Training Custom Detection Model:**
In other to train or retrain the model, run the notebook, `signature_detector.ipynb` in colab (Recommended) or your local machine.

### **⚡Inferencing and Redacting:**
To make detections and redact a pdf file, execute the following
- Download the exported-model folder from within `Tensorflow\workspace\training-demo\exported-model`
- Create a virtual environment
- Run `pip install -r requirements.txt` to install the dependencies for the model.
- Run `python -m spacy download en_core_web_sm` to install the NER model.
- Run `python redact.py [PATH_TO_PDF]` setting the following entry point accordingly.

  - **Pdf_path:** Path to your pdf

  
