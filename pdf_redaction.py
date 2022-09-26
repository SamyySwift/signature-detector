import glob
import os

import cv2
import img2pdf
import numpy as np
from pdf2image import convert_from_path

import tensorflow as tf


class PdfRedaction:
    def __init__(self, saved_model, name_model, date_model, reader) -> None:
        self.saved_model = saved_model
        self.name_model = name_model
        self.date_model = date_model
        self.reader = reader
        

    def pdf_to_image(self, pdf_path):
        """
        This function converts a pdf file into images.

        Args: pdf_path - path to pdf files
        
        Returns: None
        """
        #get path to pdf
        pdf_files = glob.glob(f'{pdf_path}/*.pdf')

        #Check if path exits already, else create it
        if os.path.exists('pdf_to_images'):
            print('Folder already exits!')
        else:
            print("creating path...")
            os.makedirs('pdf_to_images')
        output_path = 'pdf_to_images'

        #Loop through all pdfs and covert to images
        for img in pdf_files:
            print(f'Converting {img} to image...')
            images = convert_from_path(img)
            for i in range(len(images)):
                #   Save pages as images in the pdf
                images[i].save(f'{output_path}/{os.path.basename(img)}_{i}.jpg', 'JPEG')

        print('Done converting!')


    def redact_name_and_date(self, image):
        """
        Function that redacts date and names from images
        """

        print("Using OCR to convert image to text...")

        # detect texts and return bounding boxes
        bbox = self.reader.readtext(np.array(image))
        #Loop through each bounding boxes and extract min and max cordinates
        for bound in bbox:
            xmin, ymin = [min(id) for id in zip(*bound[0])]
            xmax, ymax = [max(id) for id in zip(*bound[0])]

            # Easyocr returns a list of tuples
            # Each tuples contains the cordinates, detected text and detection confidence
            # where the detected text is at index 1
            # So we pass the detected text to both models respectively 
            date_entity = self.date_model(bound[1])
            name_entity = self.name_model(bound[1])
            try:
                #check if current text is a name entity
                if (name_entity[0]['entity']=="B-PER") or (name_entity[0]['entity']=="I-PER"):
                    print("Redacting Name")
                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255,255,255), -1)
            
            except IndexError as e:
                pass
            
            #check if current text is a date entity
            for entity in date_entity.ents:
                if (entity.label_ == "CARDINAL") or (entity.label_ == "DATE"):
                    print("Redacting Date")
                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255,255,255), -1)
             
        return image

    def redact_signature(self, img, detection_threshold=.35):
        """
        Function to redact signatures
        
        """
        #Read the image using opencv
        image = cv2.imread(img)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_rgb)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        print('Detecting...')
        detections = self.saved_model(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_with_detections = image.copy()

        detection_threshold = detection_threshold #confidence of detection
        
        # extract image height and width
        height, width = image.shape[0], image.shape[1]
    
        box_index = 0

        # Loop through all the detections
        for detection in detections['detection_boxes']:
                ymin = int(detection[0]*height)
                xmin = int(detection[1]*width)
                ymax = int(detection[2]*height)
                xmax = int(detection[3]*width)
                #Filter out detections that are above the threshold value
                if detections['detection_scores'][box_index] > detection_threshold:
                    print('Redacting image with signature...')
                    cv2.rectangle(image_with_detections, (xmin, ymin), (xmax, ymax), (255, 255, 255), -1)      
                else:
                    print("Saving image with undetected signature...")
                    cv2.imwrite(f'modified_pdfs/{os.path.basename(img)}', image)
                box_index += 1

        return image_with_detections

    @staticmethod
    def convert_to_pdf(path):
        print("Converting modified images to pdf...")
        with open("redacted.pdf","wb") as f:
            f.write(img2pdf.convert(path))
        print("Done converting!")


    def redact_pdf(self):
        """
        Function that takes in set of images, 
        runs them through the signature object dection model, 
        redacts the detected signatures and returns a modied pdf file.

        Args: None

        Returns: Modified pdf
        """
        # Get path to images
        image_path = glob.glob(f"pdf_to_images/*.jpg")
        # Sort image path
        sorted_path = sorted(image_path, key=lambda k: int(k.rsplit(".")[1].split("_")[-1]))
        
        #Create path to store modified images
        if os.path.exists('modified_pdfs'):
            print('Folder already exits!')
        else:
            print("creating path...")
            os.makedirs('modified_pdfs')

        for img in sorted_path[4:7]:
            print(f'Running inference for {img}...')
            #Redact Signatures
            signature_redacted_img = self.redact_signature(img)
            #Redact Name and Date
            nameDate_redactedImg = self.redact_name_and_date(signature_redacted_img)
            #Write the modified images to output path
            cv2.imwrite(f'modified_pdfs/{os.path.basename(img)}', nameDate_redactedImg)

        # Merge sorted image path with modified image path to get the complete pdf pages
        new_path = sorted_path[:4] + glob.glob("modified_pdfs/*.jpg") + sorted_path[7:]
        #Convert modified images to pdfs
        self.convert_to_pdf(new_path)
  
