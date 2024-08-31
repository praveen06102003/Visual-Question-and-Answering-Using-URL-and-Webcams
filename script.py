import cv2
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import tkinter as tk
import requests
from io import BytesIO

# Initialize BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

def vqa_with_image(img_pil):
    while True:
        question = input("Ask a question about the picture (or type 'next' to capture/upload a new image, 'exit' to quit): ")
        if question.lower() == 'next':
            return False  # Indicate to go back to the main loop
        if question.lower() == 'exit':
            return True  # Indicate to exit the program

        # Preprocess image and question
        inputs = processor(img_pil, question, return_tensors="pt")
        
        # Generate answer
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True)
        
        # Print answer
        print("Answer:", answer)

def capture_from_webcam():
    try:
        print("Accessing webcam...")
        # Open webcam
        cap = cv2.VideoCapture(0)  # Change index to 1 if your selfie camera is at index 1
        
        # Check if webcam opened successfully
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Display the captured frame
            cv2.imshow("Webcam Feed", frame)

            # Wait for the user to press 'c' to capture the image or 'q' to cancel
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # If 'c' is pressed
                # Save the captured image
                image_path = "captured_image.jpg"
                cv2.imwrite(image_path, frame)
                print(f"Image saved as {image_path}")

                # Convert the captured image to PIL format
                img_pil = Image.open(image_path).convert('RGB')

                # Ask questions and get answers
                if vqa_with_image(img_pil):
                    break  # Exit the loop if the user chooses to quit
                else:
                    cv2.destroyAllWindows()  # Close the window
                    break  # Exit the loop to capture/upload a new image

            elif key == ord('q'):  # If 'q' is pressed
                break

        # Release the webcam and close the window
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error accessing webcam: {e}")

def capture_and_vqa():
    while True:
        print("Choose an option:")
        print("1. Capture an image from webcam (Press 'c')")
        print("2. Upload an image (Press 'u')")
        print("3. Quit (Press 'q')")
        choice = input("Your choice: ")

        if choice == 'c':
            capture_from_webcam()

        elif choice == 'u':
            image_path = input("Enter the URL or path of the image: ")
            try:
                if image_path.startswith('http'):
                    response = requests.get(image_path)
                    img_pil = Image.open(BytesIO(response.content)).convert('RGB')
                else:
                    img_pil = Image.open(image_path).convert('RGB')
                
                # Ask questions and get answers
                if vqa_with_image(img_pil):
                    return  # Exit the function if the user chooses to quit

            except Exception as e:
                print(f"Error loading image: {e}")

        elif choice == 'q':
            return  # Exit the function if the user chooses to quit
        else:
            print("Invalid choice. Please choose again.")

# Call the function to capture image from the webcam and perform VQA
capture_and_vqa()
