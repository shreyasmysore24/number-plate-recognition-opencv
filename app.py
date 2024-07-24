import streamlit as st
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

# Function to process the image and get the result
def process_image(image):
    # Convert the uploaded file to an OpenCV image format
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply bilateral filter for noise reduction
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)

    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    # Create a mask and extract the license plate area
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [location], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

    # Use EasyOCR to read text from the cropped image
    reader = easyocr.Reader(['en'])
    result = reader.readtext(cropped_image)

    # Get the detected text
    text = result[0][-2] if result else "No text detected"

    # Draw the text and rectangle on the original image
    font = cv2.FONT_HERSHEY_SIMPLEX
    res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), fontFace=font, fontScale=1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

    return res, text

# Streamlit app
st.title("License Plate Recognition")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Process the image and get the result
    result_img, detected_text = process_image(img)

    # Convert BGR image to RGB for Streamlit
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # Display the result
    st.image(result_img_rgb, caption='Processed Image', use_column_width=True)
    st.write(f"Detected Text: {detected_text}")
else:
    st.write("Upload an image to start processing.")
