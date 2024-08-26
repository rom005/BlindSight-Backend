import uvicorn
from PIL import Image
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, Response
from pathlib import Path
import uuid
from fastapi.middleware.cors import CORSMiddleware
import cv2
import os
import keras
import numpy as np


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/isAlive")
def is_alive():
    return "alive"

def predict_image_file(file: UploadFile):
    model = YOLO("OneClassBraille8.8.pt")
    im1 = Image.open(file.file)
    results = model.predict(source=im1, save=False, conf=0.4, iou=0.3)
    return results[0]

@app.post("/predict/predicted-image")
def get_predicted_image(file: UploadFile = File()):
    result = predict_image_file(file)

    random_file_name = uuid.uuid4().hex.upper()[0:6]
    predict_path = "results/" + random_file_name + ".jpg"

    result_image = result.orig_img  # This should give you the original image without boxes
    cv2.imwrite(predict_path, cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR))

    full_path = Path(predict_path)

    return FileResponse(full_path)


model = keras.models.load_model("138BrailleNet.keras")

@app.post("/predict/predicted-word")
def get_predicted_word(file: UploadFile = File()):
    result = predict_image_file(file)
    random_file_name = uuid.uuid4().hex.upper()[0:6]
    predict_path = "results/" + random_file_name + ".jpg"
    predict_path_with_rectangles = "results/" + random_file_name + "with_rectangles" + ".jpg"

    result.save(filename=predict_path_with_rectangles)

    result_image = result.orig_img  # Original image without the boxes
    cv2.imwrite(predict_path, cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR))


    boxes = result.boxes.xyxy.tolist()
    classes = result.boxes.cls.tolist()
    names = result.names
    confidences = result.boxes.conf.tolist()

    image = cv2.imread(predict_path)

    cropped_dir = "cropped_images/"
    if os.path.isdir(cropped_dir):
        for file_name in os.listdir(cropped_dir):
            os.remove(cropped_dir + file_name)

        os.removedirs(cropped_dir)

    os.makedirs(cropped_dir, exist_ok=True)

    # Combine boxes, classes, and confidences into a list of tuples
    combined = list(zip(boxes, classes, confidences))

    # Sort by the y-coordinate of the top-left corner of the box
    sorted_by_y = sorted(combined, key=lambda item: item[0][1])

    line_breaks = detect_line_breaks(sorted(boxes, key=lambda item: item[1]))
    lines = []
    start_index = 0
    print("line_breaks")
    print(line_breaks)
    for break_index in line_breaks:
        line = sorted_by_y[start_index:break_index + 1]
        lines.append(line)
        start_index = break_index + 1
    lines.append(sorted_by_y[start_index:])

    predicted_sentence = ""

    counter = 1
    for line in lines:
        # Sort by the x-coordinate of the top-left corner of the box

        sorted_by_x = sorted(line, key=lambda item: item[0][0])
        print(counter)
        print(sorted_by_x)
        counter += 1

        gaps_indices = detect_x_spaces(sorted_by_x)

        counter = 0
        for idx, (box, cls, conf) in enumerate(sorted_by_x):
            counter = counter + 1
            x1, y1, x2, y2 = [int(coord) for coord in box]
            confidence = conf
            name = names[int(cls)]
            cropped_image = image[y1:y2, x1:x2]
            cropped_image_path = os.path.join(cropped_dir, f"{chr(65 + counter)}_{name}_{idx}_{confidence:.2f}.jpg")
            cv2.imwrite(cropped_image_path, cropped_image)

        predicted_word = predict_word()
        print(predicted_word)

        if os.path.isdir(cropped_dir):
            for file_name in os.listdir(cropped_dir):
                os.remove(cropped_dir + file_name)

        curr_index = 0
        for gap_index in gaps_indices:
            predicted_sentence += predicted_word[curr_index:gap_index + 1] + " "
            curr_index = gap_index + 1

        predicted_sentence += predicted_word[curr_index:] + " "

    print(predicted_sentence)
    return predicted_sentence


def predict_word():
    cropped_dir = "cropped_images/"
    index = 0
    word = ""
    for file_name in sorted(os.listdir(cropped_dir)):
        img = Image.open(cropped_dir + file_name)
        image_array = process_image(img)
        res = model.predict(image_array)
        letter_number = np.argmax(res)

        word += chr(97 + letter_number)
        index += 1

    return word


def detect_line_breaks(boxes, gap_threshold_ratio=4):
    # Calculate the vertical distance (gap) between consecutive boxes
    gaps = [boxes[i + 1][1] - boxes[i][1] for i in range(len(boxes) - 1)]

    # Calculate the average gap
    if not gaps:
        return []
    average_gap = sum(gaps) / len(gaps)

    # Determine the threshold for detecting line breaks
    threshold = average_gap * gap_threshold_ratio

    # Identify indices where the gap exceeds the threshold
    line_breaks = [i for i, gap in enumerate(gaps) if gap > threshold]

    return line_breaks


def detect_x_spaces(items, gap_threshold_ratio=1.5):
    # Calculate the horizontal distance between consecutive items
    gaps = [items[i + 1][0][0] - items[i][0][0] for i in range(len(items) - 1)]
    if not gaps:
        return []
    # Calculate the average gap
    average_gap = sum(gaps) / len(gaps)
    # Determine the threshold for detecting spaces
    threshold = average_gap * gap_threshold_ratio

    # Identify indices where the gap exceeds the threshold
    spaces = [i for i, gap in enumerate(gaps) if gap > threshold]

    return spaces


def process_image(img):
    img = img.convert('L')

    img = img.resize((28, 36))

    img_array = np.stack((img,) * 3, axis=-1)

    img_array = np.array(img_array, dtype='uint8')

    img_array = np.expand_dims(img_array, axis=0)

    return img_array


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

