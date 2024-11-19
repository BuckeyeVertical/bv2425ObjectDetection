import cv2
import sys
import time
import psutil
from ultralytics import YOLO
from functools import reduce
import os

#FUNCTIONS 
# Validate the model
def validation(model, dataset, configuration):
    metrics = model.val(data = dataset,
                        imgsz = configuration["imgsz"],
                        batch = configuration["batch"],
                        save_hybrid = configuration["save_hybrid"],
                        conf = configuration["conf"],
                        device = configuration["device"],
                        plots = configuration["plots"],
                        save_json = configuration["save_json"]) 
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category

#process fram for performance
def process_frame(model, frame, results_memory, results_time, model_name):
        initial_memory = get_ram_usage()
        start_time = time.perf_counter()

        results = model(frame)

        end_time = time.perf_counter()
        after_memory = get_ram_usage()

        results_memory[model_name].append(after_memory - initial_memory)
        results_time[model_name].append(after_memory - initial_memory)

# Gets ram usage
def get_ram_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB

#process video
def process_video(video_path, model, results_memory, results_time, model_name):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error opening video file")

    # Loop through the video frames
    while True:
        ret, frame = video.read()
        if not ret:
            break

        process_frame(model, frame, results_memory, results_time, model_name)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close any open windows
    video.release()
    cv2.destroyAllWindows()

#process images
def process_image(images_path, model, results_memory, results_time, model_name):
    directory = images_path

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            frame = cv2.imread(f)
            if frame is None:
                print("Error: Could not load image.")
                return
            else:
                process_frame(model, frame, results_memory, results_time, model_name)
    
    
#______________________________________________________________________________________________________

def main():
    # Initialize models
    models_directory = 'bv2425ObjectDetection/ShapeModel/train_test_tune_pipeline/models'
    modelNames = ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']

    results_memory = {
         "yolo11n": [],
         "yolo11s": [],
         "yolo11m": [],
         "yolo11l": [],
         "yolo11x": []    
    }

    results_time = {
         "yolo11n": [],
         "yolo11s": [],
         "yolo11m": [],
         "yolo11l": [],
         "yolo11x": []
    }

    val_configuration = {
        "imgz": 640,
        "batch": 16,
        "save_hybrid": True,
        "conf": .5,
        "device": "cpu",
        "plots": True,
        "save_json": True
    }

    dataset_name = "temp.yaml" #change to dataset name
    dataset_path = "ShapeModel/test_pipeline/data" 
    dataset_yaml = dataset_path + "/" + dataset_name

    #User Inputs: 
    #ask if they want default configuration or not for validation?
    config = input("Do you want default configuration? (Y/n): \n")
    if config == 'N'or config == 'n':
        for key, value in val_configuration.items():
            val_configuration[key] = input('{}: '.format(key))
    else:
        print("Default Configuration for Validation\n")

    #ask if they testing on images or video?
    data_type = input("Video or Images? (V/I): \n")

    if data_type == 'V' or data_type == 'v':
        print("Video Data Selected/n")
    elif data_type == 'I' or data_type == 'i':
        print("Image Data Selected/n")
    else:
        print("Invalid Data Type/n")
        sys.exit(1)

    # Loop through each model
    for model_name in modelNames:
        print(f"Testing model: {model_name}")
        model_path = models_directory + model_name
        # Load model
        model = YOLO(model_path)
        
        
        #video and image processing for each model
        if data_type == 'V' or data_type == 'v':
            video_path = dataset_path
            process_video(video_path, model, results_memory, results_time, model_name)
        else:
            process_image(dataset_path, model, results_memory, results_time, model_name)
            #run accuracy check 
            #validation(model, dataset_yaml, val_configuration)
 
       

    #TODO: display avgs and all data cleanly
    #Note: display average instead of individual
    # Print out the time data
            
    for names_time in results_time:
        avg_time = (reduce(lambda a, b: a+b, results_time[names_time]))/2
        print(f"Model: {results_time[names_time]}, Total inference elapsed Time: {avg_time:.4f} seconds")

    for names_mem in results_memory:
        avg_mem = (reduce(lambda a, b: a+b, results_time[names_mem]))/2
        print(f"Model: {results_memory[names_mem]}, Total inference elapsed Time: {avg_mem:.4f} seconds")


if __name__ == "__main__":
    main()
