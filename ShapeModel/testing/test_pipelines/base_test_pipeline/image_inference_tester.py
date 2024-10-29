# Adarsha's script for testing several models at once
import cv2
import time
import psutil
from ultralytics import YOLO

def process_frame(model, frame, results_memory, results_time, model_name):
        initial_memory = get_ram_usage()
        start_time = time.perf_counter()

        results = model(frame)

        end_time = time.perf_counter()
        after_memory = get_ram_usage()

        results_memory[model_name].append(after_memory - initial_memory)
        results_time[model_name].append(end_time - start_time)

# Gets ram usage
def get_ram_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB


def main():
    # Initialize models
    models_directory = 'C:\\Users\\adars\\repos\\bv2425ObjectDetection\\ShapeModel\\testing\\misc\\'
    modelNames = ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']

    # Load the image from file
    image_path = 'C:\\Users\\adars\\repos\\bv2425ObjectDetection\\ShapeModel\\testing\\data\\basketball_court.jpg'

    frame = cv2.imread(image_path)

    if frame is None:
        print("Error: Could not load image.")
        return

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


    # Loop through each model
    for model_name in modelNames:
        print(f"Testing model: {model_name}")
        model_path = models_directory + model_name

        # Load model
        model = YOLO(model_path)

        # TODO: video loop

        process_frame(model, frame, results_memory, results_time, model_name) 
       

    # TODO: display average instead of individual
    # Print out the time data
    for result in      results_time:
        print(f"Model: {result['model']}, Total inference elapsed Time: {result['inference_time']:.4f} seconds")

    # Print out the memory usage
    for usage in results_memory:
        print(f"Model: {usage['model']}, Total change in memory usage: {usage['memory_usage']:.4f} MB")

if __name__ == "__main__":
    main()
