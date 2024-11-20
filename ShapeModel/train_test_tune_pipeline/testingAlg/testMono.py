import os
import sys
import cv2
import time
import psutil
from ultralytics import YOLO
import matplotlib.pyplot as plt
from functools import reduce

# FUNCTIONS 
# Validate the model
def validation(model, dataset, configuration):
    metrics = model.val(data=dataset,
                        imgsz=configuration["imgsz"],
                        batch=configuration["batch"],
                        save_hybrid=configuration["save_hybrid"],
                        conf=configuration["conf"],
                        device=configuration["device"],
                        plots=configuration["plots"],
                        save_json=configuration["save_json"]) 
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category

# Process frame for performance
def process_frame(model, frame, results_memory, results_time, model_name):
    initial_memory = get_ram_usage()
    start_time = time.perf_counter()
    #(https://docs.ultralytics.com/modes/predict/#working-with-results)

    results = model(frame)

    end_time = time.perf_counter()
    after_memory = get_ram_usage()

    results_memory[model_name].append(after_memory - initial_memory)
    results_time[model_name].append(end_time - start_time)

# Gets RAM usage
def get_ram_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB

# Process video
def process_video(video_path, model, results_memory, results_time, model_name):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error opening video file")
        return

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

# Process images
def process_image(images_path, model, results_memory, results_time, model_name):
    if not os.path.exists(images_path):
        print(f"Error: The directory {images_path} does not exist.")
        return

    # Loop through the files in the specified directory
    for filename in os.listdir(images_path):
        f = os.path.join(images_path, filename)
        # Check if it is a file
        if os.path.isfile(f):
            frame = cv2.imread(f)
            if frame is None:
                print(f"Error: Could not load image {f}.")
                continue  # Skip to the next file if there's an error
            else:
                process_frame(model, frame, results_memory, results_time, model_name)

# ______________________________________________________________________________________________________

def main():
    # Print current working directory
    print("Current Working Directory:", os.getcwd())

    # Ask the user for the model name (e.g., 'yolov8n.pt')
    model_name_input = input("Enter the model name (with .pt extension): ").strip()

    # Define the relative path for the model and dataset files
    model_path = f'ShapeModel/train_test_tune_pipeline/models/{model_name_input}'
    dataset_path = 'ShapeModel/train_test_tune_pipeline/data/'
    # Check if the model file exists before loading it
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist.")
        sys.exit(1)  # Exit if model does not exist

    # If model exists, print the path and load the model
    print(f"Model file found at: {model_path}")

    # Load the YOLO model (it will not download the model if it exists at the given path)
    model = YOLO(model_path)

    results_memory = {model_name_input: []}
    results_time = {model_name_input: []}

    val_configuration = {
        "imgsz": 640,
        "batch": 16,
        "save_hybrid": True,
        "conf": .5,
        "device": "cpu",
        "plots": True,
        "save_json": True
    }

    print("Model path:", model_path)
    #print("Dataset YAML path:", dataset_yaml)

    # User Inputs: 
    # Ask if they want default configuration or not for validation?
    config = input("Do you want default configuration? (Y/n): \n")
    if config in ['N', 'n']:
        for key in val_configuration:
            val_configuration[key] = input(f'{key}: ')
    else:
        print("Default Configuration for Validation\n")

    # Ask if they are testing on images or video?
    data_type = input("Video or Images? (V/I): \n")

    if data_type in ['V', 'v']:
        print("Video Data Selected\n")
        video_path = input("Enter the path to the video file: ")
        process_video(video_path, model, results_memory, results_time, model_name_input)
    elif data_type in ['I', 'i']:
        print("Image Data Selected\n")
        dataDir = input("Enter Dataset Directory: ")
        process_image(dataset_path+dataDir, model, results_memory, results_time, model_name_input)
        # Run accuracy check 
        #validation(model, dataset_path+dataDir+"/data.yaml", val_configuration)
    else:
        print("Invalid Data Type\n")
        sys.exit(1)

    # These results will be put into a table
    combined_results = [['Model Name', 'Avg Time (s)', 'Avg Memory (MB)']]

    # Print out the time data
    if results_time[model_name_input]:
        avg_time = (reduce(lambda a, b: a + b, results_time[model_name_input])) / len(results_time[model_name_input])
        print(f"Model: {model_name_input}, Average inference elapsed Time: {avg_time:.4f} seconds")
    else:
        print(f"No time data collected for model: {model_name_input}")

    if results_memory[model_name_input]:
        avg_mem = (reduce(lambda a, b: a + b, results_memory[model_name_input])) / len(results_memory[model_name_input])
        print(f"Model: {model_name_input}, Average memory usage: {avg_mem:.4f} MB")
    else:
        print(f"No memory data collected for model: {model_name_input}")

    # Add model names and data
    combined_results.append([model_name_input, f"{avg_time:.4f}", f"{avg_mem:.4f}"])

    # Creating figure
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    # Create and display the table
    table = ax.table(cellText=combined_results, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    plt.show()

if __name__ == "__main__":
    main()