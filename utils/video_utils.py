import cv2

# Function to read frames from a video file
def read_video(video_path):
    # Initialize a VideoCapture object to read the video
    cap = cv2.VideoCapture(video_path)
    frames = []  # List to store video frames
    
    # Loop to read all frames until the end of the video
    while True:
        ret, frame = cap.read()  # Read a frame
        if not ret:  # Break the loop if no more frames are available
            break
        frames.append(frame)  # Append the frame to the list
    
    return frames  # Return the list of frames

# Function to save frames as a video file
def save_video(output_video_frame, output_video_path):
    # Define the codec for video writing (XVID in this case)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # Initialize a VideoWriter object with the specified output path, codec, frame rate, and frame size
    out = cv2.VideoWriter(output_video_path, fourcc, 24, 
                          (output_video_frame[0].shape[1], output_video_frame[0].shape[0]))
    
    # Write each frame to the output video
    for frame in output_video_frame:
        out.write(frame)
    
    # Release the VideoWriter object to save the video file
    out.release()
