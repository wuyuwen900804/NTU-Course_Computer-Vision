import cv2
import numpy as np
from tqdm import tqdm
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor

def process_frame(frame_pair):
    # Retrieve the previous frame and the current frame
    prev_gray, gray = frame_pair

    ### Create masks based on the common characteristic of bus doors ###
    # Most of the doors are black
    lower_black, upper_black = 0, 80
    black_mask = cv2.inRange(gray, lower_black, upper_black)
    # Most door frames are straight lines
    low_threshold, high_threshold = 100, 250
    edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=3)
    rho, theta, threshold = 1, np.pi / 180, 100
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=150, maxLineGap=4)
    line_mask = np.zeros_like(black_mask)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)

        ### Calculate optical flow ###
        # Combine the black_mask and line_mask and check for any overlapping parts
        combined_mask = cv2.bitwise_and(line_mask, black_mask)
        if np.sum(combined_mask) > 0:
            # Calculate the optical flow between the previous frame and the current frame
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 0.5, 0)
            # Retrieve the magnitude and angle of the optical flow vectors for all pixels.
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # Use weighted magnitude and angle to determine optical flow strength
            weighted_magnitude = magnitude * (1 + 0.3 * np.sin(angle))
            # Obtain the optical flow strength of the current frame
            motion_magnitude = np.mean(weighted_magnitude[combined_mask == 255])
            return motion_magnitude
    return None

def compute_optical_flow(video_path):
    # Check if the video exists
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Cannot read the video file.")
        return None

    # Initialization
    frames = []
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Read and store frames
    for _ in tqdm(range(total_frames - 1), desc="Reading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append((prev_gray.copy(), gray.copy()))
        prev_gray = gray
    cap.release()
    
    # Process frames using process_frame in parallel
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_frame, frames), total=len(frames), desc="Processing frames"))
    # Filter and Return Results
    motion_magnitudes = [r for r in results if r is not None]
    return motion_magnitudes

def merge_close_peaks(peaks, peak_heights, distance_threshold):
    ### Avoid treating the opening of the same door as two separate events
    merged_peaks, merged_heights = [], []
    current_peak, current_height = peaks[0], peak_heights[0]
    for i in range(1, len(peaks)):
        # Prevent the detection of multiple peaks at short distances
        if peaks[i] - current_peak <= distance_threshold:
            # Update the highest peak
            if peak_heights[i] > current_height:
                current_peak, current_height = peaks[i], peak_heights[i]
        else:
            # Determine the current peak and search for the next peak
            merged_peaks.append(current_peak)
            merged_heights.append(current_height)
            current_peak, current_height = peaks[i], peak_heights[i]
    # Append the last peak
    merged_peaks.append(current_peak)
    merged_heights.append(current_height)
    return merged_peaks, merged_heights

def find_largest_2(motion_magnitudes):
    ### If no particularly prominent peak is found, output the top two peaks
    # Rescale the optical flow intensity
    max_magnitude = max(motion_magnitudes)
    scaled_magnitudes = [(m / max_magnitude) * 15 for m in motion_magnitudes]
    # Identify the peak
    peaks, _ = find_peaks(scaled_magnitudes, height=0, distance=50)
    peak_heights = [scaled_magnitudes[peak] for peak in peaks]
    # Find the top two peaks
    major_peaks = sorted(zip(peaks, peak_heights), key=lambda x: x[1], reverse=True)[:2]
    major_peaks = sorted(major_peaks, key=lambda x: x[0])
    opening_frame, closing_frame = major_peaks[0][0], major_peaks[1][0]
    return [opening_frame, closing_frame], [motion_magnitudes[opening_frame], motion_magnitudes[closing_frame]]

def find_door_states(motion_magnitudes, height_threshold, dist_threshold, merge_dist):
    ### Output the frame timestamps of door opening and closing
    # Rescale the optical flow intensity
    max_magnitude = max(motion_magnitudes)
    scaled_magnitudes = [(m / max_magnitude) * 15 for m in motion_magnitudes]
    # Identify the primary peak
    peaks, properties = find_peaks(scaled_magnitudes, height=height_threshold, distance=dist_threshold)
    peak_heights = properties['peak_heights']
    # If no particularly high peak is found
    if len(peaks) == 0 or len(peaks) == 1:
        peaks, peak_heights = find_largest_2(motion_magnitudes)
        return peaks, peak_heights
    else:
        # Avoid treating the opening of the same door as two separate events
        merged_peaks, merged_heights = merge_close_peaks(peaks, peak_heights, merge_dist)
        return merged_peaks, merged_heights

def guess_door_states(video_filename):
    # Compute Optical Flow
    motion_magnitudes_data = compute_optical_flow(video_filename)
    if not motion_magnitudes_data:
        return [], []
    peaks, _ = find_door_states(motion_magnitudes_data, height_threshold=7, dist_threshold=10, merge_dist=50)
    # Output the timestamps of event occurrences along with their event names
    door_states = []
    for i in range(len(peaks)):
        if i % 2 == 0:
            door_states.append((int(peaks[i]), "Opening"))
        else:
            door_states.append((int(peaks[i]), "Closing"))
    return door_states