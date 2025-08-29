import cv2
import mediapipe as mp
import websockets
import asyncio
import json
import DK_process as DK
import time

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,      
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# Store connected clients
clients = set()

# WebSocket handler for client connections
async def handle_client(websocket):
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)

# Broadcast data to all connected clients
async def broadcast(data):
    if clients:
        websockets_tasks = []
        for websocket in clients:
            websockets_tasks.append(asyncio.create_task(
                websocket.send(json.dumps(data))
            ))
        await asyncio.gather(*websockets_tasks)

# Main processing loop
async def process_video():
    
    w1, w2, lowest_point = None, None, None # Initialize parameters
    
    cap = cv2.VideoCapture(0) 
    
   # Get start time
    start_time = time.time()
    timer = 15
    
    # counter for frames
    points_counter = 0
    
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            print("No frame captured. Exiting...")
            break
        
        # counter for frames
        points_counter = 0
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and get landmarks
        results = pose.process(image_rgb)
        
        # Calculate elapsed time and remaining time
        elapsed_time = time.time() - start_time
        remaining_time = max(timer - elapsed_time, 0)
        
        # if remaining time is not 0, then format the timer text
        if remaining_time > 0:
            
            # if the elapsed time is less than 5 seconds, then show special text
            if  elapsed_time <= 5:
                # Split the text into individual lines
                timer_lines = "Take the T shape \n   for scaling".split('\n')
            
                # Calculate line height (using a sample line)
                sample_size, _ = cv2.getTextSize("Sample", cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)
                line_height = sample_size[1] + 30  # Add 10px spacing between lines

                # Starting position (centered vertically)
                start_y = image.shape[0] // 2 - (line_height * len(timer_lines)) // 2

                # Draw each line separately
                for i, line in enumerate(timer_lines):
                    y_position = start_y + i * line_height
                    cv2.putText(
                        img=image,
                        text=line,
                        org=(46, y_position),  # X remains at 10, Y increments per line
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        color=(0, 255, 0),
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )
                    
            # Format timer text (MM:SS)
            timer_text = f"Timer: {int(remaining_time // 60):02d}:{int(remaining_time % 60):02d}"

            # if remaining_time > 0:
            # Draw timer on the frame (top-left corner)
            cv2.putText(
                img=image,              # Image to draw on 
                text=timer_text,        # Text to display
                org=(20, 40),           # (x, y) position
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # Font type
                fontScale=1.0,          # Font scale    
                color=(0, 255, 0),      # Green text
                thickness=1,            # Thickness of the text
                lineType=cv2.LINE_AA    # Anti-aliasing
            )
        
            # Check if landmarks are detected
            if (results.pose_landmarks and results.pose_world_landmarks):
                
                # Extract and process landmarks
                displayed_landmark_list, world_landmark_list = DK.extract_all_landmarks(results)
                
                # Check if there are landmarks detected
                if displayed_landmark_list and world_landmark_list:
                    
                    # Get the required landmarks for the 3d world and 2d displayed images
                    world_required_landmarks, displayed_required_landmarks = DK.get_required_keypoints(world_landmark_list, displayed_landmark_list)
                    
                    # Get the Scaling parameters for the world image
                    w1, w2, lowest_point = DK.get_parameters_for_scaling(world_required_landmarks)
                    
                    # Scale the world required landmarks=
                    scaled_world_required_landmarks = DK.scale_keypoints(world_required_landmarks, w1, w2, lowest_point)

                    # Draw the keypoints with the Skeleton for the 2d displayed image
                    image = DK.draw_skeleton(displayed_required_landmarks, image, keypoints_3D=world_required_landmarks, draw_keypoints_values=True)
                    
                    # Print the Scaling parameters (W1, W2, lowest_point)
                    # print(f'W1: {w1}, W2: {w2}, Lowest Point: {lowest_point}')
                            
                    points_counter = points_counter + 1
                    
                    print(f'{points_counter}')
                    print(75 * '*')
                    needed = ['left_wrist', 'right_wrist', 'head', 'left_ankle', 'right_ankle']
                    for name, lm in world_required_landmarks.items():
                        if name in needed:
                            print(f"{name}: [{lm['x']:.2f}, {lm['y']:.2f}, {lm['z']:.2f}]")
                    print(75 * '*')
            else:
                print('No landmarks detected')
                print(75 * '*')
                print(results.pose_landmarks)
                print(75 * '*')
                print(results.pose_world_landmarks)
                print(75 * '*')
        else:
            # Check if landmarks are detected
            if (results.pose_landmarks and results.pose_world_landmarks):     
                # Extract and process landmarks
                dis_landmark_list, world_landmark_list = DK.extract_all_landmarks(results)
            
            # Check if there are landmarks detected
            if dis_landmark_list and world_landmark_list:
                
                # Get the required landmarks for the 3d world and 2d displayed images
                world_required_landmarks, dis_required_landmarks = DK.get_required_keypoints(world_landmark_list, dis_landmark_list)
                
                # Scale the keypoints for Unity model character
                processed_data = DK.scale_keypoints(world_required_landmarks, w1, w2, lowest_point)
                
                # Draw the keypoints with the Skeleton
                image = DK.draw_skeleton(dis_required_landmarks, image, keypoints_3D=world_required_landmarks, draw_keypoints_values=True)
                
                # Send the processed data to Unity via WebSocket
                await broadcast(processed_data)
                
                # points_counter = points_counter + 1
                # print(f'{points_counter}')
                # print(75 * '*')
                # needed = ['left_wrist', 'right_wrist', 'head', 'left_ankle', 'right_ankle']
                # for name, lm in processed_data.items():
                #     if name in needed:
                #         print(f"{name}: [{lm['x']:.2f}, {(lm['y']):.2f}, {lm['z']:.2f}]")
                # print(75 * '*')
            
        cv2.imshow('Video Feed', image)
        
        # Break loop after timer seconds or if 'q' is pressed
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        
        # Add a small delay to control the frame rate
        await asyncio.sleep(0.033)  # approximately 30 fps
                
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    return w1, w2, lowest_point
    

# Main function to start the WebSocket server and video processing
async def main():
    # Start WebSocket server using the modern approach
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
    # async with websockets.serve(handle_client, "localhost", 8765): 
            
        print(75 * '#')
        print("WebSocket server started and listening on ws://0.0.0.0:8765")
        print(75 * '#')
        
        # Start video processing
        w1,w2,min_point = await process_video()

        print(f'The Finial W1: {w1}')
        print(f'The Finial W2: {w2}')
        print(f'The Finial Minimum Point: {min_point}')
# Run the server
if __name__ == "__main__":
    asyncio.run(main())
    