#sgpt.py
import cv2
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def draw_black_points(image, black_points):
    """Draw black points on the image."""
    for point in black_points:
        cv2.circle(image, tuple(point), 2, (0, 0, 0), -1)

def compress_hull_inward(hull_points, normals, boundary_points, road_width):
    """Compress the hull inward by a specified road width."""
    compressed_hull = []
    for point, normal in zip(hull_points, normals):
        for boundary_point in boundary_points:
            dist = np.linalg.norm(boundary_point - point)
            if dist > road_width:
                new_point = point + normal * (road_width)  # Move inward by road width
                compressed_hull.append(new_point)
                new_point = point + normal * (2*road_width)
                compressed_hull.append(new_point)
                break
    return np.array(compressed_hull)

def plot_normals(image, black_points):
    """Plot normals to the edges of the convex green hull."""
    num_points = len(black_points)
    normals = []

    # Calculate centroid of the green area
    centroid = np.mean(black_points, axis=0)

    for i in range(num_points):
        # Get current and next point in the green contour
        p1 = black_points[i]
        p2 = black_points[(i + 5) % num_points]  # Wrap around

        # Compute edge vector and perpendicular normal
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])  # Rotate edge 90 degrees

        # Normalize and ensure the normal points inward
        normal = normal.astype(np.float64) / np.linalg.norm(normal)
        if np.dot(normal, centroid - p1) < 0:
            normal = -normal

        normals.append(normal)

        # Plot the normal for visualization
        normal_length = 20
        normal_endpoint = p1 + normal * normal_length
        #cv2.arrowedLine(image, tuple(p1), tuple(normal_endpoint.astype(int)), (0, 0, 255), 1)

    return normals

def detect_green_points(image):
    """Detect green points in the image."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    green_points = [point[0] for contour in contours for point in contour]
    return np.array(green_points)

def get_black_points_in_group(image, col_start, col_end):
    """Get black points in a specified column group."""
    black_points = []
    for col in range(col_start, col_end):
        for row in range(image.shape[0]):
            if np.all(image[row, col] == 0):  # Black pixel
                black_points.append([col, row])
    return np.array(black_points)
#refer below function for plotting intersection points
def draw_hull_points(image, hull_points):
    """Draw hull points on the image with alternating colors."""
    for i in range(len(hull_points)):
        start_point = tuple(hull_points[i].astype(int))  # Ensure points are integers
        
        # Determine color based on the index (even or odd)
        color = (255, 0, 0) if i % 2 == 0 else (0, 255, 0)  # Green for even, Red for odd
        
        # Draw the point as a circle
        cv2.circle(image, start_point, 5, color, -1)  # Draw circle with radius 1
def draw_intersections(image, intersections=[], color=(255, 0, 255)):
    """Draw the intersections of hull normal with boundary on the image."""
    for i in range(len(intersections)):
        start_point = tuple(intersections[i])
        end_point = tuple(intersections[(i + 1) % len(intersections)])
        #the line below should be modified
        cv2.line(image, int(start_point), int(end_point), color, 2)

def check_intersection_with_boundary(normals, green_points, boundary_points, image, tolerance=2, step_size=1, max_steps=100):
    """
    Check if normals intersect with black boundary points by stepping along each normal.
    
    Parameters:
    normals (np.array): Array of normal vectors.
    green_points (np.array): Points where normals originate.
    boundary_points (np.array): All black boundary points in the image.
    image (np.array): The image array (to detect black points).
    tolerance (float): Distance tolerance for intersection.
    step_size (float): Step size for sampling points along the normal.
    max_steps (int): Maximum steps to take along the normal direction.
    
    Returns:
    intersections (list of np.array): List of boundary points where normals intersect.
    """
    intersections = []

    # Iterate over each green point and its corresponding normal
    for normal, p1 in zip(normals, green_points):
        found_intersection = False
        
        # Walk along the normal direction for a specified number of steps
        for step in range(1, max_steps + 1):
            # Calculate the next point along the normal direction
            test_point = p1 + step * step_size * normal
            
            # Ensure the test point is within the image bounds
            if (test_point[0] < 0 or test_point[1] < 0 or 
                test_point[0] >= image.shape[0] or test_point[1] >= image.shape[1]):
                break  # Stop if the point is out of bounds

            # Check if the test point is black (boundary point)
            if np.all(image[int(test_point[1]), int(test_point[0])] == 0):  # Check if pixel is black
                intersections.append(test_point)
                found_intersection = True
                break  # Stop further searching for this normal if an intersection is found

    return intersections


# Load the image
image_path = "hullss.png"
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load image.")
else:
    # Convert grayscale image to BGR for visualization
    image_with_hull = image

    # Get black points in groups of 25-pixel columns
    col_group_size = 25
    height, width,_ = image.shape
    all_black_points = []
    for col_start in range(0, width, col_group_size):
        col_end = min(col_start + col_group_size, width)
        black_points = get_black_points_in_group(image, col_start, col_end)
        all_black_points.extend(black_points)

    all_black_points = np.array(all_black_points)

    # Detect green points and compute normals
    green_points = detect_green_points(image)
    normals = plot_normals(image, green_points)

    # Draw black points
    if len(all_black_points) > 0:
        draw_black_points(image_with_hull, all_black_points)

    # Check for intersections between normals and boundaries
    intersections = check_intersection_with_boundary(normals, green_points, all_black_points,image)

    # Compress the hull inward
    road_width = 10  # Example road width
    compressed_hull = compress_hull_inward(green_points, normals, all_black_points, road_width)
    if len(compressed_hull) > 0:
        draw_hull_points(image_with_hull, compressed_hull.astype(int))
        #draw_intersections(image_with_hull, intersections, color=(255, 0, 255))

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_hull, cv2.COLOR_BGR2RGB))
    plt.title("Blue is middle of track")
    #try appending path to blank image and also outer bound to other, and try other tracks
    plt.axis("off")
    plt.savefig('second.png')
    plt.show()
