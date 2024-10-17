import cv2
import numpy as np
from scipy.spatial import ConvexHull, QhullError
import matplotlib.pyplot as plt


# Function to get black points in a 25-column group
def get_black_points_in_group(image, col_start, col_end):
    black_points = []
    for col in range(col_start, col_end):
        for row in range(image.shape[0]):  # Iterate over all rows
            if image[row, col] == 0:  # Black pixel (value 0)
                black_points.append([col, row])
    return np.array(black_points)

# Function to draw convex hull over the image
def draw_convex_hull(image, hull_points):
    for i in range(len(hull_points)):
        # Draw line between consecutive points
        start_point = tuple(hull_points[i])
        end_point = tuple(
            hull_points[(i + 1) % len(hull_points)]
        )  # Wrap around to the first point
        cv2.line(
            image, start_point, end_point, (0, 255, 0), 5
        )  # Green line for convex hull


# Function to draw black points on the image
def draw_black_points(image, black_points):
    for point in black_points:
        cv2.circle(
            image, tuple(point), 2, (0, 0, 0), -1
        )  # Red dots for black points


# Main function to process the image
def main(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Could not load image.")
        return

    # Make a copy to draw convex hulls and points on
    image_with_hull = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Process image in sets of 25-pixel columns
    col_group_size = 25
    height, width = image.shape
    all_black_points = []

    for col_start in range(0, width, col_group_size):
        col_end = min(
            col_start + col_group_size, width
        )  # Ensure we don't go beyond image width

        # Get black points in the current column group
        black_points = get_black_points_in_group(image, col_start, col_end)
        all_black_points.extend(
            black_points
        )  # Collect all points in a single array

    all_black_points = np.array(all_black_points)

    # Draw the black points on the image
    if len(all_black_points) > 0:
        draw_black_points(image_with_hull, all_black_points)

    # If there are enough points, compute the convex hull
    if len(all_black_points) > 2:  # Convex hull requires at least 3 points
        try:
            hull = ConvexHull(all_black_points)
            hull_points = all_black_points[hull.vertices]  # Get hull points

            # Draw the convex hull over the image
            print(hull_points)
            draw_convex_hull(image_with_hull, hull_points)
        except QhullError:
            print(
                f"Skipping convex hull calculation due to precision error (coplanar points)."
            )

    # Show the image with the convex hull and points drawn
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image_with_hull, cv2.COLOR_BGR2RGB))
    plt.title("Convex Hull and Black Points Over the Image")
    plt.axis("off")
    plt.savefig("hullss.png")
    plt.show()
    return hull_points


# Example usage
if __name__ == "__main__":
    image_path = "monaco.png"
    main(image_path)
