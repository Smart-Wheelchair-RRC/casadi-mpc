# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "pyqt6",
#     "rosbags",
# ]
# ///


import numpy as np


def get_circle_locations_from_occupancy_map(
    map: np.ndarray,
    sector_angle: int = 5,
    field_of_view: int = 240,
    max_circles: int = 40,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Get lines from occupancy map that cover the frontier assuming the center to be ego agent.
    :param map: Occupancy map.
    :return: List of lines.
    """
    circles = []

    # Get the center of the occupancy map
    center_x = map.shape[1] // 2
    center_y = map.shape[0] // 2

    # Convert all occupied cell coordinates to polar form
    occupied_cells = np.argwhere(map == 100)
    occupied_cells_polar = occupied_cells - np.array([center_y, center_x])
    occupied_cells_polar = occupied_cells_polar.astype(float)
    occupied_cells_polar_angles = np.arctan2(
        occupied_cells_polar[:, 0], occupied_cells_polar[:, 1]
    )

    occupied_cells_polar_distances = np.linalg.norm(occupied_cells_polar, axis=1)
    occupied_cells_polar = np.column_stack(
        (occupied_cells_polar_angles, occupied_cells_polar_distances)
    )
    occupied_cells_polar = occupied_cells_polar[
        np.argsort(occupied_cells_polar[:, 0])
    ]  # Sort by angle

    # Get the angle of the ego agent
    # Assuming the ego agent is facing east, the angle is 0 degrees
    ego_angle = 90  # degrees
    # Convert to radians
    ego_angle_rad = np.deg2rad(ego_angle)

    occupied_cells_polar[:, 0] = (occupied_cells_polar[:, 0] - ego_angle_rad) % (
        2 * np.pi
    )  # Normalize to [0, 2*pi]

    # Ensure angles more than 180 degrees are negative
    occupied_cells_polar[:, 0] = np.where(
        occupied_cells_polar[:, 0] > np.pi,
        occupied_cells_polar[:, 0] - 2 * np.pi,
        occupied_cells_polar[:, 0],
    )

    # Filter those points that are in the 30 degree sector behind the ego agent

    points_in_viewport = occupied_cells_polar[
        (occupied_cells_polar[:, 0] >= np.deg2rad(-120))
        & (occupied_cells_polar[:, 0] <= np.deg2rad(120))
    ]

    total_points = points_in_viewport.shape[0]
    if total_points == 0:
        return []

    # Each sector gets minimum 1 circle, and the rest are distributed based on point density

    # Divide the points into 30 degree sectors
    for i in range(-field_of_view // 2, field_of_view // 2, int(sector_angle)):
        # Get the points in the sector
        sector = points_in_viewport[
            (points_in_viewport[:, 0] >= np.deg2rad(i))
            & (points_in_viewport[:, 0] < np.deg2rad(i + sector_angle))
        ]

        # Closest point in the sector
        sector_size = sector.shape[0]
        if sector_size > 0:
            number_of_circles = max(1, int((sector_size / total_points) * max_circles))

            # Get the closest n points in the sector
            closest_points = sector[np.argsort(sector[:, 1])[:number_of_circles]]

            # Convert back to cartesian coordinates
            closest_points_cartesian = (
                closest_points[:, 1] * np.cos(closest_points[:, 0] + ego_angle_rad)
                + center_x,
                closest_points[:, 1] * np.sin(closest_points[:, 0] + ego_angle_rad)
                + center_y,
            )

            circles += list(
                zip(
                    closest_points_cartesian[0],
                    closest_points_cartesian[1],
                )
            )

    return circles


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from rosbags.rosbag2 import Reader  # type:ignore[import]
    from rosbags.typesys import Stores, get_typestore  # type:ignore[import]

    # Create a typestore and get the string class.
    typestore = get_typestore(Stores.LATEST)

    # Create reader instance and open for reading.
    with Reader("rosbag2_2025_04_07-03_27_33_0") as reader:
        # Topic and msgtype information is available on .connections list.
        connections = [
            x for x in reader.connections if x.topic == "/local_costmap/costmap"
        ]
        fig, ax = plt.subplots()

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            # Visualize the occupancy map message of type /local_costmap/costmap nav_msgs/msg/OccupancyGrid using matplotlib
            print(f"Timestamp: {timestamp}")

            occupancy_map_width = msg.info.width
            occupancy_map_height = msg.info.height
            occupancy_map = np.array(msg.data).reshape(
                occupancy_map_height, occupancy_map_width
            )

            circles = get_circle_locations_from_occupancy_map(occupancy_map)

            print(f"Occupancy Map Shape: {occupancy_map.shape}")

            ax.clear()  # Clear the axis for the next iteration
            # Display the occupancy map using imshow
            im = ax.imshow(occupancy_map, cmap="gray", interpolation="nearest")

            # Visualize the lines
            for circle in circles:
                # Draw the lines on the occupancy map
                circle_x, circle_y = circle
                ax.plot(
                    circle_x,
                    circle_y,
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor="red",
                    markersize=10,
                )

            # Set axis labels
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

            # Set title
            ax.set_title("Occupancy Map")

            # Show the plot
            plt.pause(0.5)
