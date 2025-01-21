from sklearn.cluster import KMeans

# Class to assign players to teams based on their colors in a frame
class TeamAssigner:
    def __init__(self):
        """
        Initializes the TeamAssigner class.
        - team_colors: Dictionary to store the representative colors for each team.
        - player_team_dict: Dictionary to store the assigned team for each player.
        """
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        """
        Perform K-means clustering on the given image.

        Args:
        - image: A 2D image array of shape (height, width, 3).

        Returns:
        - kmeans: A fitted KMeans clustering model with 2 clusters.
        """
        # Reshape the image into a 2D array where each row represents a pixel (R, G, B)
        image_2d = image.reshape(-1, 3)

        # Perform K-means clustering with 2 clusters (assumes two teams)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extract the dominant player color within the bounding box.

        Args:
        - frame: The full image frame.
        - bbox: Bounding box coordinates (x_min, y_min, x_max, y_max).

        Returns:
        - player_color: The RGB color representing the player's cluster.
        """
        # Crop the bounding box region from the frame
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Focus on the top half of the cropped image to exclude legs
        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Get clustering model for the cropped image
        kmeans = self.get_clustering_model(top_half_image)

        # Get cluster labels for each pixel
        labels = kmeans.labels_

        # Reshape the labels back into the original image shape
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Determine the non-player cluster by examining corner pixels
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)

        # Assume the player cluster is the other cluster
        player_cluster = 1 - non_player_cluster

        # Get the RGB value of the player's cluster centroid
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        """
        Assign representative colors to teams based on player detections.

        Args:
        - frame: The full image frame.
        - player_detections: Dictionary containing player detections with bounding boxes.
        """
        player_colors = []

        # Extract player colors for each detected player
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Perform K-means clustering on player colors to group them into two teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        # Store the clustering model and representative team colors
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """
        Determine the team ID for a given player based on their color.

        Args:
        - frame: The full image frame.
        - player_bbox: Bounding box of the player (x_min, y_min, x_max, y_max).
        - player_id: Unique identifier for the player.

        Returns:
        - team_id: The assigned team ID (1 or 2).
        """
        # Check if the player ID is already assigned to a team
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Extract the player's color from the frame
        player_color = self.get_player_color(frame, player_bbox)

        # Predict the team ID using the clustering model
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # Convert cluster index to team ID (1 or 2)

        # Hard coding goalkeeper value
        # Force player ID 91 to be in team 1 (special case)
        if player_id == 81: # Player ID of GoalKeeper
            team_id = 1

        # Save the assigned team ID for future reference
        self.player_team_dict[player_id] = team_id

        return team_id

