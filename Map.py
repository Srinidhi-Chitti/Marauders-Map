class MaraudersMapHandler:
    def __init__(self, map_widget):
        """Handles map-related operations"""
        self.map_widget = map_widget
        self.markers = {}  # Stores markers for easy reference
        self.paths = []  # Stores paths to manage drawn paths

    def add_marker(self, name, lat, lon, color="gray"):
        """
        Adds a student marker on the map.
        
        :param name: Name of the student
        :param lat: Latitude coordinate
        :param lon: Longitude coordinate
        :param color: Optional color for marker (default: gray)
        :return: The created marker objectA
        """
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            print(f"Invalid coordinates for {name}: ({lat}, {lon})")
            return None

        marker = self.map_widget.set_marker(lat, lon, text=name)
        self.markers[name] = marker  # Store marker for future updates
        return marker

    def update_marker(self, name, lat, lon):
        """
        Updates the position of an existing marker.
        
        :param name: Student name
        :param lat: New latitude
        :param lon: New longitude
        """
        if name in self.markers:
            self.markers[name].set_position(lat, lon)
        else:
            print(f"Marker for {name} not found, creating a new one.")
            self.add_marker(name, lat, lon)

    def draw_path(self, start_coords, end_coords, color="blue"):
        """
        Draws a path between two points.
        
        :param start_coords: (lat, lon) of the starting point
        :param end_coords: (lat, lon) of the ending point
        :param color: Path color (default: blue)
        :return: The created path object
        """
        if not (isinstance(start_coords, tuple) and isinstance(end_coords, tuple)):
            print(f"Invalid path coordinates: {start_coords} -> {end_coords}")
            return None

        if start_coords and end_coords and len(start_coords) == 2 and len(end_coords) == 2:
            path = self.map_widget.set_path([start_coords, end_coords], color=color)
            self.paths.append(path)  # Store path for management
            return path
        else:
            print("Path could not be drawn due to invalid coordinates.")
            return None
