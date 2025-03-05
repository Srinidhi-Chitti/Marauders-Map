from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QMenuBar, QAction, QGroupBox, QMessageBox
)
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
import os
import folium
import random
import requests
from collections import deque
from queue import PriorityQueue

# Define the graph, heuristic, and node coordinates
london_harry_potter_graph = {
    "King's Cross Station": {"Leadenhall Market": 3.5, "St. Pancras Renaissance Hotel": 0.2, "Diagon Alley": 1.8},
    "Leadenhall Market": {"Millennium Bridge": 2.5, "Australia House": 1.8, "Piccadilly Circus": 1.8, "Diagon Alley": 2.0},
    "Millennium Bridge": {"Westminster Station": 2.2, "Lambeth Bridge": 1.5, "London Eye": 1.0},
    "Australia House": {"Piccadilly Circus": 1.2, "Westminster Station": 2.0, "Leadenhall Market": 1.8},
    "Piccadilly Circus": {"Westminster Station": 1.0, "Leadenhall Market": 1.8, "Australia House": 1.2, "Oxford Street": 1.5},
    "Westminster Station": {"Lambeth Bridge": 1.3, "Australia House": 2.0, "Millennium Bridge": 2.2, "Big Ben": 0.5},
    "Lambeth Bridge": {"Millennium Bridge": 1.5, "Westminster Station": 1.3, "London Eye": 1.2},
    "St. Pancras Renaissance Hotel": {"King's Cross Station": 0.2, "British Library": 0.8},
    "Diagon Alley": {"King's Cross Station": 1.8, "Leadenhall Market": 2.0, "Gringotts Bank": 0.5},
    "Gringotts Bank": {"Diagon Alley": 0.5, "London Eye": 1.8},
    "London Eye": {"Millennium Bridge": 1.0, "Lambeth Bridge": 1.2, "Gringotts Bank": 1.8, "Big Ben": 0.8},
    "Big Ben": {"Westminster Station": 0.5, "London Eye": 0.8, "Houses of Parliament": 0.3},
    "Houses of Parliament": {"Big Ben": 0.3, "Westminster Abbey": 0.6},
    "Westminster Abbey": {"Houses of Parliament": 0.6, "St. James's Park": 0.9},
    "St. James's Park": {"Westminster Abbey": 0.9, "Buckingham Palace": 1.2},
    "Buckingham Palace": {"St. James's Park": 1.2, "Green Park": 0.7},
    "Green Park": {"Buckingham Palace": 0.7, "Piccadilly Circus": 1.0},
    "Oxford Street": {"Piccadilly Circus": 1.5, "Tottenham Court Road": 0.8},
    "Tottenham Court Road": {"Oxford Street": 0.8, "British Museum": 0.6},
    "British Museum": {"Tottenham Court Road": 0.6, "Russell Square": 0.5},
    "Russell Square": {"British Museum": 0.5, "King's Cross Station": 1.0},
    "British Library": {"St. Pancras Renaissance Hotel": 0.8, "King's Cross Station": 0.7},
}

heuristic = {
    "King's Cross Station": 0,
    "Leadenhall Market": 3.2,
    "Millennium Bridge": 2.8,
    "Australia House": 2.5,
    "Piccadilly Circus": 2.1,
    "Westminster Station": 3.0,
    "Lambeth Bridge": 3.5,
    "St. Pancras Renaissance Hotel": 0.2,
    "Diagon Alley": 1.5,
    "Gringotts Bank": 1.8,
    "London Eye": 2.0,
    "Big Ben": 2.2,
    "Houses of Parliament": 2.3,
    "Westminster Abbey": 2.4,
    "St. James's Park": 2.5,
    "Buckingham Palace": 2.6,
    "Green Park": 2.7,
    "Oxford Street": 2.8,
    "Tottenham Court Road": 2.9,
    "British Museum": 3.0,
    "Russell Square": 3.1,
    "British Library": 0.5,
}

node_coordinates = {
    "King's Cross Station": (51.5324, -0.1260),
    "Leadenhall Market": (51.5136, -0.0836),
    "St. Pancras Renaissance Hotel": (51.5305, -0.1253),
    "Millennium Bridge": (51.5104, -0.0985),
    "Australia House": (51.5132, -0.1189),
    "Piccadilly Circus": (51.5101, -0.1340),
    "Westminster Station": (51.5012, -0.1249),
    "Lambeth Bridge": (51.4958, -0.1217),
    "Diagon Alley": (51.5225, -0.1117),
    "Gringotts Bank": (51.5155, -0.1080),
    "London Eye": (51.5033, -0.1195),
    "Big Ben": (51.5007, -0.1246),
    "Houses of Parliament": (51.4995, -0.1248),
    "Westminster Abbey": (51.4994, -0.1273),
    "St. James's Park": (51.5024, -0.1319),
    "Buckingham Palace": (51.5014, -0.1419),
    "Green Park": (51.5067, -0.1428),
    "Oxford Street": (51.5154, -0.1415),
    "Tottenham Court Road": (51.5165, -0.1309),
    "British Museum": (51.5194, -0.1269),
    "Russell Square": (51.5230, -0.1245),
    "British Library": (51.5302, -0.1277),
}

class MaraudersMap(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Marauder's Map")
        self.setGeometry(100, 100, 1024, 768)
        self.setStyleSheet("""
            background-color: #F1E9D2;
            color: #2c2c2c;
            QGroupBox {
                border: 2px solid #6b4c2a;
                border-radius: 5px;
                margin-top: 1ex;
                background-color: #E8E0C9;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
                color: #6b4c2a;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #6b4c2a;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8b6d4b;
            }
            QComboBox {
                background-color: #E8E0C9;
                color: #2c2c2c;
                padding: 5px;
                border-radius: 5px;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QLabel {
                color: #2c2c2c;
                font-size: 14px;
            }
        """)

        self.map_view = None
        self.map_file = "map.html"
        self.characters = []
        self.graph = london_harry_potter_graph
        self.heuristic = heuristic
        self.node_coordinates = node_coordinates

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)

        # Left Panel
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.create_logo_group())
        left_layout.addWidget(self.create_algorithm_controls())
        left_layout.addWidget(self.create_character_controls())
        main_layout.addLayout(left_layout, stretch=1)

        # Right Panel (Map)
        self.map_view = QWebEngineView()
        self.init_map()
        main_layout.addWidget(self.map_view, stretch=3)

        self.add_menu()

    def add_menu(self):
        menubar = QMenuBar(self)
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #3c3c3c;
                color: #e0e0e0;
                font-size: 14px;
                font-weight: bold;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 5px 10px;
            }
            QMenuBar::item:selected {
                background-color: #8b4513;
            }
            QMenu {
                background-color: #3c3c3c;
                color: #e0e0e0;
                border: 1px solid #8b4513;
            }
            QMenu::item:selected {
                background-color: #8b4513;
            }
        """)
        file_menu = menubar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        self.setMenuBar(menubar)

    def create_logo_group(self):
        logo_group = QGroupBox()
        logo_layout = QVBoxLayout()

        # Quotes
        quote1 = QLabel("I solemnly swear I am up to no good")
        quote1.setAlignment(Qt.AlignCenter)
        quote1.setFont(QFont("Cambria", 14, QFont.Bold))
        quote1.setStyleSheet("color: #8b4513;")
        
        quote2 = QLabel("Moony, Padfoot, Prongs & Wormtail\nproudly present you the\nMarauder's Map")
        quote2.setAlignment(Qt.AlignCenter)
        quote2.setFont(QFont("Cambria", 12))

        # Logos (replace with actual images if available)
        logo_row = QHBoxLayout()
        logo1 = QLabel()
        logo1.setPixmap(QPixmap("hogwarts_logo.png").scaled(200, 150, QtCore.Qt.KeepAspectRatio))
        logo2 = QLabel()
        logo2.setPixmap(QPixmap("hogwarts_icon.jpg").scaled(200, 150, QtCore.Qt.KeepAspectRatio))

        logo_row.addWidget(logo1)
        logo_row.addWidget(logo2)

        logo_layout.addWidget(quote1)
        logo_layout.addWidget(quote2)
        logo_layout.addLayout(logo_row)

        logo_group.setLayout(logo_layout)
        return logo_group

    def create_algorithm_controls(self):
        control_group = QGroupBox("Pathfinding Algorithms")
        control_layout = QVBoxLayout()

        self.start_combo = QComboBox()
        self.end_combo = QComboBox()
        self.algorithm_combo = QComboBox()
        algorithms = ["BFS", "DFS", "UCS", "Hill Climbing", "A*", "Bi_directional_search", "Greedy_best_first_search", "depth_limited_search", "iterative_deepening_dfs", "recursive_best_first_search", "minimax", "Backtracking"]
        self.algorithm_combo.addItems(algorithms)

        # Populate combos with graph nodes
        self.start_combo.addItems(self.graph.keys())
        self.end_combo.addItems(self.graph.keys())

        control_layout.addWidget(QLabel("Start Node:"))
        control_layout.addWidget(self.start_combo)
        control_layout.addWidget(QLabel("End Node:"))
        control_layout.addWidget(self.end_combo)
        control_layout.addWidget(self.algorithm_combo)
        
        self.route_button = QPushButton("Show Route")
        self.route_button.clicked.connect(self.show_route)
        control_layout.addWidget(self.route_button)

        control_group.setLayout(control_layout)
        return control_group

    def create_character_controls(self):
        control_group = QGroupBox("Character Tracking")
        control_layout = QVBoxLayout()

        self.char_button = QPushButton("Show Characters")
        self.char_button.clicked.connect(self.fetch_characters)

        control_layout.addWidget(self.char_button)
        control_group.setLayout(control_layout)
        return control_group

    def init_map(self):
        folium_map = folium.Map(location=(51.5324, -0.1260), zoom_start=14)
        folium_map.save(self.map_file)
        self.map_view.setUrl(QUrl.fromLocalFile(os.path.abspath(self.map_file)))

    def show_route(self):
        algorithm = self.algorithm_combo.currentText()
        start_node = self.start_combo.currentText()
        end_node = self.end_combo.currentText()

        if start_node not in self.graph or end_node not in self.graph:
            QMessageBox.warning(self, "Error", "Invalid start or end node!")
            return

        path = []
        if algorithm == "BFS":
            path = self.breadth_first_search(start_node, end_node)
        elif algorithm == "DFS":
            path = self.depth_first_search(start_node, end_node)
        elif algorithm == "UCS":
            path = self.uniform_cost_search(start_node, end_node)
        elif algorithm == "Hill Climbing":
            path = self.hill_climbing(start_node, end_node)
        elif algorithm == "A*":
            path = self.a_star_search(start_node, end_node)
        elif algorithm == "Bi_directional_search":
            path = self.bi_directional_search(start_node, end_node)
        elif algorithm == "Greedy_best_first_search":
            path = self.greedy_best_first_search(start_node, end_node)
        elif algorithm == "depth_limited_search":
            path = self.depth_limited_search(start_node, end_node, limit=3)
        elif algorithm == "iterative_deepening_dfs":
            path = self.iterative_deepening_dfs(start_node, end_node, max_depth=5)
        elif algorithm == "recursive_best_first_search":
            path = self.recursive_best_first_search(start_node, end_node)
        elif algorithm == "minimax":
            path = self.minimax(start_node, end_node)
        elif algorithm == "Backtracking":
            path = self.backtracking_csp(start_node, end_node)

        if not path:
            QMessageBox.information(self, "No Path", "No path found!")
            return

        # Convert nodes to coordinates
        path_coords = []
        for node in path:
            if node in self.node_coordinates:
                path_coords.append(self.node_coordinates[node])
            else:
                QMessageBox.warning(self, "Error", f"Coordinates for {node} not found!")
                return

        # Calculate total distance
        total_distance = 0.0
        for i in range(len(path)-1):
            if path[i+1] in self.graph[path[i]]:
                total_distance += self.graph[path[i]][path[i+1]]
            else:
                QMessageBox.warning(self, "Error", f"No connection between {path[i]} and {path[i+1]}!")
                return

        # Create map
        folium_map = folium.Map(location=self.node_coordinates[start_node], zoom_start=14)
        
        # Add path
        folium.PolyLine(
            path_coords,
            color="#6b4c2a",
            weight=2.5,
            opacity=0.8,
            tooltip=f"{algorithm} Path: {total_distance:.2f} km"
        ).add_to(folium_map)
        
        # Add markers
        folium.Marker(
            self.node_coordinates[start_node],
            icon=folium.Icon(color="green", icon="flag"),
            popup=f"Start: {start_node}"
        ).add_to(folium_map)
        
        folium.Marker(
            self.node_coordinates[end_node],
            icon=folium.Icon(color="red", icon="flag"),
            popup=f"End: {end_node}"
        ).add_to(folium_map)

        folium_map.save(self.map_file)
        self.map_view.setUrl(QUrl.fromLocalFile(os.path.abspath(self.map_file)))

    def breadth_first_search(self, start, goal):
        visited = set()
        queue = deque([(start, [start])])
        visited.add(start)
        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path
            for neighbor in self.graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []

    def depth_first_search(self, start, goal):
        stack = [(start, [start])]
        visited = set()
        while stack:
            current, path = stack.pop()
            if current == goal:
                return path
            if current not in visited:
                visited.add(current)
                # Randomize the order of neighbors to ensure different paths
                neighbors = list(self.graph[current].keys())
                random.shuffle(neighbors)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))
        return []

    def uniform_cost_search(self, start, goal):
        pq = PriorityQueue()
        pq.put((0, start, [start]))
        visited = set()
        while not pq.empty():
            cost, current, path = pq.get()
            if current == goal:
                return path
            if current not in visited:
                visited.add(current)
                for neighbor, weight in self.graph[current].items():
                    if neighbor not in visited:
                        pq.put((cost + weight, neighbor, path + [neighbor]))
        return []

    def hill_climbing(self, start, goal):
        current = start
        path = [current]
        visited = set([current])  # Track visited nodes to avoid loops

        while current != goal:
            neighbors = list(self.graph[current].keys())
            if not neighbors:
                break  # No more neighbors to explore

            # Sort neighbors by heuristic value (lowest first)
            neighbors.sort(key=lambda x: self.heuristic[x])

            # Choose the best neighbor (lowest heuristic)
            best_neighbor = None
            for neighbor in neighbors:
                if neighbor not in visited:
                    best_neighbor = neighbor
                    break

            if best_neighbor is None:
                # If no unvisited neighbors, backtrack
                if len(path) > 1:
                    current = path[-2]  # Move back to the previous node
                    path.pop()
                else:
                    return []  # No path found
            else:
                # Move to the best neighbor
                current = best_neighbor
                path.append(current)
                visited.add(current)

        return path if current == goal else []

    def a_star_search(self, start, goal):
        pq = PriorityQueue()
        pq.put((0 + self.heuristic[start], 0, start, [start]))
        visited = set()
        while not pq.empty():
            _, cost, current, path = pq.get()
            if current == goal:
                return path
            if current not in visited:
                visited.add(current)
                for neighbor, weight in self.graph[current].items():
                    if neighbor not in visited:
                        new_cost = cost + weight
                        pq.put((
                            new_cost + self.heuristic[neighbor],
                            new_cost,
                            neighbor,
                            path + [neighbor]
                        ))
        return []
    
    def bi_directional_search(self, start, goal):
        front_queue = {start: [start]}
        back_queue = {goal: [goal]}
        while front_queue and back_queue:
            new_front = {}
            for node, path in front_queue.items():
                for neighbor in self.graph[node]:
                    if neighbor in back_queue:
                        return path + back_queue[neighbor][::-1]
                    if neighbor not in new_front:
                        new_front[neighbor] = path + [neighbor]
            front_queue = new_front

            new_back = {}
            for node, path in back_queue.items():
                for neighbor in self.graph[node]:
                    if neighbor in front_queue:
                        return front_queue[neighbor] + path[::-1]
                    if neighbor not in new_back:
                        new_back[neighbor] = path + [neighbor]
            back_queue = new_back
        return []

    def greedy_best_first_search(self, start, goal):
        pq = PriorityQueue()
        pq.put((self.heuristic[start], start, [start]))
        while not pq.empty():
            _, node, path = pq.get()
            if node == goal:
                return path
            for neighbor in self.graph[node]:
                if neighbor not in path:
                    pq.put((self.heuristic[neighbor], neighbor, path + [neighbor]))
        return []

    def depth_limited_search(self, start, goal, limit, path=None):
        if path is None:
            path = [start]
        if start == goal:
            return path
        if limit <= 0:
            return []
        for neighbor in self.graph[start]:
            if neighbor not in path:
                new_path = self.depth_limited_search(neighbor, goal, limit - 1, path + [neighbor])
                if new_path:
                    return new_path
        return []

    def iterative_deepening_dfs(self, start, goal, max_depth):
        for depth in range(max_depth + 1):
            result = self.depth_limited_search(start, goal, depth)
            if result:
                return result
        return []

    def recursive_best_first_search(self, start, goal, path=None, f_limit=float('inf'), depth_limit=100):
        if path is None:
            path = [start]
        if start == goal:
            return path
        if depth_limit <= 0:
            return []
        successors = sorted(self.graph[start], key=lambda n: self.heuristic[n])
        if not successors:
            return []
        best = successors[0]
        best_heuristic = self.heuristic[best]
        if best_heuristic > f_limit:
            return []
        alternative_heuristic = float('inf')
        if len(successors) > 1:
            alternative_heuristic = self.heuristic[successors[1]]
        result = self.recursive_best_first_search(
            best, goal, path + [best], min(f_limit, alternative_heuristic), depth_limit - 1)
        if result:
            return result
        if len(successors) > 1:
            alternative = successors[1]
        alternative_heuristic = self.heuristic[alternative]
        result = self.recursive_best_first_search(
            alternative, goal, path + [alternative], f_limit, depth_limit - 1)
        if result:
            return result
        return []

    def minimax(self, start, goal):
        # This is a placeholder for the minimax algorithm, which is typically used in adversarial search
        # For pathfinding, it doesn't make much sense, so we'll just return a simple path
        return self.depth_first_search(start, goal)

    def backtracking_csp(self, start, goal):
        # This is a placeholder for the backtracking algorithm, which is typically used in constraint satisfaction problems
        # For pathfinding, it doesn't make much sense, so we'll just return a simple path
        return self.depth_first_search(start, goal)

    def fetch_characters(self):
        try:
            response = requests.get("https://hp-api.onrender.com/api/characters")
            self.characters = response.json()
            self.show_characters()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch characters: {str(e)}")

    def show_characters(self):
        folium_map = folium.Map(location=(51.5324, -0.1260), zoom_start=12)
        for character in self.characters:
            if character.get("name"):
                lat = 51.5324 + random.uniform(-0.01, 0.01)
                lon = -0.1260 + random.uniform(-0.01, 0.01)
                html = f"""
                    <b>{character['name']}</b><br>
                    House: {character.get('house', 'Unknown')}<br>
                    Ancestry: {character.get('ancestry', 'Unknown')}
                """
                folium.Marker(
                    [lat, lon],
                    popup=html,
                    icon=folium.Icon(
                        color="purple" if character.get("hogwartsStudent") else "orange",
                        icon="user" if character.get("hogwartsStudent") else "book"
                    )
                ).add_to(folium_map)
        folium_map.save(self.map_file)
        self.map_view.setUrl(QUrl.fromLocalFile(os.path.abspath(self.map_file)))

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MaraudersMap()
    window.showMaximized()
    sys.exit(app.exec_())