from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QMenuBar, QAction, QGroupBox, QMessageBox
)
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtWebEngineWidgets import QWebEngineView
import os
import folium
import random
import requests
import math
from math import radians, sin, cos, sqrt, atan2
from queue import PriorityQueue, Queue
from collections import deque

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
        self.start_point = (51.5324, -0.1260)  # Central London

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

        # Logos
        logo_row = QHBoxLayout()
        logo1 = QLabel()
        pixmap1 = QPixmap("hogwarts_logo.png").scaled(200, 150, QtCore.Qt.KeepAspectRatio)
        logo1.setPixmap(pixmap1)

        logo2 = QLabel()
        pixmap2 = QPixmap("hogwarts_icon.jpg").scaled(200, 150, QtCore.Qt.KeepAspectRatio)
        logo2.setPixmap(pixmap2)

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

        self.algorithm_combo = QComboBox()
        algorithms = [
            "BFS", "DFS", "UCS", 
            "Hill Climbing", "A*", 
            "Iterative Best First", "Least Cost"
        ]
        self.algorithm_combo.addItems(algorithms)
        
        self.route_button = QPushButton("Show Route")
        self.route_button.clicked.connect(self.show_route)

        control_layout.addWidget(self.algorithm_combo)
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
    def generate_path(self, algorithm):
        """Generate algorithm-specific path starting and ending at start_point"""
        path = [self.start_point]
        delta = 0.02  # Base coordinate delta
        
        if algorithm == "BFS":
            # Rectangular path
            path.extend([
                (self.start_point[0] + delta, self.start_point[1]),
                (self.start_point[0] + delta, self.start_point[1] + delta),
                (self.start_point[0], self.start_point[1] + delta),
                self.start_point
            ])
        elif algorithm == "DFS":
            # Spiral path
            for i in range(1, 6):
                angle = math.radians(72 * i)
                radius = 0.01 * i
                path.append((
                    self.start_point[0] + radius * math.cos(angle),
                    self.start_point[1] + radius * math.sin(angle)
                ))
            path.append(self.start_point)
        elif algorithm == "A*":
            # Star pattern
            for i in range(5):
                angle = math.radians(72 * i)
                path.append((
                    self.start_point[0] + delta * math.cos(angle),
                    self.start_point[1] + delta * math.sin(angle)
                ))
            path.append(self.start_point)
        else:  # Default circular path
            for i in range(0, 360, 36):
                angle = math.radians(i)
                path.append((
                    self.start_point[0] + delta * math.cos(angle),
                    self.start_point[1] + delta * math.sin(angle)
                ))
            path.append(self.start_point)
        
        return path

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        a = sin(dLat/2)*2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dLon/2)*2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    def calculate_path_distance(self, path):
        total = 0.0
        for i in range(len(path)-1):
            total += self.haversine(*path[i], *path[i+1])
        return round(total, 2)

    def show_route(self):
        algorithm = self.algorithm_combo.currentText()
        folium_map = folium.Map(location=self.start_point, zoom_start=14)
        path = self.generate_path(algorithm)
        distance = self.calculate_path_distance(path)
        
        folium.PolyLine(
            path, 
            color="#6b4c2a", 
            weight=2.5, 
            opacity=0.8,
            tooltip=f"{algorithm} Path: {distance} km"
        ).add_to(folium_map)
        
        folium.Marker(
            self.start_point,
            popup=f"Start/End ({algorithm})",
            icon=folium.Icon(color="green", icon="flag")
        ).add_to(folium_map)

        folium_map.save(self.map_file)
        self.map_view.setUrl(QUrl.fromLocalFile(os.path.abspath(self.map_file)))

    def init_map(self):
        folium_map = folium.Map(location=[51.5324, -0.1260], zoom_start=12)  # London area
        folium_map.save(self.map_file)
        self.map_view.setUrl(QUrl.fromLocalFile(os.path.abspath(self.map_file)))

    def show_route(self):
        algorithm = self.algorithm_combo.currentText()
        folium_map = folium.Map(location=[51.5324, -0.1260], zoom_start=12)
        path = self.generate_fake_path()
        
        folium.PolyLine(path, color="blue", weight=2.5, opacity=1, 
                        tooltip=f"{algorithm} Path").add_to(folium_map)
        
        folium.Marker(
            path[0], 
            popup=f"Start ({algorithm})", 
            icon=folium.Icon(color="green", icon="flag")
        ).add_to(folium_map)
        
        folium.Marker(
            path[-1], 
            popup=f"End ({algorithm})", 
            icon=folium.Icon(color="red", icon="flag")
        ).add_to(folium_map)

        folium_map.save(self.map_file)
        self.map_view.setUrl(QUrl.fromLocalFile(os.path.abspath(self.map_file)))

    def fetch_characters(self):
        try:
            response = requests.get("https://hp-api.onrender.com/api/characters")
            self.characters = response.json()
            self.show_characters()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to fetch characters: {str(e)}")

    def show_characters(self):
        folium_map = folium.Map(location=[51.5324, -0.1260], zoom_start=12)
        
        for character in self.characters:
            if character["name"]:
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
                        color="purple" if character["hogwartsStudent"] else "orange",
                        icon="user" if character["hogwartsStudent"] else "book"
                    )
                ).add_to(folium_map)

        folium_map.save(self.map_file)
        self.map_view.setUrl(QUrl.fromLocalFile(os.path.abspath(self.map_file)))

    def generate_fake_path(self):
        start_lat, start_lon = 51.5324, -0.1260
        return [
            (start_lat + random.uniform(-0.01, 0.01), 
             start_lon + random.uniform(-0.01, 0.01))
            for _ in range(10)
        ]
    
    def breadth_first_search(graph, start, goal):
        queue = deque([(start, [start])])
        while queue:
            node, path = queue.popleft()
        if node == goal:
            return path
        for neighbor in graph[node]:
            if neighbor not in path:
                queue.append((neighbor, path + [neighbor]))
                return []

def depth_first_search(graph, start, goal, path=None):
    if path is None:
        path = [start]
    if start == goal:
        return path
    for neighbor in graph[start]:
        if neighbor not in path:
            new_path = depth_first_search(graph, neighbor, goal, path + [neighbor])
            if new_path:
                return new_path
    return []

def uniform_cost_search(graph, start, goal):
    pq = PriorityQueue()
    pq.put((0, start, [start]))
    while not pq.empty():
        cost, node, path = pq.get()
        if node == goal:
            return path
        for neighbor, weight in graph[node].items():
            if neighbor not in path:
                pq.put((cost + weight, neighbor, path + [neighbor]))
    return []

def a_star_search(graph, start, goal, heuristic):
    pq = PriorityQueue()
    pq.put((0, start, [start]))
    while not pq.empty():
        cost, node, path = pq.get()
        if node == goal:
            return path
        for neighbor, weight in graph[node].items():
            if neighbor not in path:
                pq.put((cost + weight + heuristic[neighbor], neighbor, path + [neighbor]))
    return []

def bi_directional_search(graph, start, goal):
    front_queue = {start: [start]}
    back_queue = {goal: [goal]}
    while front_queue and back_queue:
        new_front = {}
        for node, path in front_queue.items():
            for neighbor in graph[node]:
                if neighbor in back_queue:
                    return path + back_queue[neighbor][::-1]
                if neighbor not in new_front:
                    new_front[neighbor] = path + [neighbor]
        front_queue = new_front

        new_back = {}
        for node, path in back_queue.items():
            for neighbor in graph[node]:
                if neighbor in front_queue:
                    return front_queue[neighbor] + path[::-1]
                if neighbor not in new_back:
                    new_back[neighbor] = path + [neighbor]
        back_queue = new_back
    return []

def greedy_best_first_search(graph, start, goal, heuristic):
    pq = PriorityQueue()
    pq.put((heuristic[start], start, [start]))
    while not pq.empty():
        _, node, path = pq.get()
        if node == goal:
            return path
        for neighbor in graph[node]:
            if neighbor not in path:
                pq.put((heuristic[neighbor], neighbor, path + [neighbor]))
    return []

def depth_limited_search(graph, start, goal, limit, path=None):
    if path is None:
        path = [start]
    if start == goal:
        return path
    if limit <= 0:
        return []
    for neighbor in graph[start]:
        if neighbor not in path:
            new_path = depth_limited_search(graph, neighbor, goal, limit - 1, path + [neighbor])
            if new_path:
                return new_path
    return []

def hill_climbing(graph, start, goal, heuristic):
    current = start
    path = [start]
    while current != goal:
        neighbors = sorted(graph[current], key=lambda n: heuristic[n])
        if not neighbors or heuristic[neighbors[0]] >= heuristic[current]:
            return []
        current = neighbors[0]
        path.append(current)
    return path

def iterative_deepening_dfs(graph, start, goal, max_depth):
    for depth in range(max_depth + 1):
        result = depth_limited_search(graph, start, goal, depth)
        if result:
            return result
    return []

def recursive_best_first_search(graph, start, goal, heuristic, path=None, f_limit=float('inf')):
    if path is None:
        path = [start]
    if start == goal:
        return path
    successors = sorted(graph[start], key=lambda n: heuristic[n])
    if not successors:
        return []
    best = successors[0]
    alternative = successors[1] if len(successors) > 1 else float('inf')
    while heuristic[best] < f_limit:
        result = recursive_best_first_search(graph, best, goal, heuristic, path + [best], min(f_limit, alternative))
        if result:
            return result
        best, alternative = alternative, float('inf')
    return []

def adversarial_search():
    return "To be implemented as Minimax or Alpha-Beta Pruning"

def constraint_satisfaction_problem():
    return "To be implemented using Backtracking or Arc Consistency"

def minimax(position, depth, maximizing_player):
    if depth == 0 or position.is_terminal():
        return position.evaluate()
    if maximizing_player:
        max_eval = float('-inf')
        for child in position.get_children():
            eval = minimax(child, depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for child in position.get_children():
            eval = minimax(child, depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval

def backtracking_csp(variables, domains, constraints, assignment={}):
    if len(assignment) == len(variables):
        return assignment
    var = [v for v in variables if v not in assignment][0]
    for value in domains[var]:
        new_assignment = assignment.copy()
        new_assignment[var] = value
        if all(constraints(var, value, new_assignment) for var in new_assignment):
            result = backtracking_csp(variables, domains, constraints, new_assignment)
            if result:
                return result
    return None


# Example graph representation
london_harry_potter_graph = {
    "King's Cross Station": {"Leadenhall Market": 3.5, "St. Pancras Renaissance Hotel": 0.2},
    "Leadenhall Market": {"Millennium Bridge": 2.5, "Australia House": 1.8},
    "Millennium Bridge": {"Westminster Station": 2.2, "Lambeth Bridge": 1.5},
    "Australia House": {"Piccadilly Circus": 1.2, "Westminster Station": 2.0},
    "Piccadilly Circus": {"Westminster Station": 1.0, "Leadenhall Market": 1.8},
    "Westminster Station": {"Lambeth Bridge": 1.3, "Australia House": 2.0},
    "Lambeth Bridge": {"Millennium Bridge": 1.5, "Westminster Station": 1.3},
    "St. Pancras Renaissance Hotel": {"King's Cross Station": 0.2}
}


heuristic = {
    "King's Cross Station": 0,
    "Leadenhall Market": 3.2,
    "Millennium Bridge": 2.8,
    "Australia House": 2.5,
    "Piccadilly Circus": 2.1,
    "Westminster Station": 3.0,
    "Lambeth Bridge": 3.5,
    "St. Pancras Renaissance Hotel": 0.2
}

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MaraudersMap()
    window.showMaximized()
    sys.exit(app.exec_())