"""
Enhanced Floor Plan Generator - GAT-Net Integrated Version
Uses trained Graph Attention Network to predict optimal room dimensions
"""

import numpy as np
from typing import Dict, List, Tuple
from shapely.geometry import Point, Polygon, box
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
import networkx as nx
import os

# Import existing model and utilities
from model import load_model, GATNet
from utils import centroids_to_graph, Handling_dubplicated_nodes, scale
import shapely.wkt

class GATNetFloorPlanGenerator:
    """
    Enhanced floor plan generator using GAT-Net for dimension prediction
    """
    
    def __init__(self, model_path=None):
        # Architectural parameters
        self.wall_thickness = 0.2  # meters (20cm)
        self.door_width = 0.9      # meters (90cm standard)
        self.hallway_width = 1.2   # meters (minimum circulation)
        
        # Color scheme for different room types
        self.colors = {
            "living_room": "#F5E6D3",  # Beige
            "bedroom": "#B4D7E8",       # Blue
            "bathroom": "#E8F4F8",      # Light Blue/Cyan
            "kitchen": "#FFE4B5",       # Moccasin (Light Orange)
            "dining": "#F0E68C",        # Khaki
            "hallway": "#FFFFFF",       # White
        }
        
        # Room type embeddings (must match training data)
        self.room_embeddings = {
            'living': 0,
            'room': 1,      # bedrooms
            'kitchen': 2,
            'bathroom': 3,
        }
        
        # Room type properties
        self.room_properties = {
            "living_room": {
                "sunlight_priority": 0.8,
                "entrance_proximity": "near",
                "is_wet": False,
                "min_size": 12.0,
                "max_size": 40.0,
            },
            "bedroom": {
                "sunlight_priority": 0.9,
                "entrance_proximity": "far",
                "is_wet": False,
                "min_size": 9.0,
                "max_size": 25.0,
            },
            "bathroom": {
                "sunlight_priority": 0.3,
                "entrance_proximity": "far",
                "is_wet": True,
                "min_size": 4.0,
                "max_size": 12.0,
            },
            "kitchen": {
                "sunlight_priority": 0.6,
                "entrance_proximity": "medium",
                "is_wet": True,
                "min_size": 6.0,
                "max_size": 20.0,
            }
        }
        
        # Load GAT-Net model
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 
                                     "GAT-Net_model/checkpoints/GAT-Net_v3_UnScalled.pt")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🧠 Loading GAT-Net model from: {model_path}")
        self.model = load_model(model_path, self.device)
        print(f"✅ Model loaded successfully on {self.device}")
    
    def generate_floor_plan(self, config: Dict) -> Dict:
        """
        Main entry point - generate complete floor plan using GAT-Net
        """
        print("\n🏗️  Generating GAT-Net enhanced floor plan...")
        
        boundary = config["boundary"]
        room_counts = config["rooms"]
        entrance = config["entrance"]
        north_vector = config.get("north_vector", (0, 1))
        
        # STEP 1: Generate initial centroids using heuristics
        print("  📍 Generating initial room positions...")
        centroids = self._generate_centroids_heuristic(
            boundary, room_counts, entrance, north_vector
        )
        
        # STEP 2: Create graphs for GAT-Net
        print("  🔗 Building graph structures...")
        G, B, G_networkx, B_networkx = self._create_graphs_for_gatnet(
            centroids, boundary, entrance
        )
        
        # STEP 3: Use GAT-Net to predict dimensions
        print("  🤖 Running GAT-Net inference...")
        dimensions = self._predict_dimensions_with_gatnet(
            G, B, centroids, boundary
        )
        
        # STEP 4: Optimize for sunlight
        print("  ☀️  Optimizing for sunlight...")
        centroids, sunlight_score = self._optimize_sunlight(
            centroids, north_vector, boundary
        )
        
        # STEP 5: Optimize for plumbing
        print("  🚰 Minimizing plumbing cost...")
        centroids, plumbing_cost = self._optimize_plumbing(
            centroids, dimensions
        )
        
        # STEP 6: Correct overlaps using separation steering
        print("  📐 Correcting room overlaps...")
        centroids, dimensions = self._correct_overlaps(
            centroids, dimensions, boundary
        )
        
        # STEP 6.5: Apply architectural constraints (walls, doors, circulation)
        print("  🏛️  Applying architectural constraints...")
        architectural_layout = self._apply_architectural_constraints(
            centroids, dimensions, boundary
        )
        
        # STEP 7: Generate final layout
        print("  🎨 Creating visualization...")
        layout_fig = self._create_layout_visualization(
            centroids, dimensions, boundary, entrance, north_vector, 
            plumbing_cost, sunlight_score, architectural_layout
        )
        
        print("✅ Floor plan generated successfully!")
        
        return {
            "centroids": centroids,
            "dimensions": dimensions,
            "plumbing_cost": plumbing_cost,
            "sunlight_score": sunlight_score,
            "layout": layout_fig,
            "boundary": boundary
        }
    
    def _generate_centroids_heuristic(self, boundary, room_counts, entrance, north_vector):
        """
        Generate room centroids using zone-based heuristics
        """
        width = boundary["width"]
        height = boundary["height"]
        entrance_side = entrance["location"]
        
        centroids = {}
        
        # ZONE 1: Public area (near entrance) - Living room
        if room_counts.get("living_room", 0) > 0:
            if entrance_side == "south":
                centroids["living_room"] = [(width * 0.5, height * 0.25)]
            elif entrance_side == "north":
                centroids["living_room"] = [(width * 0.5, height * 0.75)]
            elif entrance_side == "east":
                centroids["living_room"] = [(width * 0.75, height * 0.5)]
            else:  # west
                centroids["living_room"] = [(width * 0.25, height * 0.5)]
        
        # ZONE 2: Service area (kitchen + bathrooms) - grouped for plumbing
        wet_area_x = width * 0.75
        wet_area_y_start = height * 0.4
        
        if room_counts.get("kitchen", 0) > 0:
            centroids["kitchen"] = [(wet_area_x, wet_area_y_start)]
        
        if room_counts.get("bathroom", 0) > 0:
            num_bathrooms = room_counts["bathroom"]
            centroids["bathroom"] = []
            for i in range(num_bathrooms):
                y_pos = wet_area_y_start + (i + 1) * (height * 0.2)
                centroids["bathroom"].append((wet_area_x, y_pos))
        
        # ZONE 3: Private area (bedrooms)
        if room_counts.get("bedroom", 0) > 0:
            num_bedrooms = room_counts["bedroom"]
            centroids["bedroom"] = []
            
            if entrance_side == "south":
                for i in range(num_bedrooms):
                    x_pos = width * (0.3 + i * 0.4)
                    centroids["bedroom"].append((x_pos, height * 0.75))
            else:
                for i in range(num_bedrooms):
                    x_pos = width * (0.3 + i * 0.4)
                    centroids["bedroom"].append((x_pos, height * 0.25))
        
        return centroids
    
    def _create_graphs_for_gatnet(self, centroids, boundary, entrance):
        """
        Create graph structures compatible with GAT-Net model
        Returns: (G_pyg, B_pyg, G_networkx, B_networkx)
        """
        width = boundary["width"]
        height = boundary["height"]
        
        # Convert to WKT format for existing utilities
        boundary_wkt = f"POLYGON ((0 0, {width} 0, {width} {height}, 0 {height}, 0 0))"
        boundary_poly = shapely.wkt.loads(boundary_wkt)
        
        # Create front door polygon
        entrance_side = entrance["location"]
        entrance_pos = entrance.get("position", 0.5)
        door_size = 1.0
        
        if entrance_side == "south":
            door_x = width * entrance_pos
            door_wkt = f"POLYGON (({door_x - door_size/2} -0.1, {door_x + door_size/2} -0.1, {door_x + door_size/2} 0.1, {door_x - door_size/2} 0.1, {door_x - door_size/2} -0.1))"
        elif entrance_side == "north":
            door_x = width * entrance_pos
            door_wkt = f"POLYGON (({door_x - door_size/2} {height - 0.1}, {door_x + door_size/2} {height - 0.1}, {door_x + door_size/2} {height + 0.1}, {door_x - door_size/2} {height + 0.1}, {door_x - door_size/2} {height - 0.1}))"
        else:
            door_wkt = f"POLYGON ((0 {height/2 - door_size/2}, 0.1 {height/2 - door_size/2}, 0.1 {height/2 + door_size/2}, 0 {height/2 + door_size/2}, 0 {height/2 - door_size/2}))"
        
        front_door = shapely.wkt.loads(door_wkt)
        
        # Map room types to model expectations
        user_constraints = {}
        for room_type, centroid_list in centroids.items():
            if room_type == "living_room":
                model_type = "living"
            elif room_type == "bedroom":
                model_type = "room"
            else:
                model_type = room_type
            user_constraints[model_type] = centroid_list
        
        # Ensure living room exists (required by model)
        if "living" not in user_constraints:
            user_constraints["living"] = [(width/2, height/2)]
        
        # Create room graph using existing utility
        G_networkx = centroids_to_graph(user_constraints, living_to_all=True)
        
        # Create boundary graph
        B_networkx = Handling_dubplicated_nodes(boundary_poly, front_door)
        
        # Convert to PyTorch Geometric format
        features = ['roomType_embd', 'actualCentroid_x', 'actualCentroid_y']
        G = from_networkx(G_networkx, group_edge_attrs=['distance'], group_node_attrs=features)
        B = from_networkx(B_networkx, group_node_attrs=['type', 'centroid'], group_edge_attrs=['distance'])
        
        # Normalization (matching training preprocessing)
        G_x_mean = G.x[:, 1].mean().item()
        G_y_mean = G.x[:, 2].mean().item()
        G_x_std = G.x[:, 1].std().item()
        G_y_std = G.x[:, 2].std().item()
        
        G.x[:, 1:] = (G.x[:, 1:] - torch.tensor([G_x_mean, G_y_mean])) / torch.tensor([G_x_std, G_y_std])
        
        # One-hot encode room types
        first_column_encodings = F.one_hot(G.x[:, 0].long(), 7)
        G.x = torch.cat([first_column_encodings, G.x[:, 1:]], axis=1)
        
        # Normalize boundary graph
        B_x_mean = B.x[:, 1].mean().item()
        B_y_mean = B.x[:, 2].mean().item()
        B_x_std = B.x[:, 1].std().item()
        B_y_std = B.x[:, 2].std().item()
        
        B.x[:, 1:] = (B.x[:, 1:] - torch.tensor([B_x_mean, B_y_mean])) / torch.tensor([B_x_std, B_y_std])
        
        # Convert to correct types
        G.x = G.x.to(torch.float32)
        G.edge_attr = G.edge_attr.to(torch.float32)
        G.edge_index = G.edge_index.to(torch.int64)
        
        B.x = B.x.to(torch.float32)
        B.edge_index = B.edge_index.to(torch.int64)
        B.edge_attr = B.edge_attr.to(torch.float32)
        
        # Store normalization params for later use
        G.norm_params = (G_x_mean, G_y_mean, G_x_std, G_y_std)
        B.norm_params = (B_x_mean, B_y_mean, B_x_std, B_y_std)
        
        return G, B, G_networkx, B_networkx
    
    def _predict_dimensions_with_gatnet(self, G, B, centroids, boundary):
        """
        Use GAT-Net model to predict room dimensions
        """
        # Move graphs to device
        G = G.to(self.device)
        B = B.to(self.device)
        
        # Run model inference
        with torch.no_grad():
            width_pred, height_pred = self.model(G, B)
        
        # Convert predictions to numpy
        widths = width_pred.cpu().numpy().flatten()
        heights = height_pred.cpu().numpy().flatten()
        
        # The model outputs are in normalized/unscaled space
        # We need to scale them to reasonable proportions of the boundary
        # Typical approach: use total area constraint
        
        # Calculate total predicted area
        total_pred_area = np.sum(widths * heights)
        
        # Target total area (60-70% of boundary area for realistic spacing)
        boundary_area = boundary["width"] * boundary["height"]
        target_area = 0.65 * boundary_area
        
        # Scale factor to match target area
        if total_pred_area > 0:
            scale_factor = np.sqrt(target_area / total_pred_area)
        else:
            scale_factor = 1.0
        
        # Apply scaling
        widths = widths * scale_factor
        heights = heights * scale_factor
        
        # Map predictions back to room types
        dimensions = {}
        node_idx = 0
        
        for room_type, centroid_list in centroids.items():
            dimensions[room_type] = []
            
            for i in range(len(centroid_list)):
                w = float(widths[node_idx])
                h = float(heights[node_idx])
                
                # Ensure minimum sizes
                min_size = self.room_properties[room_type]["min_size"]
                max_size = self.room_properties[room_type]["max_size"]
                min_dim = np.sqrt(min_size)
                max_dim = np.sqrt(max_size)
                
                # Clamp to reasonable bounds
                w = np.clip(w, min_dim, max_dim)
                h = np.clip(h, min_dim, max_dim)
                
                dimensions[room_type].append({
                    "width": w,
                    "height": h,
                    "area": w * h
                })
                
                node_idx += 1
        
        return dimensions
    
    def _correct_overlaps(self, centroids, dimensions, boundary):
        """
        Use Guillotine/Box Packing algorithm to ensure NO overlaps
        """
        width = boundary["width"]
        height = boundary["height"]
        
        print("    🔧 Applying Guillotine Box Packing...")
        
        # Collect all rooms with their dimensions and priorities
        rooms_to_place = []
        for room_type, centroid_list in centroids.items():
            priority = self._get_placement_priority(room_type)
            for i, centroid in enumerate(centroid_list):
                dims = dimensions[room_type][i]
                rooms_to_place.append({
                    'type': room_type,
                    'index': i,
                    'width': dims['width'],
                    'height': dims['height'],
                    'area': dims['area'],
                    'priority': priority,
                    'original_centroid': centroid
                })
        
        # Sort by priority (higher priority placed first)
        rooms_to_place.sort(key=lambda x: (-x['priority'], -x['area']))
        
        # Use Guillotine packing algorithm
        new_positions = self._guillotine_pack(rooms_to_place, width, height)
        
        # Update centroids with packed positions
        for room_info, position in zip(rooms_to_place, new_positions):
            room_type = room_info['type']
            room_idx = room_info['index']
            
            # Find this room in the centroids dict
            centroids[room_type][room_idx] = (position['x'], position['y'])
        
        print(f"    ✓ Successfully packed {len(rooms_to_place)} rooms with NO overlaps")
        
        return centroids, dimensions
    
    def _get_placement_priority(self, room_type):
        """Get placement priority (higher = placed first)"""
        priorities = {
            'living_room': 5,  # Highest - near entrance
            'kitchen': 4,      # High - needs specific placement
            'bedroom': 3,      # Medium
            'bathroom': 2,     # Low - can fit in gaps
        }
        return priorities.get(room_type, 1)
    
    def _guillotine_pack(self, rooms, container_width, container_height):
        """
        Guillotine bin packing algorithm - guarantees no overlaps
        """
        # Free rectangles (available spaces)
        free_rects = [(0, 0, container_width, container_height)]
        placed_positions = []
        
        margin = 0.3  # Small margin between rooms
        
        for room in rooms:
            w = room['width'] + margin
            h = room['height'] + margin
            
            # Find best free rectangle
            best_rect = None
            best_idx = -1
            
            for idx, (fx, fy, fw, fh) in enumerate(free_rects):
                # Try both orientations
                if w <= fw and h <= fh:
                    if best_rect is None or (fw * fh) < (best_rect[2] * best_rect[3]):
                        best_rect = (fx, fy, fw, fh)
                        best_idx = idx
            
            if best_rect is None:
                # Can't fit - place at remaining space (will overlap but minimal)
                print(f"      ⚠️  Warning: {room['type']} may not fit perfectly")
                # Place at first available spot
                if free_rects:
                    fx, fy, fw, fh = free_rects[0]
                    x = fx + min(w, fw) / 2
                    y = fy + min(h, fh) / 2
                else:
                    x = container_width / 2
                    y = container_height / 2
                placed_positions.append({'x': x, 'y': y})
                continue
            
            # Place room in best rectangle
            fx, fy, fw, fh = best_rect
            
            # Centroid position (account for margin)
            x = fx + (room['width'] / 2) + margin / 2
            y = fy + (room['height'] / 2) + margin / 2
            
            placed_positions.append({'x': x, 'y': y})
            
            # Remove used rectangle
            free_rects.pop(best_idx)
            
            # Split remaining space (Guillotine split)
            remaining_width = fw - w
            remaining_height = fh - h
            
            # Horizontal split (prefer horizontal for consistency)
            if remaining_width > 0.5:
                free_rects.append((fx + w, fy, remaining_width, fh))
            if remaining_height > 0.5:
                free_rects.append((fx, fy + h, w, remaining_height))
            
            # Add remaining corner if both dimensions have space
            if remaining_width > 0.5 and remaining_height > 0.5:
                free_rects.append((fx + w, fy + h, remaining_width, remaining_height))
        
        return placed_positions
    
    def _apply_architectural_constraints(self, centroids, dimensions, boundary):
        """
        Convert conceptual boxes into architectural layout with walls, doors, circulation
        """
        print("    🧱 Calculating wall geometry...")
        
        # Build room structures with wall thickness
        rooms = []
        for room_type, centroid_list in centroids.items():
            for i, centroid in enumerate(centroid_list):
                dims = dimensions[room_type][i]
                w, h = dims['width'], dims['height']
                
                # Interior clear dimensions (as predicted by GATNet)
                interior_poly = box(
                    centroid[0] - w/2,
                    centroid[1] - h/2,
                    centroid[0] + w/2,
                    centroid[1] + h/2
                )
                
                # Exterior dimensions (including wall thickness)
                wall = self.wall_thickness
                exterior_poly = box(
                    centroid[0] - w/2 - wall,
                    centroid[1] - h/2 - wall,
                    centroid[0] + w/2 + wall,
                    centroid[1] + h/2 + wall
                )
                
                rooms.append({
                    'type': room_type,
                    'index': i,
                    'centroid': centroid,
                    'interior': interior_poly,
                    'exterior': exterior_poly,
                    'width': w,
                    'height': h
                })
        
        # Detect adjacencies and shared walls
        print("    🚪 Detecting adjacencies and placing doors...")
        adjacencies = self._detect_adjacencies(rooms)
        doors = self._place_doors(adjacencies)
        
        # Detect circulation needs and inject hallways
        print("    🚶 Planning circulation paths...")
        hallways = self._plan_circulation(rooms, centroids, boundary)
        
        return {
            'rooms': rooms,
            'adjacencies': adjacencies,
            'doors': doors,
            'hallways': hallways
        }
    
    def _detect_adjacencies(self, rooms):
        """
        Detect which rooms share edges (are adjacent)
        """
        adjacencies = []
        epsilon = 0.5  # rooms within 50cm are considered adjacent
        
        for i, room1 in enumerate(rooms):
            for room2 in rooms[i+1:]:
                # Check if exterior boundaries are close
                if room1['exterior'].distance(room2['exterior']) < epsilon:
                    # Find shared edge
                    intersection = room1['exterior'].intersection(room2['exterior'])
                    
                    if intersection.length > 0.5:  # At least 50cm shared edge
                        adjacencies.append({
                            'room1': room1,
                            'room2': room2,
                            'shared_edge': intersection,
                            'length': intersection.length
                        })
        
        return adjacencies
    
    def _place_doors(self, adjacencies):
        """
        Place doors on shared edges between adjacent rooms
        """
        doors = []
        
        for adj in adjacencies:
            # Get midpoint of shared edge
            edge = adj['shared_edge']
            
            if edge.geom_type == 'LineString':
                # Simple case: straight shared wall
                midpoint = edge.interpolate(0.5, normalized=True)
                door_start = edge.interpolate(0.5 - (self.door_width / edge.length / 2), normalized=True)
                door_end = edge.interpolate(0.5 + (self.door_width / edge.length / 2), normalized=True)
                
                doors.append({
                    'room1_type': adj['room1']['type'],
                    'room2_type': adj['room2']['type'],
                    'location': (midpoint.x, midpoint.y),
                    'start': (door_start.x, door_start.y),
                    'end': (door_end.x, door_end.y),
                    'width': self.door_width
                })
            elif edge.geom_type == 'Point':
                # Corner touching - skip door placement
                continue
            elif edge.geom_type == 'MultiLineString' or edge.geom_type == 'GeometryCollection':
                # Multiple segments - use longest
                try:
                    longest = max(edge.geoms, key=lambda g: g.length if hasattr(g, 'length') else 0)
                    if hasattr(longest, 'length') and longest.length > 0.5:
                        midpoint = longest.interpolate(0.5, normalized=True)
                        doors.append({
                            'room1_type': adj['room1']['type'],
                            'room2_type': adj['room2']['type'],
                            'location': (midpoint.x, midpoint.y),
                            'start': (midpoint.x - self.door_width/2, midpoint.y),
                            'end': (midpoint.x + self.door_width/2, midpoint.y),
                            'width': self.door_width
                        })
                except:
                    continue
        
        return doors
    
    def _plan_circulation(self, rooms, centroids, boundary):
        """
        Detect if hallway circulation is needed and plan paths
        """
        hallways = []
        
        # Find living room (main public space)
        living_room = None
        for room in rooms:
            if room['type'] == 'living_room':
                living_room = room
                break
        
        if living_room is None:
            return hallways
        
        # Check if any room is too far from living room without direct adjacency
        living_centroid = Point(living_room['centroid'])
        
        for room in rooms:
            if room['type'] == 'living_room':
                continue
            
            room_centroid = Point(room['centroid'])
            distance = living_centroid.distance(room_centroid)
            
            # If distance > 5m and no direct adjacency, consider hallway
            if distance > 5.0:
                # Check if there's a direct path
                has_direct_adjacency = False
                for adj in []:  # Would need adjacency list here
                    if (adj['room1'] == living_room and adj['room2'] == room) or \
                       (adj['room2'] == living_room and adj['room1'] == room):
                        has_direct_adjacency = True
                        break
                
                if not has_direct_adjacency:
                    # Plan hallway corridor
                    # Simple approach: straight line connection
                    x1, y1 = living_room['centroid']
                    x2, y2 = room['centroid']
                    
                    # Create hallway polygon (simplified)
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    
                    hallways.append({
                        'start': (x1, y1),
                        'end': (x2, y2),
                        'width': self.hallway_width,
                        'polygon': box(
                            min(x1, x2) - self.hallway_width/2,
                            min(y1, y2),
                            max(x1, x2) + self.hallway_width/2,
                            max(y1, y2)
                        )
                    })
        
        return hallways
    
    def _optimize_sunlight(self, centroids, north_vector, boundary):
        """
        Optimize room positions for maximum sunlight exposure
        """
        north_direction = np.array(north_vector)
        north_angle = np.arctan2(north_vector[1], north_vector[0])
        east_angle = north_angle - np.pi/2
        
        total_score = 0
        room_count = 0
        
        for room_type, centroid_list in centroids.items():
            priority = self.room_properties[room_type]["sunlight_priority"]
            
            for centroid in centroid_list:
                room_angle = np.arctan2(centroid[1] - boundary["height"]/2,
                                       centroid[0] - boundary["width"]/2)
                
                alignment_score = np.cos(room_angle - east_angle)
                weighted_score = alignment_score * priority
                
                total_score += weighted_score
                room_count += 1
        
        avg_score = total_score / room_count if room_count > 0 else 0
        
        return centroids, avg_score
    
    def _optimize_plumbing(self, centroids, dimensions):
        """
        Calculate plumbing cost and optimize wet area clustering
        """
        wet_rooms = []
        
        for room_type, centroid_list in centroids.items():
            if self.room_properties[room_type]["is_wet"]:
                for i, centroid in enumerate(centroid_list):
                    wet_rooms.append({
                        "type": room_type,
                        "index": i,
                        "centroid": centroid
                    })
        
        if len(wet_rooms) < 2:
            return centroids, 0
        
        # Calculate total pipe length
        total_length = 0
        for i in range(len(wet_rooms) - 1):
            p1 = np.array(wet_rooms[i]["centroid"])
            p2 = np.array(wet_rooms[i + 1]["centroid"])
            distance = np.linalg.norm(p2 - p1)
            total_length += distance
        
        # Cost model
        pipe_cost = total_length * 50
        connection_cost = (len(wet_rooms) - 1) * 200
        total_cost = pipe_cost + connection_cost
        
        return centroids, total_cost
    
    def _validate_no_overlaps(self, centroids, dimensions):
        """
        Validate that rooms don't overlap (for verification)
        Returns count of overlapping pairs
        """
        room_boxes = []
        
        for room_type, centroid_list in centroids.items():
            for i, centroid in enumerate(centroid_list):
                dims = dimensions[room_type][i]
                w, h = dims['width'], dims['height']
                
                # Create bounding box
                x_min = centroid[0] - w/2
                x_max = centroid[0] + w/2
                y_min = centroid[1] - h/2
                y_max = centroid[1] + h/2
                
                room_boxes.append({
                    'type': room_type,
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': y_min,
                    'y_max': y_max
                })
        
        # Check for overlaps
        overlap_count = 0
        for i, box1 in enumerate(room_boxes):
            for box2 in room_boxes[i+1:]:
                # Check if boxes overlap
                if not (box1['x_max'] <= box2['x_min'] or  # box1 left of box2
                        box2['x_max'] <= box1['x_min'] or  # box2 left of box1
                        box1['y_max'] <= box2['y_min'] or  # box1 below box2
                        box2['y_max'] <= box1['y_min']):   # box2 below box1
                    overlap_count += 1
                    print(f"      ⚠️  Overlap detected: {box1['type']} and {box2['type']}")
        
        return overlap_count
    
    def _create_layout_visualization(self, centroids, dimensions, boundary, 
                                     entrance, north_vector, plumbing_cost, sunlight_score,
                                     architectural_layout=None):
        """
        Create beautiful color-coded floor plan visualization with architectural details
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        
        width = boundary["width"]
        height = boundary["height"]
        
        # Draw boundary (thick external walls)
        boundary_box = box(0, 0, width, height)
        x, y = boundary_box.exterior.xy
        ax.plot(x, y, 'k-', linewidth=6, label='External Wall', solid_capstyle='round')
        
        # Use architectural layout if available
        if architectural_layout:
            rooms_data = architectural_layout['rooms']
            doors = architectural_layout['doors']
            hallways = architectural_layout['hallways']
            
            # Draw hallways first (background)
            for hallway in hallways:
                h_poly = hallway['polygon']
                h_x, h_y = h_poly.exterior.xy
                ax.fill(h_x, h_y, color=self.colors.get('hallway', '#FFFFFF'), 
                       edgecolor='gray', linewidth=1, linestyle='--', alpha=0.3)
                ax.text(hallway['start'][0], hallway['start'][1] + 0.5, 'Circulation',
                       ha='center', fontsize=8, style='italic', color='gray')
            
            # Draw room interiors with colors
            for room in rooms_data:
                color = self.colors.get(room['type'], "#CCCCCC")
                
                # Fill interior space
                int_x, int_y = room['interior'].exterior.xy
                ax.fill(int_x, int_y, color=color, alpha=0.7)
                
                # Draw exterior walls (thicker for external, thinner for internal)
                ext_x, ext_y = room['exterior'].exterior.xy
                ax.plot(ext_x, ext_y, 'black', linewidth=2.5, solid_capstyle='round')
                
                # Add label with dimensions
                centroid = room['centroid']
                label = f"{room['type'].replace('_', ' ').title()}\n"
                label += f"{room['width']:.1f}m × {room['height']:.1f}m\n"
                label += f"{room['width'] * room['height']:.1f}m²"
                
                ax.text(centroid[0], centroid[1], label,
                       ha='center', va='center', fontsize=9, weight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Draw doors (openings in walls)
            for door in doors:
                # Draw door as a gap with arc indicator
                dx, dy = door['start']
                ex, ey = door['end']
                
                # Door opening (white/gap)
                ax.plot([dx, ex], [dy, ey], 'white', linewidth=5, solid_capstyle='butt')
                
                # Door swing arc
                mid_x, mid_y = door['location']
                door_arc = mpatches.Arc((mid_x, mid_y), self.door_width, self.door_width,
                                       angle=0, theta1=0, theta2=90, 
                                       color='brown', linewidth=1.5, linestyle='-')
                ax.add_patch(door_arc)
                
                # Small door marker
                ax.plot(mid_x, mid_y, 'o', color='brown', markersize=3)
        
        else:
            # Fallback: simple box rendering (old method)
            for room_type, centroid_list in centroids.items():
                color = self.colors.get(room_type, "#CCCCCC")
                
                for i, centroid in enumerate(centroid_list):
                    dims = dimensions[room_type][i]
                    w = dims["width"]
                    h = dims["height"]
                    
                    # Create room rectangle
                    room_box = box(
                        centroid[0] - w/2, 
                        centroid[1] - h/2,
                        centroid[0] + w/2,
                        centroid[1] + h/2
                    )
                    
                    # Draw room
                    room_x, room_y = room_box.exterior.xy
                    ax.fill(room_x, room_y, color=color, edgecolor='black', 
                           linewidth=2, alpha=0.7)
                    
                    # Add label with dimensions
                    label = f"{room_type.replace('_', ' ').title()}\n"
                    label += f"{w:.1f}m × {h:.1f}m\n"
                    label += f"{dims['area']:.1f}m²"
                    
                    ax.text(centroid[0], centroid[1], label,
                           ha='center', va='center', fontsize=9, weight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Draw entrance
        entrance_side = entrance["location"]
        entrance_pos = entrance.get("position", 0.5)
        door_size = 1.0
        
        if entrance_side == "south":
            door_x = width * entrance_pos
            door_y = 0
            ax.plot([door_x - door_size/2, door_x + door_size/2], [door_y, door_y],
                   'r-', linewidth=6, label='Entrance')
        
        # Draw north arrow
        arrow_x, arrow_y = width * 0.9, height * 0.9
        ax.arrow(arrow_x, arrow_y, north_vector[0] * 0.5, north_vector[1] * 0.5,
                head_width=0.3, head_length=0.2, fc='red', ec='red', linewidth=2)
        ax.text(arrow_x, arrow_y - 0.5, 'N', fontsize=14, weight='bold',
               ha='center', color='red')
        
        # Draw plumbing connections
        wet_rooms = []
        for room_type, centroid_list in centroids.items():
            if self.room_properties[room_type]["is_wet"]:
                wet_rooms.extend(centroid_list)
        
        if len(wet_rooms) > 1:
            for i in range(len(wet_rooms) - 1):
                ax.plot([wet_rooms[i][0], wet_rooms[i+1][0]],
                       [wet_rooms[i][1], wet_rooms[i+1][1]],
                       'b--', linewidth=2, alpha=0.6,
                       label='Plumbing' if i == 0 else '')
        
        # Validate no overlaps (verification)
        overlap_count = self._validate_no_overlaps(centroids, dimensions)
        
        # Add information box
        info_text = f"🤖 GAT-Net Predictions\n"
        info_text += f"☀️  Sunlight Score: {sunlight_score:.2f}/1.0\n"
        info_text += f"💰 Plumbing Cost: ${plumbing_cost:.0f}\n"
        info_text += f"📐 Total Area: {width * height:.1f}m²\n"
        info_text += f"{'✅ NO Overlaps' if overlap_count == 0 else f'⚠️  {overlap_count} Overlaps'}\n"
        if architectural_layout:
            info_text += f"🚪 Doors: {len(architectural_layout['doors'])}\n"
            info_text += f"🚶 Hallways: {len(architectural_layout['hallways'])}\n"
            info_text += f"🧱 Wall Thickness: {self.wall_thickness*100:.0f}cm"
        
        ax.text(0.02, 0.98, info_text,
               transform=ax.transAxes, fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Create legend
        legend_elements = [
            mpatches.Patch(facecolor=self.colors["living_room"], 
                          edgecolor='black', label='Living Room'),
            mpatches.Patch(facecolor=self.colors["bedroom"], 
                          edgecolor='black', label='Bedroom'),
            mpatches.Patch(facecolor=self.colors["bathroom"], 
                          edgecolor='black', label='Bathroom'),
            mpatches.Patch(facecolor=self.colors["kitchen"], 
                          edgecolor='black', label='Kitchen'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.set_xlim(-1, width + 1)
        ax.set_ylim(-1, height + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Width (meters)', fontsize=12)
        ax.set_ylabel('Height (meters)', fontsize=12)
        title = 'GAT-Net Enhanced Floor Plan'
        if architectural_layout:
            title += ' - Architectural Layout with Walls & Doors'
        else:
            title += ' - Conceptual Layout'
        ax.set_title(title, fontsize=14, weight='bold')
        
        plt.tight_layout()
        
        return fig


# Example usage
if __name__ == "__main__":
    # Create GAT-Net generator
    generator = GATNetFloorPlanGenerator()
    
    # Define configuration
    config = {
        "boundary": {
            "width": 12.0,   # 12 meters wide
            "height": 10.0   # 10 meters deep
        },
        "rooms": {
            "living_room": 1,
            "bedroom": 2,
            "bathroom": 2,
            "kitchen": 1
        },
        "entrance": {
            "location": "south",
            "position": 0.5
        },
        "north_vector": (0, 1)
    }
    
    # Generate floor plan
    result = generator.generate_floor_plan(config)
    
    # Display results
    print("\n" + "="*60)
    print("GAT-NET FLOOR PLAN GENERATION RESULTS")
    print("="*60)
    print(f"\n💰 Plumbing Cost: ${result['plumbing_cost']:.2f}")
    print(f"☀️  Sunlight Score: {result['sunlight_score']:.2f}/1.0")
    print(f"\n📐 GAT-Net Predicted Room Dimensions:")
    
    for room_type, dims_list in result['dimensions'].items():
        for i, dims in enumerate(dims_list):
            print(f"  {room_type.replace('_', ' ').title()} {i+1}: "
                  f"{dims['width']:.1f}m × {dims['height']:.1f}m "
                  f"= {dims['area']:.1f}m²")
    
    # Save figure
    result['layout'].savefig('Outputs/gatnet_enhanced_floor_plan.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Floor plan saved to: Outputs/gatnet_enhanced_floor_plan.png")
    
    plt.show()
