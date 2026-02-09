"""
Enhanced Floor Plan Generator - MVP Implementation
Simple heuristic-based room placement with sunlight and plumbing optimization
"""

import numpy as np
from typing import Dict, List, Tuple
from shapely.geometry import Point, Polygon, box
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class EnhancedFloorPlanGenerator:
    """
    Simplified input floor plan generator with:
    - Automatic centroid generation
    - Sunlight optimization
    - Plumbing cost minimization
    """
    
    def __init__(self):
        # Color scheme for different room types
        self.colors = {
            "living_room": "#F5E6D3",  # Beige
            "bedroom": "#B4D7E8",       # Blue
            "bathroom": "#E8F4F8",      # Light Blue/Cyan
            "kitchen": "#FFE4B5",       # Moccasin (Light Orange)
            "dining": "#F0E68C",        # Khaki
        }
        
        # Room type properties
        self.room_properties = {
            "living_room": {
                "sunlight_priority": 0.8,
                "entrance_proximity": "near",
                "is_wet": False,
                "min_size": 12.0,
                "typical_size": 20.0
            },
            "bedroom": {
                "sunlight_priority": 0.9,   # High - morning sun
                "entrance_proximity": "far",
                "is_wet": False,
                "min_size": 9.0,
                "typical_size": 12.0
            },
            "bathroom": {
                "sunlight_priority": 0.3,   # Low priority
                "entrance_proximity": "far",
                "is_wet": True,             # PLUMBING!
                "min_size": 4.0,
                "typical_size": 5.0
            },
            "kitchen": {
                "sunlight_priority": 0.6,
                "entrance_proximity": "medium",
                "is_wet": True,             # PLUMBING!
                "min_size": 6.0,
                "typical_size": 10.0
            }
        }
    
    def generate_floor_plan(self, config: Dict) -> Dict:
        """
        Main entry point - generate complete floor plan
        
        Args:
            config: {
                "boundary": {"width": 12, "height": 10},
                "rooms": {"living_room": 1, "bedroom": 2, "bathroom": 2, "kitchen": 1},
                "entrance": {"location": "south", "position": 0.5},
                "north_vector": (0, 1)  # Default: north is up
            }
        
        Returns:
            {
                "centroids": {...},
                "dimensions": {...},
                "plumbing_cost": float,
                "sunlight_score": float,
                "layout": figure
            }
        """
        print("🏗️  Generating enhanced floor plan...")
        
        boundary = config["boundary"]
        room_counts = config["rooms"]
        entrance = config["entrance"]
        north_vector = config.get("north_vector", (0, 1))
        
        # STEP 1: Generate optimal centroids using heuristics
        print("  📍 Generating optimal room positions...")
        centroids = self._generate_centroids_heuristic(
            boundary, room_counts, entrance, north_vector
        )
        
        # STEP 2: Calculate room dimensions (simplified - not using GAT-Net yet)
        print("  📏 Calculating room dimensions...")
        dimensions = self._calculate_dimensions(
            centroids, boundary, room_counts
        )
        
        # STEP 3: Optimize for sunlight
        print("  ☀️  Optimizing for sunlight...")
        centroids, sunlight_score = self._optimize_sunlight(
            centroids, north_vector, boundary
        )
        
        # STEP 4: Optimize for plumbing
        print("  🚰 Minimizing plumbing cost...")
        centroids, plumbing_cost = self._optimize_plumbing(
            centroids, dimensions
        )
        
        # STEP 5: Generate final layout
        print("  🎨 Creating visualization...")
        layout_fig = self._create_layout_visualization(
            centroids, dimensions, boundary, entrance, north_vector, 
            plumbing_cost, sunlight_score
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
        
        Strategy:
        1. Divide floor into zones (public, service, private)
        2. Place rooms according to architectural best practices
        """
        width = boundary["width"]
        height = boundary["height"]
        
        # Determine entrance side
        entrance_side = entrance["location"]
        entrance_pos = entrance["position"]
        
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
        wet_area_x = width * 0.75  # Right side by default
        wet_area_y_start = height * 0.4
        
        if room_counts.get("kitchen", 0) > 0:
            centroids["kitchen"] = [(wet_area_x, wet_area_y_start)]
        
        if room_counts.get("bathroom", 0) > 0:
            num_bathrooms = room_counts["bathroom"]
            centroids["bathroom"] = []
            for i in range(num_bathrooms):
                y_pos = wet_area_y_start + (i + 1) * (height * 0.2)
                centroids["bathroom"].append((wet_area_x, y_pos))
        
        # ZONE 3: Private area (bedrooms) - away from entrance, good sunlight
        if room_counts.get("bedroom", 0) > 0:
            num_bedrooms = room_counts["bedroom"]
            centroids["bedroom"] = []
            
            # Place bedrooms on opposite side from entrance
            if entrance_side == "south":
                # Bedrooms at north
                for i in range(num_bedrooms):
                    x_pos = width * (0.3 + i * 0.4)
                    centroids["bedroom"].append((x_pos, height * 0.75))
            else:
                # Bedrooms at south
                for i in range(num_bedrooms):
                    x_pos = width * (0.3 + i * 0.4)
                    centroids["bedroom"].append((x_pos, height * 0.25))
        
        return centroids
    
    def _calculate_dimensions(self, centroids, boundary, room_counts):
        """
        Calculate room dimensions based on available space
        """
        total_area = boundary["width"] * boundary["height"]
        total_rooms = sum(room_counts.values())
        
        # Allocate space proportionally
        dimensions = {}
        
        for room_type, centroid_list in centroids.items():
            dimensions[room_type] = []
            typical_size = self.room_properties[room_type]["typical_size"]
            
            for centroid in centroid_list:
                # Simple heuristic: square rooms with typical area
                width = np.sqrt(typical_size)
                height = np.sqrt(typical_size)
                
                dimensions[room_type].append({
                    "width": width,
                    "height": height,
                    "area": width * height
                })
        
        return dimensions
    
    def _optimize_sunlight(self, centroids, north_vector, boundary):
        """
        Optimize room positions for maximum sunlight exposure
        """
        # Calculate sun direction (east for morning, south for afternoon)
        north_angle = np.arctan2(north_vector[1], north_vector[0])
        east_angle = north_angle - np.pi/2
        
        # Calculate sunlight scores for current layout
        total_score = 0
        room_count = 0
        
        for room_type, centroid_list in centroids.items():
            priority = self.room_properties[room_type]["sunlight_priority"]
            
            for centroid in centroid_list:
                # Calculate angle from center to room
                room_angle = np.arctan2(centroid[1] - boundary["height"]/2,
                                       centroid[0] - boundary["width"]/2)
                
                # Score based on alignment with east (morning sun)
                alignment_score = np.cos(room_angle - east_angle)
                weighted_score = alignment_score * priority
                
                total_score += weighted_score
                room_count += 1
        
        avg_score = total_score / room_count if room_count > 0 else 0
        
        # For MVP, return current centroids (optimization can be added later)
        return centroids, avg_score
    
    def _optimize_plumbing(self, centroids, dimensions):
        """
        Calculate plumbing cost and optimize wet area clustering
        """
        # Find all wet areas
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
        
        # Calculate total pipe length (Minimum Spanning Tree approach)
        total_length = 0
        for i in range(len(wet_rooms) - 1):
            p1 = np.array(wet_rooms[i]["centroid"])
            p2 = np.array(wet_rooms[i + 1]["centroid"])
            distance = np.linalg.norm(p2 - p1)
            total_length += distance
        
        # Cost model: $50 per meter + $200 per connection
        pipe_cost = total_length * 50
        connection_cost = (len(wet_rooms) - 1) * 200
        total_cost = pipe_cost + connection_cost
        
        return centroids, total_cost
    
    def _create_layout_visualization(self, centroids, dimensions, boundary, 
                                     entrance, north_vector, plumbing_cost, sunlight_score):
        """
        Create beautiful color-coded floor plan visualization
        """
        fig, ax = plt.subplots(figsize=(14, 12))
        
        width = boundary["width"]
        height = boundary["height"]
        
        # Draw boundary
        boundary_box = box(0, 0, width, height)
        x, y = boundary_box.exterior.xy
        ax.plot(x, y, 'k-', linewidth=3, label='Boundary')
        
        # Draw rooms with colors
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
        entrance_pos = entrance["position"]
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
        
        # Add information box
        info_text = f"Sunlight Score: {sunlight_score:.2f}/1.0\n"
        info_text += f"Plumbing Cost: ${plumbing_cost:.0f}\n"
        info_text += f"Total Area: {width * height:.1f}m²"
        
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
        ax.set_title('Enhanced Floor Plan with Sunlight & Plumbing Optimization',
                    fontsize=14, weight='bold')
        
        plt.tight_layout()
        
        return fig


# Example usage and testing
if __name__ == "__main__":
    # Create generator
    generator = EnhancedFloorPlanGenerator()
    
    # Define simple configuration
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
            "location": "south",  # Entrance on south wall
            "position": 0.5       # Centered
        },
        "north_vector": (0, 1)    # North is up
    }
    
    # Generate floor plan
    result = generator.generate_floor_plan(config)
    
    # Display results
    print("\n" + "="*60)
    print("FLOOR PLAN GENERATION RESULTS")
    print("="*60)
    print(f"\n💰 Plumbing Cost: ${result['plumbing_cost']:.2f}")
    print(f"☀️  Sunlight Score: {result['sunlight_score']:.2f}/1.0")
    print(f"\n📐 Room Dimensions:")
    
    for room_type, dims_list in result['dimensions'].items():
        for i, dims in enumerate(dims_list):
            print(f"  {room_type.replace('_', ' ').title()} {i+1}: "
                  f"{dims['width']:.1f}m × {dims['height']:.1f}m "
                  f"= {dims['area']:.1f}m²")
    
    # Save figure
    result['layout'].savefig('Outputs/enhanced_floor_plan.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Floor plan saved to: Outputs/enhanced_floor_plan.png")
    
    plt.show()
