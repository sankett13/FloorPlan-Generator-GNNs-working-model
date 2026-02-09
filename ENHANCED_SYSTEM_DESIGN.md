# 🏗️ Enhanced Floor Plan Generator - Design Document

## 🎯 System Overview

### Current Limitations:

- ❌ User must manually provide room centroids
- ❌ No sunlight optimization
- ❌ No plumbing cost consideration
- ❌ No automatic intelligent placement

### Enhanced Features:

- ✅ **Automatic centroid generation** from room counts
- ✅ **Sunlight optimization engine** using north vector
- ✅ **Plumbing cost minimization** (wet area clustering)
- ✅ **Smart entrance placement**
- ✅ **Color-coded output** with dimensions

---

## 📥 New Input Specification

```python
enhanced_input = {
    # Simple bounding box instead of complex polygon
    "boundary": {
        "type": "rectangle",
        "width": 15.0,      # meters
        "height": 12.0,     # meters
        "center": (0, 0)    # optional, default origin
    },

    # Room requirements (counts only)
    "rooms": {
        "living_room": 1,
        "bedroom": 2,
        "bathroom": 2,
        "kitchen": 1,
        "dining": 0,        # optional
        "study": 0          # optional
    },

    # Entrance specification
    "entrance": {
        "location": "south",  # north, south, east, west
        "position": 0.5       # 0-1, position along that wall (0.5 = center)
    },

    # Environmental factors
    "environment": {
        "north_vector": (0, 1),  # (x, y) direction of north
        "sunlight_priority": True,
        "plumbing_optimization": True,
        "ventilation_priority": True
    },

    # Preferences (optional)
    "preferences": {
        "min_room_size": 9.0,        # sq meters
        "min_bathroom_size": 4.0,    # sq meters
        "corridor_width": 1.2,       # meters
        "wall_thickness": 0.15       # meters
    }
}
```

---

## 🧠 Three-Stage Pipeline

### **Stage 1: Heuristic Room Placement Engine**

Generate optimal centroids based on rules and constraints.

### **Stage 2: GAT-Net Size Prediction**

Use existing model to predict optimal room dimensions.

### **Stage 3: Geometric Optimization**

Refine layout based on sunlight, plumbing, and fit constraints.

---

## 🔧 Stage 1: Heuristic Placement Engine

### **Algorithm: Rule-Based Centroid Generation**

```python
class HeuristicPlacementEngine:
    """
    Generates optimal room centroids based on architectural best practices
    """

    def __init__(self, boundary, room_counts, entrance, north_vector):
        self.boundary = boundary
        self.room_counts = room_counts
        self.entrance = entrance
        self.north_vector = north_vector

        # Architectural rules
        self.rules = {
            "living_room": {
                "priority": 1,
                "sunlight": "high",      # Needs good natural light
                "entrance_distance": "near",  # Near entrance
                "size_ratio": 1.0        # Reference size
            },
            "bedroom": {
                "priority": 2,
                "sunlight": "high",      # Morning/evening sun
                "privacy": "high",       # Away from entrance
                "quiet": "high",         # Away from kitchen/living
                "size_ratio": 0.7
            },
            "kitchen": {
                "priority": 3,
                "is_wet_area": True,     # For plumbing optimization
                "ventilation": "high",   # Needs windows
                "entrance_distance": "medium",
                "size_ratio": 0.5
            },
            "bathroom": {
                "priority": 4,
                "is_wet_area": True,     # CRITICAL: Stack with kitchen
                "privacy": "high",
                "sunlight": "low",       # Less important
                "size_ratio": 0.3
            }
        }

    def generate_centroids(self):
        """
        Main algorithm: Generate optimal centroids
        """
        # Step 1: Divide space into zones
        zones = self._create_zones()

        # Step 2: Assign rooms to zones based on rules
        room_assignments = self._assign_rooms_to_zones(zones)

        # Step 3: Optimize for sunlight
        room_assignments = self._optimize_sunlight(room_assignments)

        # Step 4: Cluster wet areas (plumbing optimization)
        room_assignments = self._cluster_wet_areas(room_assignments)

        # Step 5: Generate final centroids
        centroids = self._compute_centroids(room_assignments)

        return centroids

    def _create_zones(self):
        """
        Divide floor plan into functional zones

        Zone Strategy:
        ┌─────────────────────────┐
        │  North (Sunlight) ☀️    │
        │  ┌─────────────────┐   │
        │  │  Public Zone    │   │
        │  │  (Living, Dining)│   │
        ├──┴─────────────────┴───┤
        │  Service Zone           │
        │  (Kitchen, Bathrooms)   │
        ├─────────────────────────┤
        │  Private Zone           │
        │  (Bedrooms)             │
        └─────────────────────────┘
        """
        width, height = self.boundary["width"], self.boundary["height"]

        # Calculate north direction influence
        north_angle = np.arctan2(self.north_vector[1], self.north_vector[0])

        zones = {
            "public": {
                "region": self._get_near_entrance_zone(),
                "priority": ["living_room", "dining"],
                "sunlight_score": 0.8
            },
            "service": {
                "region": self._get_core_zone(),
                "priority": ["kitchen", "bathroom"],
                "sunlight_score": 0.5
            },
            "private": {
                "region": self._get_far_from_entrance_zone(),
                "priority": ["bedroom"],
                "sunlight_score": 0.9  # Bedrooms need good light
            }
        }

        return zones

    def _optimize_sunlight(self, room_assignments):
        """
        Optimize room placement for maximum sunlight exposure

        Strategy:
        1. Calculate sun path (based on north vector)
        2. Priority: Bedrooms > Living > Kitchen > Bathrooms
        3. Place high-priority rooms on sun-facing sides
        """
        north_direction = np.array(self.north_vector)

        # Determine sun-facing side
        # North = (0, 1) means north is up
        # Morning sun = East, Evening sun = West
        east_vector = np.array([north_direction[1], -north_direction[0]])

        sunlight_scores = {}
        for room_id, assignment in room_assignments.items():
            room_type = assignment["type"]
            centroid = assignment["centroid"]

            # Calculate sunlight exposure
            # Rooms on east/south-east get morning sun (better for bedrooms)
            # Rooms on west get evening sun (ok for living room)
            dot_east = np.dot(centroid, east_vector)
            dot_north = np.dot(centroid, north_direction)

            sunlight_score = 0.6 * dot_east + 0.4 * dot_north

            # Weight by room priority
            priority_weight = self.rules[room_type]["sunlight"]
            if priority_weight == "high":
                sunlight_score *= 1.5
            elif priority_weight == "low":
                sunlight_score *= 0.5

            sunlight_scores[room_id] = sunlight_score

        # Reassign positions to maximize total sunlight score
        optimized = self._greedy_sunlight_optimization(
            room_assignments,
            sunlight_scores
        )

        return optimized

    def _cluster_wet_areas(self, room_assignments):
        """
        Minimize plumbing cost by clustering wet areas

        Plumbing Cost Formula:
        cost = Σ (pipe_length * pipe_complexity)

        Strategy:
        1. Find all wet areas (kitchen, bathrooms)
        2. Calculate vertical stacking score
        3. Modify centroids to stack wet areas
        """
        wet_rooms = []

        for room_id, assignment in room_assignments.items():
            room_type = assignment["type"]
            if self.rules[room_type].get("is_wet_area", False):
                wet_rooms.append(room_id)

        if len(wet_rooms) <= 1:
            return room_assignments  # Nothing to optimize

        # Strategy: Stack wet areas vertically or align horizontally
        # Vertical stacking is best (shared plumbing shaft)

        # Option 1: Find optimal wet area cluster center
        cluster_center = self._find_wet_area_cluster_center(wet_rooms)

        # Option 2: Adjust centroids to be close to cluster center
        for room_id in wet_rooms:
            original_centroid = room_assignments[room_id]["centroid"]

            # Pull centroid toward cluster center (but not too much)
            # Maintain 80% of original position, 20% toward cluster
            adjusted_centroid = (
                0.8 * np.array(original_centroid) +
                0.2 * np.array(cluster_center)
            )

            room_assignments[room_id]["centroid"] = tuple(adjusted_centroid)
            room_assignments[room_id]["plumbing_optimized"] = True

        return room_assignments

    def _find_wet_area_cluster_center(self, wet_room_ids):
        """
        Find optimal location for wet area cluster

        Ideal location:
        - Central (minimize max pipe length)
        - Near existing water main (if known)
        - Avoid exterior walls (insulation issues)
        """
        # For now, use geometric center of wet rooms
        # In real system, could use building's water main location

        width, height = self.boundary["width"], self.boundary["height"]

        # Prefer one side (e.g., kitchen wall side)
        # This allows other side for bedrooms
        cluster_center = (width * 0.7, height * 0.5)  # Slightly off-center

        return cluster_center

    def _compute_centroids(self, room_assignments):
        """
        Convert room assignments to final centroid dictionary
        """
        centroids = {
            "living_room": [],
            "bedroom": [],
            "bathroom": [],
            "kitchen": []
        }

        for room_id, assignment in room_assignments.items():
            room_type = assignment["type"]
            centroid = assignment["centroid"]
            centroids[room_type].append(centroid)

        return centroids
```

---

## 🎨 Stage 2: Enhanced GAT-Net Integration

The existing GAT-Net model can be used as-is, but we add:

```python
def predict_with_context(gat_model, centroids, boundary, constraints):
    """
    Enhanced prediction with architectural constraints
    """
    # Use existing GAT-Net
    predictions = gat_model(graph, boundary)

    # Post-process predictions with constraints
    predictions = apply_minimum_sizes(predictions, constraints)
    predictions = adjust_for_aspect_ratios(predictions)

    return predictions
```

---

## 🎯 Stage 3: Geometric Optimization & Output

### **Final Layout Generation**

```python
class EnhancedFloorPlanGenerator:

    def generate_colored_output(self, rooms, dimensions):
        """
        Generate color-coded floor plan with dimensions
        """
        # Color scheme
        colors = {
            "living_room": "#F5E6D3",  # Beige
            "bedroom": "#B4D7E8",       # Blue
            "bathroom": "#E8F4F8",      # Light Blue
            "kitchen": "#FFE4B5",       # Moccasin/Light Orange
            "dining": "#F0E68C",        # Khaki
            "corridor": "#FFFFFF"       # White
        }

        fig, ax = plt.subplots(figsize=(12, 10))

        for room_type, room_list in rooms.items():
            for i, room_polygon in enumerate(room_list):
                # Draw room
                x, y = room_polygon.exterior.xy
                ax.fill(x, y, color=colors[room_type],
                       edgecolor='black', linewidth=2, alpha=0.8)

                # Add label with dimensions
                centroid = room_polygon.centroid
                width = dimensions[room_type][i]["width"]
                height = dimensions[room_type][i]["height"]
                area = width * height

                label = f"{room_type.replace('_', ' ').title()}\n"
                label += f"{width:.1f}m × {height:.1f}m\n"
                label += f"Area: {area:.1f}m²"

                ax.text(centroid.x, centroid.y, label,
                       ha='center', va='center',
                       fontsize=10, weight='bold',
                       bbox=dict(boxstyle='round',
                                facecolor='white',
                                alpha=0.7))

        # Add north arrow
        self._add_north_arrow(ax, north_vector)

        # Add sunlight indicator
        self._add_sunlight_zones(ax)

        # Add plumbing visualization
        self._add_plumbing_lines(ax, wet_rooms)

        return fig

    def _add_plumbing_lines(self, ax, wet_rooms):
        """
        Visualize optimized plumbing connections
        """
        # Draw lines connecting wet areas
        for i in range(len(wet_rooms) - 1):
            p1 = wet_rooms[i].centroid
            p2 = wet_rooms[i+1].centroid

            ax.plot([p1.x, p2.x], [p1.y, p2.y],
                   'b--', linewidth=2, alpha=0.5,
                   label='Plumbing Connection' if i == 0 else '')

        # Add plumbing cost estimate
        total_length = sum(
            wet_rooms[i].centroid.distance(wet_rooms[i+1].centroid)
            for i in range(len(wet_rooms) - 1)
        )

        ax.text(0.02, 0.98,
               f"Plumbing Length: {total_length:.1f}m\n"
               f"Estimated Cost: ${total_length * 50:.0f}",
               transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue'))
```

---

## 📊 Complete Implementation Roadmap

### **Phase 1: Core Heuristics (Week 1-2)**

```python
# New files to create:
├── heuristic_engine.py          # Zone-based placement
├── sunlight_optimizer.py        # Sun path calculations
├── plumbing_optimizer.py        # Wet area clustering
└── enhanced_visualizer.py       # Color-coded output
```

### **Phase 2: Integration (Week 3)**

```python
# Modified files:
├── main.py                      # Add heuristic stage before GAT-Net
├── app.py                       # New UI for simplified input
└── model.py                     # Context-aware predictions
```

### **Phase 3: Advanced Features (Week 4)**

- Corridor generation
- Ventilation scoring
- Privacy optimization
- Multi-floor support

---

## 🧪 Testing Strategy

### **Test Cases:**

```python
# Test 1: Simple rectangular flat
input_1 = {
    "boundary": {"width": 12, "height": 10},
    "rooms": {"living_room": 1, "bedroom": 2, "bathroom": 2, "kitchen": 1},
    "entrance": {"location": "south", "position": 0.5},
    "environment": {"north_vector": (0, 1)}
}

# Expected output:
# - Living room near entrance (south)
# - Bedrooms on north/east (sunlight)
# - Kitchen + bathrooms clustered (west side)
# - Total plumbing < 8m

# Test 2: Narrow rectangular flat
input_2 = {
    "boundary": {"width": 6, "height": 18},
    ...
}

# Expected output:
# - Linear arrangement
# - Living at entrance end
# - Bedrooms at far end
# - Wet areas in middle (minimize plumbing)
```

---

## 💡 Key Algorithms Explained

### **Sunlight Optimization Math:**

```python
def calculate_sunlight_score(room_centroid, north_vector, room_type):
    """
    Sunlight score based on sun path

    Sun path in northern hemisphere:
    - Rise: East (90°)
    - Noon: South (180°)
    - Set: West (270°)

    Best positions:
    - Bedrooms: East (morning sun)
    - Living: South (daytime sun)
    - Kitchen: North (avoid overheating)
    """
    # Convert north vector to angle
    north_angle = np.arctan2(north_vector[1], north_vector[0])

    # Calculate room position relative to north
    room_angle = np.arctan2(room_centroid[1], room_centroid[0])

    # Morning sun direction (East)
    morning_sun_angle = north_angle - np.pi/2

    # Afternoon sun direction (South/Southwest)
    afternoon_sun_angle = north_angle + np.pi

    # Calculate exposure
    morning_exposure = np.cos(room_angle - morning_sun_angle)
    afternoon_exposure = np.cos(room_angle - afternoon_sun_angle)

    # Weight by room type
    if room_type == "bedroom":
        score = 0.7 * morning_exposure + 0.3 * afternoon_exposure
    elif room_type == "living_room":
        score = 0.3 * morning_exposure + 0.7 * afternoon_exposure
    else:
        score = 0.5 * (morning_exposure + afternoon_exposure)

    return max(0, score)  # Clamp to positive
```

### **Plumbing Cost Calculation:**

```python
def calculate_plumbing_cost(wet_room_centroids):
    """
    Minimize: Σ distance(wet_room_i, wet_room_j)
    Subject to: All rooms connected to main water line

    Use Minimum Spanning Tree (MST) algorithm
    """
    from scipy.spatial.distance import pdist, squareform
    from scipy.sparse.csgraph import minimum_spanning_tree

    # Distance matrix
    distances = squareform(pdist(wet_room_centroids))

    # Find MST
    mst = minimum_spanning_tree(distances)

    # Total pipe length = sum of MST edges
    total_length = mst.sum()

    # Cost model: $50 per meter + $200 per connection
    pipe_cost = total_length * 50
    connection_cost = (len(wet_room_centroids) - 1) * 200
    total_cost = pipe_cost + connection_cost

    return {
        "total_length": total_length,
        "pipe_cost": pipe_cost,
        "connection_cost": connection_cost,
        "total_cost": total_cost
    }
```

---

## 🚀 Quick Start Implementation

### **Minimal Viable Product (MVP):**

```python
# Step 1: Create simple heuristic engine
def simple_placement(boundary, room_counts, entrance, north):
    """Quick MVP version"""
    width, height = boundary["width"], boundary["height"]

    # Divide into 3 zones
    zone_height = height / 3

    centroids = {
        "living_room": [(width/2, zone_height * 0.5)],  # Bottom third
        "kitchen": [(width * 0.25, zone_height * 1.5)],  # Middle third
        "bathroom": [
            (width * 0.75, zone_height * 1.5),  # Near kitchen
            (width * 0.75, zone_height * 2.5)   # Near bedrooms
        ],
        "bedroom": [
            (width * 0.25, zone_height * 2.5),  # Top third
            (width * 0.75, zone_height * 2.5)
        ]
    }

    return centroids

# Step 2: Use existing GAT-Net
predictions = model(graph, boundary)

# Step 3: Color-coded output
colors = {"living_room": "beige", "bedroom": "blue", ...}
plot_with_colors(rooms, colors)
```

---

## 📈 Expected Improvements

| Metric                | Current System          | Enhanced System   |
| --------------------- | ----------------------- | ----------------- |
| User effort           | High (manual centroids) | Low (just counts) |
| Sunlight optimization | 0%                      | 60-80% optimal    |
| Plumbing cost         | Random                  | 30-50% reduction  |
| Architectural quality | Medium                  | High              |
| Setup time            | 5-10 minutes            | 30 seconds        |

---

## 🎯 Next Steps

1. **Implement MVP** (1-2 days)
   - Simple zone-based placement
   - Basic color coding
2. **Add Sunlight** (2-3 days)
   - Sun path calculation
   - Room reordering
3. **Add Plumbing** (2-3 days)
   - Wet area clustering
   - MST algorithm
4. **Polish UI** (1-2 days)
   - Streamlit interface
   - Dimension annotations

Total estimated time: **1-2 weeks** for full implementation.
