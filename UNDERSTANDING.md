# 🏠 Floor Plan Generator - Complete Workflow Explanation

## 📥 THE INPUT

### What the User Provides:

#### 1. **Boundary Polygon** (WKT Format String)

```python
boundary_wkt = "POLYGON ((25.6 65.3, 200.4 65.3, 200.4 75.9, ...))"
```

- **What it is:** The outer walls of your floor plan (like drawing the outline of your house)
- **Coordinates:** (x, y) points that define the shape
- **Example:** An L-shaped or rectangular outline

#### 2. **Front Door Polygon** (WKT Format String)

```python
front_door_wkt = "POLYGON ((38.4 179.7, 63.6 179.7, ...))"
```

- **What it is:** Location and size of the entrance/main door
- **Purpose:** Tells the model where people enter the house

#### 3. **Room Centroids** (List of Points)

```python
room_centroids = [(201, 163), (193, 106)]
```

- **What it is:** Center points (x, y) for bedrooms/living rooms
- **Meaning:** "I want 2 rooms approximately at these locations"

#### 4. **Bathroom Centroids** (List of Points)

```python
bathroom_centroids = [(91, 91), (52, 95)]
```

- **What it is:** Center points for bathrooms
- **Meaning:** "I want 2 bathrooms approximately here"

#### 5. **Kitchen Centroids** (List of Points)

```python
kitchen_centroids = [(137, 89)]
```

- **What it is:** Center points for kitchen(s)
- **Meaning:** "I want 1 kitchen approximately here"

### Visual Input Example:

```
Input Example 1:
┌─────────────────────┐
│                     │  ← Boundary (outer walls)
│  🚪 ← Door         │
│                     │
│  🛏️ Room 1         │
│  🛁 Bath 1          │
│  🍳 Kitchen         │
│  🛏️ Room 2         │
│  🛁 Bath 2          │
│                     │
└─────────────────────┘
User provides ONLY:
- Outline shape
- Door location
- Approximate center points for rooms/bathrooms/kitchen
```

---

## ⚙️ WHAT HAPPENS (The Processing Pipeline)

### **STEP 1: Data Loading** (`get_example()` or user input)

```python
# Load the 5 inputs mentioned above
boundary_wkt, front_door_wkt, room_centroids, bathroom_centroids, kitchen_centroids = get_example('EX 1')
```

### **STEP 2: Convert to Geometry** (`preProcessing_toGraphs()`)

#### 2.1 Parse WKT Strings

```python
Boundary = shapely.wkt.loads(boundary_wkt)      # String → Polygon object
front_door = shapely.wkt.loads(front_door_wkt)  # String → Polygon object
```

#### 2.2 Flip Y-Axis (Coordinate System Adjustment)

```python
Boundary = scale(Boundary)  # Flip for screen coordinates
```

- **Why:** Computer screens have Y increasing downward, we flip to match

#### 2.3 Create Living Room Centroid

```python
living_centroid = [(Boundary.centroid.x, Boundary.centroid.y)]
```

- **Automatically adds:** A living room at the center of the boundary

#### 2.4 Organize User Constraints

```python
user_constraints = {
    'living': living_centroid,      # 1 living room (auto-created)
    'room': room_centroids,          # User's bedrooms
    'bathroom': bathroom_centroids,  # User's bathrooms
    'kitchen': kitchen_centroids     # User's kitchens
}
```

### **STEP 3: Create Two Graphs** (Graph Neural Network Representation)

#### Graph 1: **Boundary Graph** (`B_n`)

```python
B_n = Handling_dubplicated_nodes(Boundary, front_door)
```

**What it contains:**

- **Nodes:** Each corner/vertex of the boundary + front door
- **Node Features:**
  - Type (0 = wall corner, 1 = door)
  - Centroid (x, y coordinates)
- **Edges:** Connect adjacent corners (walls between corners)
- **Edge Features:**
  - Distance between corners

**Visual Example:**

```
     Node 0 ●─────● Node 1
            │     │
            │     │
   Node 3 ●│     │● Node 2
          │       │
    🚪 Door Node  │
          │       │
          └───────┘
```

#### Graph 2: **Room Graph** (`G_n`)

```python
G_n = centroids_to_graph(user_constraints, living_to_all=True)
```

**What it contains:**

- **Nodes:** Each room/bathroom/kitchen/living room
- **Node Features:**
  - Room type (0=living, 1=room, 2=kitchen, 3=bathroom)
  - Centroid X coordinate
  - Centroid Y coordinate
- **Edges:** All rooms connected to living room (star topology)
- **Edge Features:**
  - Distance between room centroids

**Visual Example:**

```
       Room 1 ●
              ╱
             ╱
Living ●────● Kitchen
    ╲
     ╲
      ● Bathroom 1
       ╲
        ● Room 2
         ╲
          ● Bathroom 2
```

### **STEP 4: Convert to PyTorch Geometric Data**

```python
B = from_networkx(B_n, ...)  # NetworkX → PyTorch Geometric
G = from_networkx(G_n, ...)  # NetworkX → PyTorch Geometric
```

**Feature Engineering:**

- **Normalize** centroids (mean=0, std=1) for better neural network training
- **One-hot encode** room types:
  - Living: [1,0,0,0,0,0,0]
  - Room: [0,1,0,0,0,0,0]
  - Kitchen: [0,0,1,0,0,0,0]
  - Bathroom: [0,0,0,1,0,0,0]

**Final Graph Features:**

```python
G.x = [7 one-hot encoded room types, normalized_x, normalized_y]  # Shape: [num_rooms, 9]
B.x = [type, normalized_x, normalized_y]  # Shape: [num_boundary_nodes, 3]
```

### **STEP 5: Load Trained Model**

```python
model = load_model("GAT-Net_v3_UnScalled.pt", device)
```

**Model Architecture (GAT-Net):**

```
Input: Room Graph (G) + Boundary Graph (B)

Room Graph Processing:
  G → GATConv1[4 heads] → Concat with original →
  G → GATConv2[8 heads] → Concat with original →
  G → GATConv3[8 heads] → Concat with original →
  G → GATConv4[8 heads] → Concat with original →
  [Shape: nodes × 1033 features]

Boundary Graph Processing:
  B → GATConv1[4 heads] → Concat with original →
  B → GATConv2[8 heads] → Concat with original →
  B → MaxPool (all nodes → 1 vector) →
  [Shape: 1 × 259 features]

Combine:
  Concatenate [Room features + Boundary features (repeated for each room)]
  → GATConv fusion layer
  → Split into two branches:

  Width Branch:              Height Branch:
  Linear(1024 → 128)        Linear(1024 → 128)
  → Dropout                  → Dropout
  → Linear(128 → 1)         → Linear(128 → 1)

Output: [width, height] for each room
```

**What the Model Learns:**

- **Room sizes** based on room type (bathrooms are smaller than living rooms)
- **Spatial relationships** (rooms near each other should fit together)
- **Boundary constraints** (rooms must fit within the outline)
- **Proportions** (bedroom is typically 70-80% of living room size)

### **STEP 6: Model Inference**

```python
prediction = model(G.to(device), B.to(device))
w_predicted = prediction[0]  # Predicted widths for each room
h_predicted = prediction[1]  # Predicted heights for each room
```

**Example Output:**

```python
Room Layout Predictions:
├─ Living Room:   width=50.2, height=45.8
├─ Room 1:        width=35.1, height=30.2
├─ Room 2:        width=33.8, height=28.9
├─ Bathroom 1:    width=15.5, height=12.3
├─ Bathroom 2:    width=14.2, height=11.8
└─ Kitchen:       width=25.6, height=20.1
```

### **STEP 7: Create Rectangles from Predictions**

```python
output = FloorPlan_multipolygon(G_not_normalized, prediction)
```

**For each room:**

1. Take the centroid (x, y)
2. Use predicted width/height
3. Create rectangle:
   ```python
   x1 = centroid_x - width/2
   x2 = centroid_x + width/2
   y1 = centroid_y - height/2
   y2 = centroid_y + height/2
   box = Polygon([(x1,y1), (x2,y1), (x2,y2), (x1,y2)])
   ```

### **STEP 8: Handle Overlaps and Fit to Boundary**

```python
polygons = output.get_multipoly(Boundary_as_polygon, the_door)
```

**Smart Overlap Resolution:**

1. **Clip to boundary:**

   ```python
   room = room.intersection(boundary.buffer(-3))  # Shrink boundary by 3 units
   ```

2. **Bathroom inside room?**

   ```python
   if bathroom.intersects(room):
       if intersection.area >= 20% of bathroom:
           # Keep bathroom inside room
           bathroom = bathroom.intersection(room.buffer(-3))
       else:
           # Cut room to avoid bathroom
           room = room.difference(intersection.buffer(0.3))
   ```

3. **Room overlaps with room?**
   ```python
   if room1.intersects(room2):
       # Cut from the smaller room
       smaller_room = smaller_room.difference(intersection.buffer(4))
   ```

### **STEP 9: Visualization**

```python
polygons.plot(cmap='twilight', figsize=(4, 4), ...)
plt.savefig("Outputs/model_output.png")
```

**Color Coding:**

- Each room type gets a different color
- Rooms, bathrooms, kitchens are visually distinct

---

## 📤 THE OUTPUT

### **What You Get:**

#### 1. **model_output.png** - The Final Floor Plan

```
┌─────────────────────────────────┐
│  ┌──────────┐                   │
│  │ Bathroom │                   │
│  └──────────┘                   │
│                                  │ ← Boundary
│  ┌─────────────────────┐        │
│  │                     │        │
│  │    Living Room      │        │
│  │                     │        │
│  └─────────────────────┘        │
│                                  │
│  ┌─────────┐  ┌──────────────┐ │
│  │ Kitchen │  │   Bedroom    │ │
│  └─────────┘  └──────────────┘ │
│                                  │
│  ┌──────────┐                   │
│  │ Bathroom │                   │
│  └──────────┘                   │
└─────────────────────────────────┘
    🚪 ← Door
```

**Features:**

- ✅ All rooms properly sized
- ✅ Rooms fit within boundary
- ✅ Minimal overlaps
- ✅ Doors positioned correctly
- ✅ Realistic proportions

#### 2. **boundary.png** - Just the Outline

Shows the input boundary and door

#### 3. **user_inputs.png** - Centroids Visualization

Shows the boundary with colorful dots representing room centers

#### 4. **both_graphs.png** - Network Visualization

Shows the two graphs (boundary graph + room graph) as networks

### **Data Returned:**

```python
path, B_n, G_n = Run(boundary, door, rooms, baths, kitchen)
```

- `path`: File path to the output image
- `B_n`: NetworkX boundary graph (for analysis)
- `G_n`: NetworkX room graph (for analysis)

---

## 🔬 HOW THE MODEL MAKES PREDICTIONS

### **Training Data:**

- Trained on **80,000 real floor plans** from RPlan dataset
- Each floor plan converted to graphs
- Model learned patterns like:
  - "Bathrooms are typically 10-20 square meters"
  - "Living rooms are usually the largest room"
  - "Rooms adjacent to each other should have compatible sizes"

### **Attention Mechanism:**

The GAT (Graph Attention Network) learns which rooms should "pay attention" to each other:

```
When predicting Bathroom size:
  ┌─ Living room: 10% attention
  ├─ Kitchen:     15% attention
  ├─ Bedroom:     60% attention  ← High! (bathrooms often near bedrooms)
  └─ Boundary:    15% attention
```

### **Why It Works:**

1. **Graph structure** captures spatial relationships
2. **Attention** focuses on relevant neighbors
3. **Boundary info** constrains room sizes
4. **Residual connections** preserve original centroid information through layers

---

## 📊 COMPLETE EXAMPLE

### Input:

```python
Boundary: L-shaped house outline (256x256 grid)
Door: Bottom-left entrance
Rooms: 2 bedrooms at (201,163) and (193,106)
Bathrooms: 2 bathrooms at (91,91) and (52,95)
Kitchen: 1 kitchen at (137,89)
```

### Processing:

```
1. Parse geometries ✓
2. Create 6 nodes: living + 2 rooms + 2 baths + 1 kitchen
3. Build graphs:
   - Boundary: 8 corner nodes + 1 door node
   - Rooms: 6 room nodes, living connected to all
4. Normalize features
5. Model processes:
   - 4 GAT layers on room graph
   - 2 GAT layers on boundary
   - Fusion layer combines both
   - Predict sizes
6. Generate rectangles
7. Resolve overlaps
8. Clip to boundary
```

### Output:

```
Final Floor Plan (saved to Outputs/model_output.png):
- Living: 48×44 units
- Bedroom 1: 34×29 units
- Bedroom 2: 33×28 units
- Bathroom 1: 15×12 units (inside Bedroom 1)
- Bathroom 2: 14×11 units
- Kitchen: 25×20 units
Total area: ~85% of boundary filled
```

---

## 🎯 Key Takeaways

| Aspect               | Details                                                |
| -------------------- | ------------------------------------------------------ |
| **Input Complexity** | 5 simple inputs (1 outline + 1 door + lists of points) |
| **Processing**       | Graph neural networks analyze spatial relationships    |
| **Intelligence**     | Model learned from 80K real floor plans                |
| **Output**           | Full 2D floor plan with sized & positioned rooms       |
| **Time**             | 1-2 seconds on CPU                                     |
| **Accuracy**         | Realistic room sizes and minimal overlaps              |

The model essentially acts as an **AI architect** that knows:

- How big rooms should be
- How they should relate to each other spatially
- How to fit everything within the boundary
- How to create livable, realistic floor plans

**Magic happens** because the Graph Attention Network can "reason" about:

- "This bathroom is near this bedroom → make them compatible sizes"
- "This is a corner location → adjust room shape to fit"
- "The boundary is narrow here → make rooms longer and thinner"
