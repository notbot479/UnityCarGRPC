# UnityCarGRPC - Reinforcement Learning Car Simulation

A **Deep Deterministic Policy Gradient (DDPG)** reinforcement learning project that trains a neural network agent to autonomously navigate and control a simulated car in a virtual environment. The project features real-time communication between a Python RL server and a Unity car simulator using gRPC.

## 🎯 Project Overview

This project demonstrates a complete RL training pipeline:

- **Python Server**: Houses the DDPG agent (Actor-Critic model) that learns optimal control policies
- **Unity Client**: Simulates a car in a 3D environment with sensors, cameras, and navigation tasks
- **gRPC Communication**: Real-time command/response exchange between the agent and simulator
- **Training Loop**: Continuous learning through state observation, action execution, and reward optimization

### Key Features

- **DDPG Agent**: Actor-Critic deep RL implementation using PyTorch
- **Autonomous Navigation**: Car learns to reach target routers and avoid obstacles
- **Multi-Sensor Input**: Distance sensors, camera feed, collision detection, WiFi signal readings (RSSI)
- **Task-Based Learning**: Dynamic task assignment system for diverse training scenarios
- **Model Management**: Automatic checkpointing and restoration of best models
- **TensorBoard Integration**: Real-time training visualization and metrics tracking

## 🏗️ Design & Architecture

### Virtual Environment & Simulation

The project uses **Blender + Unity** combination for 3D modeling and physics simulation:

- **3D Model Inspiration**: Based on Arduino RC car with basic hobbyist components, ensuring accessibility
- **Physics Engine**: SimplePhysics for realistic tire friction and wheel interactions with surfaces
- **Environment**: Warehouse-like setting with navigable aisles and obstacles
- **Colliders**: Mesh Collider for car body, Wheel Collider for realistic wheel physics
- **Camera System**: Cinemachine for dynamic camera control and object tracking

### Sensor Systems

**Distance Sensors (Ultrasonic Simulation)**
- Implemented via Unity Raycast method
- 6 sensors: front-left, front, front-right, back-left, back, back-right
- Range: 2-4 meters (matches HC-SR04 specifications)
- Accuracy: millimeter-level precision
- Invalid readings: `positive infinity` for broken sensors

**Navigation: Zigbee Mesh Network**
- Alternative to expensive LiDAR/mapping systems
- Network of Zigbee routers create flexible mesh topology
- Car receives RSSI (signal strength in dB) from each accessible router
- Navigation algorithm: Graph-based path planning through router network
- Enables location-independent autonomous navigation within facility

**Camera System**
- Image resolution: 224×224 pixels (optimized for neural network training)
- Captured via direct scene rendering to texture
- Can be reduced to 64×64 for lower bandwidth
- Used for object detection and QR code scanning

**QR Code Detection**
- Each warehouse item has unique QR code
- Car scans boxes in target area to identify items
- Task matching: Verifies found item against assigned task

### Communication Architecture

**Why gRPC + Python Server?**
- Python: Centralized RL agent, easier training loop management
- Unity as client: Simulator-only role (physics not thread-safe if server in same thread)
- gRPC: HTTP/2 for low-latency bidirectional streaming
- Trade-off: Async training vs. data race prevention (sync approach chosen)

**YetAnotherHttpHandler Integration**
- Provides HTTP/2 support for Unity (not natively supported)
- Replaces deprecated C-core gRPC with grpc-dotnet
- Supports: TLS 1.2/1.3, ALPN, multiple streams per connection
- Platform support: Unity 2021.3 LTS+, x64/arm64 architectures

## 📁 Project Structure

```
UnityCarGRPC/
├── csharp/                          # C# gRPC client (generated from proto)
│   └── grpc/
│       └── car_communication/       # Protocol buffer definitions
│           └── client/Protos/
├── python/                          # Python RL server implementation
│   ├── pyproject.toml              # Poetry configuration
│   ├── requirements_win.txt         # Python dependencies
│   └── grpc/
│       └── car_communication/
│           ├── server.py           # Main gRPC server & DDPG servicer
│           ├── config.py           # Server configuration
│           ├── ml/                 # Machine learning components
│           │   ├── agent.py        # DDPG Agent implementation
│           │   ├── actor.py        # Actor network (policy)
│           │   ├── critic.py       # Critic network (value function)
│           │   ├── buffer.py       # Experience replay buffer
│           │   ├── noise.py        # Exploration noise (OU process)
│           │   ├── inputs.py       # Model input preprocessing
│           │   └── reward.py       # Reward calculation
│           ├── services/           # Supporting services
│           │   ├── task_manager.py    # Task distribution
│           │   ├── video_manager.py   # Camera frame processing
│           │   └── route_manager.py   # Navigation routing
│           ├── client/             # gRPC client utilities
│           ├── models/             # Trained model checkpoints
│           └── logs/               # TensorBoard logs
└── trunk/                           # Unity project
    ├── Assets/
    │   ├── Scripts/
    │   │   ├── CarCommunication.cs      # Main gRPC client
    │   │   ├── Car/                     # Car control scripts
    │   │   ├── CarSensors/              # Sensor implementations
    │   │   └── Room/                    # Environment setup
    │   ├── Plugins/                     # gRPC/protobuf libraries
    │   └── Resources/                   # 3D models and materials
    ├── trunk.sln                    # Visual Studio solution
    └── ProjectSettings/             # Unity configuration
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.12+
- **Unity**: 2022 LTS or newer
- **Windows/Linux/macOS**

### Setup Python Environment

#### Using Poetry (Recommended)

```bash
cd python
poetry install
poetry shell
```

#### Using pip

```bash
cd python
pip install -r requirements_win.txt
```

**Dependencies:**
- `torch` / `torchvision` / `torchaudio` - Deep learning framework
- `grpcio` / `grpcio-tools` - gRPC communication
- `numpy` - Numerical computing
- `opencv-python` - Image processing
- `tensorboard` - Training visualization

### Running the Python Server

From the `python/grpc/car_communication/` directory:

```bash
python server.py
```

**Expected output:**
```
Starting gRPC server on localhost:50051
Server started successfully and listening...
```

The server will:
1. Initialize the DDPG agent
2. Load the best trained model if available (from `models/` directory)
3. Start listening for Unity client connections on `localhost:50051`
4. Begin logging training metrics to TensorBoard

**Monitor training with TensorBoard:**
```bash
tensorboard --logdir=logs/
```
Then open `http://localhost:6006` in your browser.

### Running the Unity Client

1. **Open the project:**
   ```
   Open trunk/ as a Unity project
   ```

2. **Configure settings** (in Inspector):
   - Open `Assets/Scenes/roomDemo.unity`
   - Select the `CarCommunication` GameObject
   - Adjust these parameters:
     - **Server Domain**: `localhost`
     - **Server Port**: `50051` (must match server config)
     - **Requests Per Second**: `10` (communication frequency)
     - **Move Car By AI**: `true` (enable RL-based control)
     - **Randomize Spawn**: `true` (random car starting positions)

3. **Play the scene:**
   - Press Play in the Unity Editor
   - The car should spawn in the environment
   - Observe gRPC communication starting in the Python server console
   - Training metrics will appear in TensorBoard

## 🧠 How It Works

### Communication Flow

```
┌─────────────────────────────────────────────────────────┐
│                    Unity Client                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ ClientRequest (gRPC)                             │   │
│  │ - Car ID, Position, Speed                        │   │
│  │ - Distance Sensor Data (6 floats, meters)        │   │
│  │ - Camera Image (224×224 JPEG bytes)              │   │
│  │ - Router RSSI Signal Strengths (dB)              │   │
│  │ - Collision Data, QR Code Metadata              │   │
│  └────────────────────────┬─────────────────────────┘   │
│                           │                              │
│              HTTP/2 (gRPC) port 50051                    │
│        ~10 requests/second (100ms latency)              │
│                           │                              │
│  ┌────────────────────────▼─────────────────────────┐   │
│  │ ServerResponse (gRPC)                            │   │
│  │ - Command (LEFT, RIGHT, FORWARD, BACKWARD, etc)  │   │
│  │ - Car Parameters:                                │   │
│  │    * Steering: [-1, 1] normalized               │   │
│  │    * Forward/Backward: [-1, 1] normalized        │   │
│  └────────────────────────────────────────────────┘   │
└──────────────┬────────────────────────────────────────┘
               │
      ┌────────▼──────────┐
      │  Python Server    │
      │  ┌──────────────┐ │
      │  │ DDPG Agent   │ │
      │  ├──────────────┤ │
      │  │ Actor Net    │ │
      │  │ Critic Net   │ │
      │  │ Replay Buf   │ │
      │  │ OU Noise     │ │
      │  └──────────────┘ │
      │  RL Training Loop │
      │  TensorBoard Logs │
      └──────────────────┘
```

### Data Flow & State Processing

**Unity → Server (ClientRequest)**:
1. **Sensor Acquisition** (every frame):
   - 6 distance sensors via Physics.Raycast
   - Camera capture to 224×224 texture
   - Router RSSI readings
   - Collision/QR detection flags

2. **Serialization** (MonoBehaviour.Update):
   - All data packed into protobuf ClientRequest message
   - Image encoded to JPEG bytes
   - Numeric data normalized (0-1 range)

3. **gRPC Transmission** (HTTP/2):
   - Request sent via async gRPC channel
   - YetAnotherHttpHandler handles HTTP/2
   - Typical latency: <100ms

**Server Processing**:
1. **Deserialization**: gRPC auto-parses protobuf message
2. **Data Extraction & Validation**:
   - Filters infinite sensor readings
   - Decodes image bytes to numpy array
   - Normalizes all inputs to [0, 1]
3. **State Composition**: Creates `ModelInputData` object
4. **Network Inference**: Actor network predicts actions
5. **Exploration**: Adds OU noise, clips to [-1, 1]
6. **Response Formation**: Packages action into ServerResponse

**Server → Unity (ServerResponse)**:
1. **Action Encoding**: Steering + throttle values
2. **Protobuf Serialization**: Binary-efficient format
3. **gRPC Transmission**: HTTP/2 response
4. **Unity Application**: Car physics updated with new parameters

### Training Loop

1. **State Collection**: Unity sends sensor data (distance, camera, RSSI, position)
   - Distance sensors: 6 float values (meters)
   - Camera image: 224×224 JPEG bytes
   - Router RSSI: Multiple signal strength readings (dB)
   - Metadata: Collision flags, QR detection, target area status

2. **State Preprocessing**: Server normalizes inputs
   - Image: CNN feature extraction via ResNet backbone
   - Numeric data: Min-max normalization to [0,1] range
   - Invalid readings: Filtered using positive infinity checks

3. **Action Selection**: Actor network outputs continuous control
   - Steering: [-1, 1] (left turn to right turn)
   - Throttle: [-1, 1] (reverse to forward)
   - Exploration: Ornstein-Uhlenbeck noise added for discovery
   - Epsilon decay: Gradually reduces exploration over episodes

4. **Command Execution** (Unity Physics):
   - Apply steering angle to wheels
   - Apply throttle to forward/reverse motion  
   - Physics engine updates: Position, velocity, collisions
   - Raycast sensors measure new distances
   - New state obtained after ~100ms (10 requests/second)

5. **Reward Calculation**:
   - Assess progress toward target router
   - Penalize collisions
   - Bonus for task completion
   - Discount factor: γ (typically 0.99)

6. **Experience Storage** (Replay Buffer):
   - Tuple (state, action, reward, next_state, done) added
   - Circular buffer of max 1M transitions
   - Later: Random batch sampling for training

7. **Batch Training** (Separate Thread - prevents timeout):
   - **Batch Size**: 64 samples from replay buffer
   - **Critic Update**:
     - Target Q: $y = r + \gamma Q_{target}(s', \mu_{target}(s'))$
     - Minimize MSE loss
   - **Actor Update**:
     - Policy loss: $L = -\text{mean}(Q(s, \mu(s)))$
   - **Soft Target Update**: $\theta' \leftarrow \tau \theta + (1-\tau) \theta'$

8. **Episode Completion**:
   - Aggregate metrics and log to TensorBoard
   - Reset environment for next episode
   - Decrease exploration: epsilon × decay_rate

### Reward System (Advanced Shaping)

The agent optimizes based on multiple weighted components:
- **Target Proximity**: Negative reward scaled by distance to target router
- **Waypoint Progress**: Bonus for moving through intermediate waypoints in mesh network
- **Collision Penalty**: Large negative value for hitting obstacles
- **Efficiency Penalty**: Proportional to steering magnitude + throttle magnitude (discourage erratic behavior)
- **Task Completion**: +100 reward upon finding correct QR code
- **Time Penalty**: Small negative per timestep (encourages completion speed)

**Reward Composition**:
$$r_t = r_{proximity} + r_{progress} + r_{collision} + r_{efficiency} + r_{task} + r_{time}$$

This multi-component design encourages:
- Smart pathfinding (proximity + waypoint rewards)
- Collision avoidance (penalty)
- Smooth driving (efficiency penalty for jerky inputs)
- Quick task completion (time penalty)

## 📊 Neural Network Architecture

### Actor Network (Policy)
- **Input**: Concatenated state vector
  - CNN features from 224×224 camera image (via Conv2d layers)
  - Normalized numeric inputs: distances (6), RSSI (N), position, speed
- **Hidden Layers**: Fully connected with ReLU activation
- **Output**: 2 continuous values (steering, throttle) with tanh activation [-1, 1]
- **Purpose**: Learn deterministic policy μ(s) for state-to-action mapping

### Critic Network (Value Function)
- **Input**: State concatenated with Actor's actions
  - Same CNN features + numeric data as Actor
  - Plus: Action values from Actor output
- **Hidden Layers**: Fully connected with ReLU activation
- **Output**: Single Q-value prediction (scalar)
- **Purpose**: Estimate expected cumulative reward Q(s,a)

### Target Networks
- **Purpose**: Stabilize training via delayed updates
- **Update Rule**: Soft update with tau parameter (~0.001)
- **Benefits**: Prevents divergence, reduces overestimation

### Exploration Strategy

**Ornstein-Uhlenbeck Noise**
- Generates correlated noise mimicking physical acceleration constraints
- Stateful: Current noise influenced by previous noise values
- Scaled by decreasing epsilon: Less exploration over training
- Final epsilon: ~0.01 (maintains some stochasticity)

**Data Normalization**
- Prevents high-magnitude features from dominating loss
- Speeds up convergence, improves stability
- Applied to: All sensor inputs, images, numeric data

## 🎓 Training Details

### Exploration vs. Exploitation
- **Combined approach**: Noise-based exploration (good for continuous spaces) + epsilon decay
- **Noise scaling**: Clipped to [-1, 1] action bounds to prevent invalid commands
- **Decay schedule**: Reduces noise influence as agent confidence increases

### Batch Training Setup
- **Frequency**: Training occurs after episode completion (not per-step)
- **Batch Size**: 64 transitions sampled from replay buffer
- **Update Rate**: Multiple gradient steps per episode
- **Worker Threads**: Separate thread for training to avoid client timeout

### Loss Functions
**Critic Loss**:
$$L_{critic} = MSE(Q(s,a), r + \gamma Q_{target}(s', a'))$$

**Actor Loss**:
$$L_{actor} = -mean(Q(s, \mu(s)))$$

### Monitoring & Metrics
- **Per-Episode**: Cumulative reward, episode length, success flag
- **Aggregated**: Moving average (smoothed over 10 episodes)
- **Network Stats**: Actor/Critic loss values, Q-value ranges
- **Visualization**: Real-time TensorBoard graphs for comparison

## 🧠 How It Works

### Reward System

The agent learns to maximize cumulative reward based on:
- **Target Proximity**: Reward for moving closer to target router
- **Navigation Success**: Reward for reaching designated areas
- **Collision Avoidance**: Penalty for collisions with obstacles
- **Efficiency**: Penalty for excessive movement/energy usage

## 📊 Neural Network Architecture

### Actor Network (Policy Network)
- **Input**: Concatenated state representation
  - CNN backbone from 224×224 camera image (Conv2d layers)
  - Normalized numeric inputs: 6 distance sensors, N router RSSI values, position, speed
- **Hidden Layers**: Fully connected layers with ReLU activation
- **Output**: 2 continuous values (steering, throttle) with tanh activation [-1, 1]
- **Purpose**: Learn deterministic policy μ(s) for state-to-action mapping
- **Architecture**: Conv layers + FC layers combining visual and numeric features

### Critic Network (Q-Value Network)
- **Input**: State concatenated with Actor's action output
  - Same CNN features + numeric data as Actor
  - Plus: Action vector from Actor network
- **Hidden Layers**: Fully connected layers with ReLU activation
- **Output**: Single Q-value prediction (scalar)
- **Purpose**: Estimate expected cumulative reward Q(s,a)
- **Loss**: Mean Squared Error (MSE) between predicted and target Q-values

### Target Networks
- **Auto-Stabilization**: Delayed updates prevent divergence
- **Soft Update Rule**: $w_{target} \leftarrow \tau w_{current} + (1-\tau) w_{target}$
- **Tau Parameter**: ~0.001 (small update rate)
- **Benefit**: Reduces overestimation bias, improves training stability

### Exploration Strategy: Ornstein-Uhlenbeck Process
- **Type**: Correlated noise (unlike random Gaussian noise)
- **Characteristics**: Mimics physical acceleration constraints
- **State Dependent**: Current noise influenced by previous noise values
- **Decay**: Epsilon parameter decreases over training episodes (0.01 final)
- **Clipping**: Actions clipped to [-1, 1] bounds via numpy clip

## ⚙️ Training Configuration

### Exploration vs. Exploitation Balance
- **Combined Approach**: Noise-based exploration + epsilon decay
- **Advantage**: Works well for continuous action spaces (unlike ε-greedy for discrete)
- **Noise Scaling**: Gradually reduced as agent confidence increases
- **Minimum Exploration**: Maintain epsilon ~0.01 for ongoing discovery

### Batch Training Pipeline
- **Timing**: Training occurs after episode completion (not per-step)
- **Buffer Size**: 64 random samples from replay buffer
- **Update Frequency**: Multiple gradient steps per episode
- **Threading**: Separate thread for training prevents client timeout
- **Data Decorrelation**: Random sampling breaks temporal correlations

### Loss Functions
**Critic Loss (Value Function)**:
- Target Q-value: $y = r + \gamma Q_{target}(s', \mu_{target}(s'))$
- Loss: $L_{critic} = MSE(Q(s,a) - y)$
- Optimization: Minimize prediction error via gradient descent

**Actor Loss (Policy Gradient)**:
- Loss: $L_{actor} = -mean(Q(s, \mu(s)))$
- Objective: Maximize expected reward (negative loss)
- Optimization: Policy gradient via deterministic actor

### Training Metrics & Monitoring
- **Per-Episode**: Cumulative reward, episode length, success/failure
- **Aggregated**: Moving average over 10 episodes (smoothing)
- **Network Stats**: Actor/Critic loss trends, Q-value statistics
- **Visualization**: TensorBoard for real-time comparison
- **Logging**: Frame-by-frame data sent to console (diagnostic mode)

## ⚙️ Configuration

Edit `python/grpc/car_communication/config.py` to customize:

```python
# Service settings
SHOW_STREAM_VIDEO = False          # Display camera feed in server
SHOW_CLIENT_DATA = False           # Print client data logs

# gRPC server
PORT = 50051                       # Server port
MAX_WORKERS = 60                   # Max concurrent connections

# Agent paths
AGENT_MODELS_PATH = "models"       # Model checkpoint directory
AGENT_LOGS_PATH = "logs"           # TensorBoard logs directory
```

Edit `python/grpc/car_communication/server.py` class attributes to tune training:

```python
# Training hyperparameters
_agent_episodes_count = 10000           # Total training episodes
_agent_exploration_seconds = 30         # Exploration duration
_agent_train_batch_size = 64            # Batch size for training
_agent_epsilon_decay = 0.9995           # Policy exploration decay

# Environment settings
_env_requests_per_second = 10           # Unity communication frequency
_car_respawn_on_object_hit = True       # Auto-respawn on collision
_car_target_patience = 5                # Frames to find target before reset
```

## 📊 Training Metrics

The server logs to TensorBoard:
- **Cumulative Reward**: Total reward per episode
- **Average Q-Values**: Agent's value estimates
- **Loss**: Actor and Critic network losses
- **Success Rate**: Percentage of episodes reaching target
- **Collision Rate**: Obstacle collisions per episode

View with:
```bash
tensorboard --logdir=logs/
```

## 🔧 Troubleshooting

### Proto File Synchronization (CRITICAL)
- **Issue**: Any mismatch between Python and C# `.proto` files causes RPC failures
- **Solution**: Regenerate code whenever modifying `.proto`
- **Common Mistake**: Renaming service methods breaks compatibility (RPC names immutable)
- **Best Practice**: Keep both versions in version control, regenerate together

### Server Blocking During Training
- **Issue**: gRPC client timeout if server trains synchronously
- **Solution**: Use separate thread for batch training or train after episode completion
- **Trade-off**: Per-step training faster but risks client disconnect (separate thread approach used)
- **Monitor**: Check server logs for "client timeout" warnings

### Server Won't Start
```bash
# Check if port is in use
netstat -ano | findstr :50051  # Windows
lsof -i :50051                  # macOS/Linux

# Kill existing process if needed
taskkill /PID <PID> /F          # Windows
kill -9 <PID>                   # macOS/Linux
```

### Unity Can't Connect
- Verify server is running: `python server.py`
- Check firewall allows port 50051
- Ensure server domain is correct in CarCommunication script
- Verify `.proto` files match between projects (proto synchronization critical!)
- Check server logs for serialization errors

### Poor Training Results / Agent Not Learning
- **Action Space**: Verify steering/throttle bounded to [-1, 1]
- **Reward Shape**: Check reward values not too sparse (update `ml/reward.py`)
- **Exploration**: Increase `_agent_exploration_seconds` for more discovery
- **Learning Rate**: Reduce `_agent_train_batch_size` for more frequent updates
- **Network Capacity**: Consider adding layers if model underfits
- **Data Normalization**: Verify sensor readings normalized to [0, 1]

### Camera Feed Not Processed
- Image resolution must match model input (224×224 expected)
- Check JPEG encoding/decoding in `video_manager.py`
- Verify bytes transmitted correctly via gRPC (`ByteString` type)
- Monitor: Use `SHOW_STREAM_VIDEO = True` in config to debug

### Sensor Readings Invalid
- Sensors return `positive infinity` when broken or out of range
- Preprocessing should filter these values
- Check collision layer masks in Unity match sensor raycast masks
- Verify raycast distance set correctly (2-4 meters for HC-SR04 equivalent)

## 📚 Key Components

### DDPG Agent (`ml/agent.py`)
Implements Deep Deterministic Policy Gradient for continuous action spaces:
- **Actor Network**: Maps states to continuous steering + throttle actions
- **Critic Network**: Estimates state-action value (Q-function) for reward prediction
- **Replay Buffer**: Stores and samples transitions for efficient batch learning
- **Target Networks**: Delayed copies for stability via soft update (tau=0.001)
- **Exploration**: Ornstein-Uhlenbeck noise decay for exploration-exploitation trade-off

### Replay Buffer (`ml/buffer.py`)
- **Experience Storage**: Circular buffer of last N transitions
- **Sampling**: Random batch selection for decorrelation
- **Memory Efficiency**: Only stores necessary (state, action, reward, next_state, done)

### Input Processing (`ml/inputs.py`)
- **Image Processing**: CNN feature extraction from raw camera frames
- **Normalization**: Min-max scaling of numeric inputs to [0, 1]
- **Concatenation**: Combines visual and numeric features into single input vector
- **Validation**: Handles invalid readings (sensor failures, out-of-range values)

### Services

**task_manager.py**: Navigation task assignment
- Mesh network routing: Calculates shortest path through router graph
- Task dispatch: Assigns target locations and QR codes to find
- Status tracking: Monitors completion and failures

**video_manager.py**: Camera frame processing
- Image capture: Renders scene to 224×224 texture
- Frame encoding: Converts to JPEG bytes for transmission
- QR detection: Scans for item identification codes

**route_manager.py**: Path planning
- Graph algorithms: Dijkstra/A* through router mesh
- Waypoint generation: Intermediate targets between routers
- Replanning: Updates path if obstacles encountered

## 📝 Protobuf Communication

Communication schema defined in `car_communication.proto`:

```protobuf
service Communication {
  rpc SendRequest(ClientRequest) returns (ServerResponse);
}
```

**ClientRequest**: Car telemetry & sensor data  
**ServerResponse**: AI commands & control parameters

Generated code in:
- Python: `Protos/car_communication_pb2.py` & `car_communication_pb2_grpc.py`
- C#: `Assets/Scripts/Protos/`

## 🎓 Learning Resources

- [DDPG Paper](https://arxiv.org/abs/1509.02971) - Algorithm details
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Unity gRPC Integration](https://grpc.io/docs/languages/csharp/)
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard/get_started)
- [Ornstein-Uhlenbeck Process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) - Exploration noise

## 📖 Project Background

- **Research Goal**: Implement practical RL training pipeline for continuous control
- **Academic Focus**: Deep reinforcement learning (DDPG), neural network architecture design, distributed system communication
- **Real-World Inspiration**: Arduino RC car model adapted into virtual simulation
- **Technical Stack**: Blender (modeling) → Unity (physics simulation) ↔ gRPC ↔ Python (ML)
- **Key Contributions**:
  - Navigation via Zigbee mesh RSSI (alternative to expensive LiDAR)
  - Warehouse environment with collision avoidance
  - Complete training pipeline from sensor input to RL optimization
  - Cross-platform compatibility (Windows, macOS, Linux)


## Development Branches

This repository contains multiple experimental branches exploring different neural network architectures and RL algorithms:

| Branch | Status | Purpose |
|--------|--------|---------|
| **master** | Default | Stable release version (baseline) |
| **ddpg-torch** | Active | DDPG with PyTorch |
| **ddpg-resnet18** | Experimental | DDPG with ResNet18 CNN backbone for visual processing |
| **_old-ddpg-resnet18** | Archived | Legacy ResNet18 implementation |
| **dqn-tf** | Archived | DQN algorithm using TensorFlow (older approach) |
| **unity-grpc-server** | Archived | Initial gRPC server setup |
