# Instructions for Humanoid Walking with Donet (Dynamic Oscillatory Neural Embodied Tuning)

## What We Want to Build

We want to create a network for humanoid walking in MuJoCo that works differently from traditional reinforcement learning. Instead of learning policies directly, we want to use the ideas from REMI paper but adapt them into Donet for continuous robot control where planning and execution cannot be separated.

## Understanding Phase Variables

In the brain, different types of cells represent different kinds of variables:

**Oscillatory phases** are variables that go around in circles, like a clock. In the brain, grid cells and direction cells work this way. For robots, these would be things like rotating joints that can spin continuously. The math looks like:

$$
\phi_{t+1} = f_{\text{osc}}(\phi_t, h_t) \bmod 2\pi
$$

This means the phase can keep growing but wraps around every 2π, like going from 359 degrees back to 0 degrees.

**Saturating phases** are variables that hit limits and stop, like filling up a cup. In the brain, border cells work this way. For robots, these are things like joint limits (your elbow can only bend so far) or gripping strength (you can only squeeze so hard). The math looks like:

$$
\psi_{t+1} = \text{clamp}(f_{\text{sat}}(\psi_t, h_t), 0, 2\pi)
$$

This means the variable gets stuck at its limits rather than wrapping around.

## The Core Problem

REMI works great for spatial navigation because it can separate exploration and planning into different phases. During exploration, the robot moves around and builds up memories. During planning, it can "imagine" moving without actually moving, replaying those memories to plan a path.

But in real robotics, especially for walking, you cannot separate these phases. The robot needs to plan while it walks, and it needs to walk while it plans. This creates a chicken-and-egg problem that REMI was not designed to handle.

## Our Solution Approach

We want to remove the oscillatory phases (no need for them in walking) and focus on saturating phases for joints. Each joint will have phase variables that represent things like position limits, velocity limits, and force limits.

We will add many more sensory inputs than REMI had. The robot will have sensors on every body part to feel where it is in space relative to its own body (egocentric sensing). This includes things like:

- Where is my head relative to my torso?
- How are my feet touching the ground?
- What forces am I feeling in each joint?

We will use harmonic expansion on these phase variables. This means instead of just using the raw phase value, we use sine and cosine functions of the phase to create richer representations.

## The Network Design

We want to think of this as having two types of network states (not necessarily two separate networks):

**Network State Type 1: Encoder/Dynamics States**
These network state variables represent the current situation of the robot. They include:

- Each joint represented as a saturating phase variable
- Perceptual states that process all the sensor information (like pose estimators for each limb)
- Association states that work like place cells in the brain, connecting different pieces of information together

When the robot is exploring how to move its joints, it can generate fake actions to drive these states. This part of the network records both the actions being input and the resulting states (joint angles, perceptual feelings, etc.) during execution. This serves as an "encoder" of the dynamics of both the humanoid and the world.

**Network State Type 2: Action Generation States**
These states serve as high-dimensional planning variables. Their purpose is to generate "fake" movements that can drive the dynamics in the encoder states. This lets the agent examine the resulting humanoid states to validate trajectories - like playing things out in the mind first before acting.

## How REMI Works with RNNs

REMI uses a large RNN network that combines multiple types of states. A normal RNN updates its hidden state like this:

$$
z_{t+1} = \alpha \cdot z_t + (1 - \alpha) \cdot 
\left( W^{in} u_t + W^{rec} f(z_{t-1}) \right)
$$

where $z$ is the hidden state, $u$ is the input, $\alpha$ controls memory, and the output is $y_t = W^{out} z_t$.

REMI extends this by adding special input and output sections. Instead of just $z$, we have three parts: $z$ (main states), $z^I$ (input states), and $z^O$ (output states):

$$
\begin{bmatrix}
z_{t+1} \\
z^I_{t+1} \\
z^O_{t+1}
\end{bmatrix}
=
\alpha \odot
\begin{bmatrix}
z_t \\
z^I_t \\
z^O_t
\end{bmatrix}
+
(1-\alpha) \odot
\left(
\begin{bmatrix}
0 \\ u_t \\ 0
\end{bmatrix}
+
\begin{bmatrix}
W^{rec} & \tilde{W}^{in} & W^{(13)} \\
W^{(21)} & W^{(22)} & W^{(23)} \\
\tilde{W}^{out} & W^{(32)} & W^{(33)}
\end{bmatrix}
\cdot
f\!\left(
\begin{bmatrix}
z_t \\ z^I_t \\ z^O_t
\end{bmatrix}
\right)
\right)
$$

The $z^I$ states directly receive inputs $u_t$, while $z^O$ states are supervised to match known brain cell responses. The $\alpha$ values can be different for each state, allowing some to remember longer than others.

REMI trains the encoder first by exploring environments. The input states learn to copy and predict sensory inputs like speed and visual observations. Sometimes these inputs are masked out, forcing the network to fill in missing information. Grid cells emerge naturally as oscillatory patterns, and some output states are trained to match place cell responses from brain data.

After encoder training, REMI adds an action generator by expanding the connectivity matrix from $N_e \times N_e$ to $(N_e + N_a) \times (N_e + N_a)$, where $N_a$ is the number of action states.

During planning, the system receives a goal state (grid cell patterns without perceptual details). The encoder states start at the current position, and the action network generates movements that drive the encoder toward the goal. The state of the planning network can be regard as an encoding of the transition pair $(s_{current}, s_{goal})$. The action states only project to movement commands, while the encoder only sends its phase variables to the action network.

## Generalizing REMI to Humanoid Walking with Donet GRU

We adapt REMI's RNN architecture into Donet using GRUs for controlling a humanoid robot in MuJoCo. Instead of spatial navigation with grid cells, we focus on joint control with sparse-encoded joint angles and rich perceptual information.

### Joint Angle Encoding with Difference of Gaussians

Rather than harmonic expansion, we use sparse encoding for joint angles with Difference of Gaussians (DoG) tuning curves. Each joint angle $\theta \in [0, 2\pi]$ is encoded using:

$$
s_i(\theta) = \exp\left(-\frac{(\theta - \mu_i)^2}{2\sigma_1^2}\right) - k \cdot \exp\left(-\frac{(\theta - \mu_i)^2}{2\sigma_2^2}\right)
$$

where $\mu_i$ are evenly distributed preferred angles, $\sigma_1 < \sigma_2$ (narrow excitation, broader inhibition), and $k < 1$ is the inhibition strength. This creates sharper tuning, lateral inhibition between neighboring units, and sparser activation patterns. For $N$ joints and $K$ encoding units per joint, we get $N \times K$ sparse encoding units.

### GRU Architecture for Humanoid Control

The GRU state vector is partitioned into specialized functional groups, all using tanh activation to allow bipolar saturating dynamics:

$$
h_t = \begin{bmatrix}
h_t^{joint} \\
h_t^{percept} \\
h_t^{place} \\
h_t^{planning} \\
h_t^{torque}
\end{bmatrix}
$$

**Joint State Units ($h_t^{joint}$)**: Receive the DoG-encoded joint angles and learn to represent joint configurations. These correspond to your "saturating phase variables" but with bipolar saturation allowing both positive and negative values.

**Perceptual Units ($h_t^{percept}$)**: Process MuJoCo's sensory information including IMU data, contact forces, joint torques/velocities, center of mass dynamics, ground reaction forces, and body part positions.

**Place Units ($h_t^{place}$)**: Learn to associate joint configurations with perceptual contexts, encoding the robot's "pose state" - the combination of body configuration and physical situation. These emerge naturally through the training process.

**Planning Units ($h_t^{planning}$)**: Generate "fake" action sequences for internal simulation. These units can drive the joint and perceptual units internally, allowing the robot to simulate potential movements and their consequences without actually moving. This is the key to the intertwined planning/execution process.

**Torque Control Units ($h_t^{torque}$)**: Generate joint torque commands that are applied directly to MuJoCo's physics simulation. During real execution, these receive input from planning units. Their outputs are scaled to appropriate torque ranges: $\tau_t = W^{torque} \cdot h_t^{torque}$.

### Modified GRU Update Rules

We modify the standard GRU to handle these specialized unit types:

$$
\begin{align}
r_t &= \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \\
z_t &= \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \\
\tilde{h}_t &= \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h) \\
h_t &= (1-z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
\end{align}
$$

where $x_t$ contains both DoG joint encodings and perceptual inputs. The weight matrices respect functional groups - place units connect to joint and perceptual units, planning units can drive joint/perceptual units for simulation, and torque units receive from planning units. The control operates in two modes: real execution (joint angles → DoG encoding → GRU → torque commands → MuJoCo physics → new joint angles) and internal planning (planning units → simulated joint/perceptual dynamics).

### Training with Input Masking

Following REMI's approach, we apply masking at the input level. Random portions of $x_t$ are masked with probability $p_{mask} = 0.3$, zeroing out some DoG-encoded joint angles and perceptual inputs. After masking, inputs are normalized to maintain consistent activation: $x_t^{normalized} = x_t^{masked} \cdot \frac{\|x_t^{original}\|}{\|x_t^{masked}\|}$. The GRU must reconstruct the original unmasked inputs through its internal representations.

### Place Unit Emergence

Like in REMI, we expect place units will emerge naturally through the masked autoencoding process. These "body pose place" units develop by learning to reconstruct missing information - they automatically learn to associate joint configurations with perceptual contexts without explicit supervision. The place units simply participate in the same reconstruction loss as other units.

### Iterative Training Approach

We train the system using a curriculum learning approach with noise annealing over extremely short trajectories, gradually building up from pure exploration to goal-directed movement.

#### Phase 1: Pure Exploration with High Noise

- **Trajectory Length**: Very short (5-10 timesteps)
- **Torque Noise**: High noise added to all torque commands: $\tau_{actual} = \tau_{planned} + \mathcal{N}(0, \sigma_{high}^2)$ where $\sigma_{high}$ is large
- **Planning Goals**: Random goal joint configurations (DoG-encoded) fed to planning units
- **Planner Behavior**: Planning units generate random actions due to lack of training
- **Learning Objective**: Encoder learns to reconstruct masked inputs from chaotic but real movement data

During this phase, the encoder learns the basic dynamics of the humanoid body through random exploration. Place units begin to emerge by associating joint configurations with the resulting perceptual contexts (contact forces, balance, etc.).

#### Phase 2: Gradual Noise Reduction with Goal-Directed Training

- **Trajectory Length**: Gradually increase (10-50 timesteps)
- **Torque Noise**: Progressively decrease noise: $\sigma_t = \sigma_{high} \cdot e^{-\alpha t}$ where $t$ is training iteration
- **Planning Goals**: Provide diverse goal joint configurations to planning units
- **Training Dynamic**: Planning units learn to generate coherent action sequences toward goals, but noise ensures exploration continues

The encoder continues learning from mismatches between planned and actual outcomes. This robust training ensures the dynamics model can handle both planned actions and execution errors.

#### Phase 3: Refined Goal-Directed Movement

- **Trajectory Length**: Longer sequences (50-200 timesteps)
- **Torque Noise**: Minimal noise, mainly for robustness
- **Planning Goals**: Structured goal sequences that require multi-step planning
- **Capability**: System can reliably reach specified joint configurations

The planning units can now generate smooth action sequences toward arbitrary joint configuration goals, and the encoder provides accurate dynamics predictions.

### Training Loss

The training objective remains the reconstruction loss for masked inputs:

$$
L_{total} = \sum_{masked} \|h_t - x_{true,t}\|^2
$$

where the encoder must reconstruct missing sensory and joint information from the available context, learning robust dynamics representations throughout the noise annealing process.

## Intertwined Planning-Execution Pipeline with RL Bridge

After the encoder develops robust dynamics understanding through the iterative training, we implement the full intertwined planning-execution system with an RL bridge to guide toward walking behaviors.

### Planning-Execution Loop

The system operates through a continuous plan-act-replan cycle:

**Step 1: Goal Setting**

- RL component provides high-level walking goals (forward velocity targets, balance objectives)
- These are translated into desired joint configuration sequences or contextual targets for the planning units

**Step 2: Internal Planning**

- Encoder state initialized to current real state: $s_{encoder} = s_0$, pick some random goal state $s_g$
- Planning units generate a full action sequence: $[a_1, a_2, ..., a_T]$ that drives the encoder from $s_0$ toward goal $s_g$
- Internal simulation: encoder state evolves through planned dynamics: $s_0 \rightarrow s_1 \rightarrow ... \rightarrow s_T$
- Planning stops when distance to goal drops below threshold: $\|s_T - s_g\| < \epsilon$ or beyond a certain time budget $T$

**Step 3: Single-Step Execution**

- Take only the first planned action: $a_1$
- Execute $a_1$ in real MuJoCo environment with torque control
- Observe actual resulting state: $s_1^{real}$ (which may differ from planned $s_1$ due to noise, model errors, or disturbances)

**Step 4: State Reset and Update**

- Reset encoder state to actual current state: $s_{encoder} = s_1^{real}$
- Continue learning: encoder observes the mismatch between planned $s_1$ and actual $s_1^{real}$
- Update internal dynamics model through reconstruction loss

**Step 5: Repeat**

- Return to Step 2 with updated encoder state and potentially updated goals from RL

### RL Bridge for Walking

The RL component provides walking guidance without interfering with the core REMI dynamics:

**RL State**: High-level walking metrics (center of mass velocity, balance, energy efficiency)

**RL Actions**: Walking context signals or goal biases fed to planning units

- Forward velocity targets
- Balance maintenance signals
- Gait rhythm suggestions
- Terrain adaptation cues

**RL Rewards**:

- Forward progress: $r_{forward} = \Delta x_{com}$
- Balance maintenance: $r_{balance} = -|\text{angular deviation}|$
- Energy efficiency: $r_{energy} = -\|\tau\|^2$
- Smooth gait: $r_{smooth} = -\|\Delta\tau\|^2$

**RL Training**: Standard policy gradient (PPO/SAC) on the high-level walking objectives, while the REMI system handles all low-level motor control and planning.

# Coding Instructions
1. Only code necessary assembling functions/code. For the other function handlers, just write the function definition and the docstrings, so that we can first code up a backbone before we move to the next step.
2. Then, use hydra to define the config. Parameterize most of the parameters mentioend above.
3. To implement the network, we will use the torch implemented version. Notice that, during encoding, the inputs (joint angles converted to DoG encodings, perceptural inputs etc), are probed to the network, but at the moment there won't be any planning inputs. So the planning network will not receiving any input at the moment. Then, during planning, the encoder does not receive any input, but the planning network will receive the goal state as the input. Be careful about this implementation. One why is to make the entire projection layer of the GRU to be identity matrix, and pad the input with zeros for the regions that does not have input at certaining training phase. For this ew might need to modify the GRU implementation, potentially overwrite the weigths and also make the input layer to requires_grad = False, so that this layer will always be identity projection.
4. Use conda environment with name `mujoco_env`.