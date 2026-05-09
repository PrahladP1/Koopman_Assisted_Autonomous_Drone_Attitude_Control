# Koopman-Assisted Autonomous Drone Attitude Control

## Project Overview
This project investigates an operator-based learning approach to 
the discovery of linear representations of drone dynamics for predictive control. 
Specifically, a deep neural network (DNN) learns a linear physics model of drone dynamics
directly from position and attitude measurements and control history. This is done so by:

1) Linear lifting of nonlinear control system via autoencoder.
2) Multistep rollout supervision
3) Accounting for spectrum of eigenvalues to ensure learning of linear dynamic representations is state-dependent.
4) Hyperparameters tuning with Optuna
5) Design of loss functions to enforce linearity of learned dynamics, future state predictions, identifying intrinsic coordinates that help with reconstruction, one-step and multistep rollouts, and control consistency.

# Features
1) Deep Koopman modeling.
2) Encoder-decoder structure for lifting state dynamics to observable function space.
3) State-dependent Koopman operator approximation.
4) Multistep rollout training.
5) Splitting datasets trajectory-wise.

# System Overview
We learn a nonlinear mapping of drone dynamics given as $z = \phi(x)$ such that

$z_{k+1} = Kz_{k} + Bu_{k}$

Here, $z$ denotes the learned Koopman embedding, $x$ the physical drone state, and $u$ the control input.
The dynamics in Koopman space are trained using reconstruction loss, one-step prediction loss, multistep rollout loss, linearity loss, and physics-based regularization.

### State Representation & Training
The state representation of quadcopter dynamics is given as:

$x = [p_{x}\ p_{y}\ p_{z}\ v_{x}\ v_{y}\ v_{z}\ \phi\ \theta\ \psi\ \dot{\phi}\ \dot{\theta}\ \dot{\psi}]$

with control inputs as $u = [u_{thrust}\ u_{roll}\ u_{pitch}\ u_{yaw}]$. For system identification, training data was gathered
with sinusoidal, chirp, and PRBS excitations as control inputs, which facilitated the learning of transient
dynamics across various initial conditions (e.g., hover, aggressive maneuvers) and improve nonlinear state space coverage.


Training is executed in two stages. In Stage 1, we enforce the model to reconstruct the state $x$ by identifying
a few intrinsic coordinates where drone dynamics evolve linearly; conversely, we also learn the inverse of the intrinsic coordinates
to recover $x$. We also ensure consistency in learned dynamics by enforcing that dynamics in Koopman space
evolve coherently with physical trajectories. In Stage 2, we train the model to learn a consistent dynamics evolution 
over multistep prediction horizons. This is done by enforcing linearity and learning $K$ on intrinsic coordinates, which allows for future prediction
over $m$ time steps. By multistep rollouts, we minimize the accumuulation of prediction errors across multistep trajectories.

