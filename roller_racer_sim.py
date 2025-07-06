"""roller_racer_sim.py
Simulation & animation of a planar Roller Racer with two articulated platforms.

Formulation follows equations (2.6)–(2.7) in the referenced paper.  The state is

    s = [v1, psi, x, y]

where
    v1   – linear velocity of the first platform (m/s)
    psi  – absolute orientation of the first platform (rad)
    x, y – inertial position of the joint between platforms (m)

The control input φ(t) is prescribed: φ(t) = α sin(Ω t) + β.
The second platform orientation in the world frame is ψ + φ.

Missing physical constants are marked with TODO.  Fill them in before running.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
from tqdm import tqdm

###############################################################################
# Model parameters (fill in any TODOs)                                       #
###############################################################################

@dataclass
class Params:
    # Given in the screenshot (Fig. 5)
    J1: float = 20.0
    J2: float = 3.0
    M: float = 3.0
    c1: float = 2.0
    c2: float = 2.0
    # Additional constants appearing in A, B, C, D
    b1: float = 1.0
    b2: float = 1.0
    k1: float = 0.1
    k2: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    delta: float = 2.0
    epsilon: float = 1.0

    # Control signal coefficients φ(t) = α sin(Ω t) + β
    alpha: float = 0.5  # 1/3
    Omega: float = 1
    beta: float = 0
    phi_0: float = 0

    # Initial conditions
    v1_0: float = 0
    psi_0: float = 0.0
    x_0: float = 0.0
    y_0: float = 0.0

    # Simulation horizon & gfx
    T: float = 25.0  # seconds
    fps: int = 30

    # Geometry for drawing (metres)
    L1: float = c1*2  # length of platform 1 rectangle
    W1: float = 2
    L2: float = c2*2  # length of platform 2 rectangle
    W2: float = 2
    


# Global singleton (adjust as needed)
P = Params()

colormap = plt.cm.rainbow

###############################################################################
# Helper functions                                                            #
###############################################################################

def phi(t: float) -> float:
    """Prescribed articulation angle φ(t)."""
    return P.alpha * math.sin(P.Omega * t + P.phi_0) + P.beta

def phi_dot(t: float) -> float:
    """First derivative φ̇(t)."""
    return P.alpha * P.Omega * math.cos(P.Omega * t + P.phi_0)

def phi_ddot(t: float) -> float:
    """Second derivative φ̈(t)."""
    return -P.alpha * (P.Omega ** 2) * math.sin(P.Omega * t + P.phi_0)


def _s1_s2(phi_val: float) -> tuple[float, float]:
    """Compute S₁ and S₂ helper terms."""
    s1 = P.c1 * math.cos(phi_val) + P.c2
    s2 = P.c1 + P.c2 * math.cos(phi_val)
    return s1, s2


###############################################################################
# Coefficient functions A(t), B(t), C(t), D(t)                                #
###############################################################################

def _denominator(phi_val: float, s1: float) -> float:
    return P.J1 * math.sin(phi_val) ** 2 + P.M * s1 ** 2


def A(t: float) -> float:
    φ = phi(t)
    φ̇ = phi_dot(t)
    s1, s2 = _s1_s2(φ)
    denom = _denominator(φ, s1)
    num = -φ̇ * math.sin(φ) * (P.J1 * s2 + P.delta * s1)
    return num / (s1 * denom)


def B(t: float) -> float:
    """Equation (2.6) numerator. Requires missing parameters to be filled."""
    if None in (P.b1, P.b2, P.k1, P.k2, P.a1, P.a2):
        raise ValueError("Please provide b1, b2, k1, k2, a1, a2 before running.")

    φ = phi(t)
    φ̇ = phi_dot(t)
    φ̈ = phi_ddot(t)
    s1, _ = _s1_s2(φ)
    denom = _denominator(φ, s1)

    term1 = φ̈ * math.sin(φ) * s1 * (P.J1 * P.c2 - P.J2 * s1)
    term2 = (φ̇ ** 2) * (
        P.J1 * P.c1 * P.c2 * (math.sin(φ) ** 2) -
        s1 * (P.c1 * P.delta * math.cos(φ) + P.epsilon * P.c2 * s1)
    )
    return (term1 + term2) / (s1 * denom)


def C(t: float) -> float:
    if None in (P.b1, P.b2, P.k1, P.k2):
        raise ValueError("Please provide b1, b2, k1, k2 before running.")
    φ = phi(t)
    s1, s2 = _s1_s2(φ)
    denom = _denominator(φ, s1)
    num = 2 * (P.b1 ** 2 * P.k1 + P.k2 * P.b2 ** 2) * math.sin(φ) ** 2 + P.k1 * s1 ** 2 + P.k2 * s2 ** 2
    return num / denom


def D(t: float) -> float:
    if None in (P.b1, P.b2, P.k1, P.k2):
        raise ValueError("Please provide b1, b2, k1, k2 before running.")
    φ = phi(t)
    φ̇ = phi_dot(t)
    s1, _ = _s1_s2(φ)
    denom = _denominator(φ, s1)
    num = 2 * φ̇ * math.sin(φ) * (P.c1 * P.k2 * (P.b2 ** 2 - P.c2 ** 2) * math.cos(φ) - P.c2 * (P.b1 ** 2 * P.k1 + P.k2 * P.c2 ** 2))
    return num / denom


###############################################################################
# Right-hand side of the ODE                                                  #
###############################################################################

def rhs(t: float, s: NDArray[np.float64]) -> NDArray[np.float64]:
    v1, psi, x, y = s

    φ = phi(t)
    φ̇ = phi_dot(t)

    s1, _ = _s1_s2(φ)

    v1_dot = (A(t) - C(t)) * v1 - B(t) + D(t)

    psi_dot = -(v1 * math.sin(φ) + P.c2 * φ̇) / (P.c1 * math.cos(φ) + P.c2)

    # Equation (2.7):
    #   ẋ = v₁ cosψ − c₁ (v₁ sinφ + c₂ φ̇)/(c₁ cosφ + c₂) sinψ
    #       = v₁ cosψ + c₁ ψ̇ sinψ  (because ψ̇ = −(v₁ sinφ + c₂ φ̇)/(c₁ cosφ + c₂))
    #   ẏ = v₁ sinψ + c₁ (v₁ sinφ + c₂ φ̇)/(c₁ cosφ + c₂) cosψ
    #       = v₁ sinψ − c₁ ψ̇ cosψ
    x_dot = v1 * math.cos(psi) + P.c1 * psi_dot * math.sin(psi)
    y_dot = v1 * math.sin(psi) - P.c1 * psi_dot * math.cos(psi)

    return np.array([v1_dot, psi_dot, x_dot, y_dot])


###############################################################################
# Simulation + animation                                                      #
###############################################################################

def simulate() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the ODE and return t, v1, psi, x, y, phi arrays."""
    t_eval = np.linspace(0.0, P.T, int(P.T * P.fps) + 1)
    y0 = np.zeros(4)  # v1=0, psi=0, x=0, y=0 per user preference
    
    y0[0] = P.v1_0
    y0[1] = P.psi_0
    y0[2] = P.x_0
    y0[3] = P.y_0

    sol = solve_ivp(rhs, (0.0, P.T), y0, t_eval=t_eval, vectorized=False, rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(sol.message)
    
    # Compute phi values for all time points
    phi_values = np.array([phi(t) for t in sol.t])
    
    return sol.t, sol.y[0], sol.y[1], sol.y[2], sol.y[3], phi_values


def _rectangle_vertices(L: float, W: float) -> NDArray[np.float64]:
    """Rectangle centered at origin, axis-aligned along +x, returned as (4,2)."""
    return np.array([[-L / 2, -W / 2], [L / 2, -W / 2], [L / 2, W / 2], [-L / 2, W / 2]])


def animate(t: np.ndarray, x: np.ndarray, y: np.ndarray, psi: np.ndarray, phi_vals: np.ndarray) -> None:
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Roller Racer trajectory')

    # Trace line
    (trace,) = ax.plot([], [], 'k-', lw=0.5, alpha=0.5)

    # Create platform polygons
    verts1 = _rectangle_vertices(P.L1, P.W1)
    verts2 = _rectangle_vertices(P.L2, P.W2)
    poly1 = Polygon(verts1, fc='skyblue', ec='k', alpha=0.1)
    poly2 = Polygon(verts2, fc='salmon', ec='k', alpha=0.1)
    ax.add_patch(poly1)
    ax.add_patch(poly2)

    # View limits (auto-scale initially)
    margin = max(P.L1, P.L2) * 1.5
    ax.set_xlim(x.min() - margin, x.max() + margin)
    ax.set_ylim(y.min() - margin, y.max() + margin)

    def update(frame: int):
        # Update trace
        trace.set_data(x[:frame + 1], y[:frame + 1])

        # Platform 1 transformation
        transform1 = Affine2D().rotate(psi[frame]).translate(x[frame], y[frame]) + ax.transData
        poly1.set_transform(transform1)

        # Platform 2 orientation = psi + phi(t) - use pre-computed phi values
        φ = phi_vals[frame]
        offsetx = -P.c2 * math.cos(psi[frame] + φ) - P.c1 * math.cos(psi[frame])
        offsety = -P.c2 * math.sin(psi[frame] + φ) - P.c1 * math.sin(psi[frame])
        transform2 = Affine2D().rotate(psi[frame] + φ).translate(x[frame] + offsetx, y[frame] + offsety) + ax.transData
        poly2.set_transform(transform2)
        return trace, poly1, poly2

    ani = FuncAnimation(fig, update, frames=len(t), interval=0, blit=True)
    plt.show()


def animate_multiple(num_robots: int = 100) -> None:
    """Animate multiple roller racers with different alpha values."""
    # Set dark style
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]', fontsize=12, color='white')
    ax.set_ylabel('y [m]', fontsize=12, color='white')
    
    # Generate different alpha values
    alpha_values = np.linspace(20,1000, num_robots)
    
    # Colors for different robots - invert colormap so red is slowest, blue is fastest
    # colors = colormap(np.linspace(1, 0, num_robots))  # Inverted: 1 to 0 instead of 0 to 1
    colors = ['yellow']
    
    # Store simulation data for all robots
    all_trajectories = []
    all_orientations = []
    all_times = []
    all_phi_values = []
    
    # Simulate all robots
    for i, alpha in tqdm(enumerate(alpha_values)):
        # Create temporary params with different alpha
        temp_params = Params()
        # temp_params.J1 = alpha
        
        # Temporarily replace global params
        global P
        old_P = P
        P = temp_params
        
        try:
            t, v1, psi, x, y, phi = simulate()
            all_trajectories.append((x, y))
            all_orientations.append(psi)
            all_times.append(t)
            all_phi_values.append(phi)
        finally:
            P = old_P
    
    # Set up plot limits
    all_x = np.concatenate([traj[0] for traj in all_trajectories])
    all_y = np.concatenate([traj[1] for traj in all_trajectories])
    margin = max(P.L1, P.L2) * 2
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    # Set dark background
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Create trace lines for all robots
    trace_lines = []
    for i in range(num_robots):
        line, = ax.plot([], [], '-', color=colors[i], alpha=0.7, linewidth=0.5)
        trace_lines.append(line)
    
    # Create robot polygons (only show a subset to avoid clutter)
    robot_polygons = []
    # show_robots = min(1, num_robots)  # Show at most 20 actual robot shapes
    show_robots = num_robots
    robot_indices = np.linspace(0, num_robots-1, show_robots, dtype=int)
    
    for i in robot_indices:
        # Platform 1
        verts1 = _rectangle_vertices(P.L1, P.W1)
        poly1 = Polygon(verts1, fc=colors[i], ec='gray', alpha=0.5, linewidth=1)
        ax.add_patch(poly1)
        
        # Platform 2
        verts2 = _rectangle_vertices(P.L2, P.W2)
        poly2 = Polygon(verts2, fc=colors[i], ec='gray', alpha=0.3, linewidth=1)
        ax.add_patch(poly2)
        
        robot_polygons.append((poly1, poly2))
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=alpha_values.min(), vmax=alpha_values.max()))
    sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    # cbar.set_label('Delta', fontsize=12, color='white')
    # cbar.ax.tick_params(colors='white')
    
    # Initialize title
    title = ax.set_title(f'Roller Racer Swarm {num_robots} robot', fontsize=16, color='white', pad=20)
    
    def update(frame: int):
        artists = []
        
        # Update title with current time
        current_time = all_times[0][frame] if frame < len(all_times[0]) else all_times[0][-1]
        # title.set_text(f'Roller Racer Swarm ({num_robots} robots) - t = {current_time:.2f} s')
        # artists.append(title)
        
        # Update all trace lines
        for i, line in enumerate(trace_lines):
            x, y = all_trajectories[i]
            line.set_data(x[:frame + 1], y[:frame + 1])
            # line.set_data(x[-1-frame:-1], y[-1-frame:-1])
            artists.append(line)
        
        # Update robot shapes (subset only)
        for idx, (poly1, poly2) in enumerate(robot_polygons):
            robot_idx = robot_indices[idx]
            x, y = all_trajectories[robot_idx]
            psi = all_orientations[robot_idx]
            phi_vals = all_phi_values[robot_idx]
            
            if frame < len(x):
                # Platform 1 transformation
                transform1 = Affine2D().rotate(psi[frame]).translate(x[frame], y[frame]) + ax.transData
                poly1.set_transform(transform1)
                
                # Platform 2 transformation - use pre-computed phi values
                φ = phi_vals[frame]
                offsetx = -P.c2 * math.cos(psi[frame] + φ) - P.c1 * math.cos(psi[frame])
                offsety = -P.c2 * math.sin(psi[frame] + φ) - P.c1 * math.sin(psi[frame])
                transform2 = Affine2D().rotate(psi[frame] + φ).translate(x[frame] + offsetx, y[frame] + offsety) + ax.transData
                poly2.set_transform(transform2)
                
                artists.extend([poly1, poly2])
        
        return artists
    
    # Find the maximum number of frames across all simulations
    max_frames = max(len(traj[0]) for traj in all_trajectories)
    
    ani = FuncAnimation(fig, update, frames=max_frames, interval=5 / P.fps, blit=True, repeat=True)
    
    # Adjust layout to accommodate colorbar
    plt.tight_layout()
    plt.show()


###############################################################################
# Main entry point                                                            #
###############################################################################

def main():
    # Single robot animation
    # t, v1, psi, x, y, phi_vals = simulate()
    # animate(t, x, y, psi, phi_vals)
    
    # Multi-robot animation
    animate_multiple(1)


if __name__ == "__main__":
    main() 