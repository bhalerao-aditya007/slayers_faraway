import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class HopfFibrationGrid:
    """
    Constructs a uniform grid of SO(3) rotation matrices using
    Fibonacci lattice on S2 combined with discrete in-plane rotations.
    
    This serves as the anchor set for the discrete pose classification head.
    The model classifies which anchor is closest to the true rotation,
    then refines with a tangent-space offset.
    
    Why Hopf Fibration:
    - Standard Euler angle sampling clusters at poles (gimbal lock regions)
    - Random quaternion sampling is non-uniform on SO(3)
    - Fibonacci lattice gives mathematically uniform coverage of S2
    - Combining with uniform in-plane rotations gives uniform SO(3) coverage
    """

    def __init__(self, n_elevation: int = 32, n_inplane: int = 16):
        """
        Args:
            n_elevation: Number of direction samples on S2 sphere
                         (Fibonacci lattice points)
            n_inplane:   Number of discrete in-plane rotations per direction
                         Total anchors = n_elevation * n_inplane
        """
        self.n_elevation = n_elevation
        self.n_inplane = n_inplane
        self.total_anchors = n_elevation * n_inplane

        # Built on init
        self.anchor_rotations = None    # (K, 3, 3) rotation matrices
        self.anchor_quaternions = None  # (K, 4) quaternions [w, x, y, z]
        self.s2_directions = None       # (n_elevation, 3) unit vectors

        self._build()

    def _fibonacci_s2(self) -> np.ndarray:
        """
        Generate n_elevation uniformly distributed points on S2
        using the Fibonacci/golden ratio lattice.
        
        This avoids clustering at the poles that you get with
        naive (theta, phi) grid sampling.
        
        Returns:
            directions: (n_elevation, 3) unit vectors on S2
        """
        golden_ratio = (1 + np.sqrt(5)) / 2
        directions = []

        for i in range(self.n_elevation):
            # Uniform in z (avoids polar clustering)
            z = 1 - (2 * i + 1) / self.n_elevation
            
            # Radius at this z level
            r = np.sqrt(max(0, 1 - z * z))
            
            # Golden angle in azimuth
            phi = 2 * np.pi * i / golden_ratio
            
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            
            directions.append([x, y, z])

        return np.array(directions)  # (n_elevation, 3)

    def _direction_to_rotation(self, direction: np.ndarray) -> np.ndarray:
        """
        Convert a unit vector (direction on S2) to a rotation matrix
        where the z-axis of the rotated frame points along 'direction'.
        
        We construct an orthonormal frame {u, v, direction} and
        build R = [u | v | direction] as columns.
        
        Args:
            direction: (3,) unit vector
        Returns:
            R: (3, 3) rotation matrix
        """
        d = direction / np.linalg.norm(direction)

        # Find a vector not parallel to d for cross product
        # Use x-axis unless d is nearly parallel to it
        if abs(d[0]) < 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0])
        else:
            arbitrary = np.array([0.0, 1.0, 0.0])

        # Gram-Schmidt to get orthonormal frame
        u = np.cross(d, arbitrary)
        u = u / np.linalg.norm(u)
        v = np.cross(d, u)
        v = v / np.linalg.norm(v)

        # Build rotation matrix: columns are u, v, d
        R = np.column_stack([u, v, d])
        return R

    def _inplane_rotation(self, angle_rad: float) -> np.ndarray:
        """
        3x3 rotation matrix for in-plane rotation around z-axis.
        
        Args:
            angle_rad: rotation angle in radians
        Returns:
            R_z: (3, 3) rotation around z
        """
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

    def _build(self):
        """
        Build the full SO(3) anchor grid.
        
        For each of the n_elevation directions on S2:
            For each of the n_inplane rotation angles:
                R_anchor = R_direction @ R_inplane(angle)
        
        Total: n_elevation * n_inplane rotation matrices
        """
        self.s2_directions = self._fibonacci_s2()

        anchors_R = []
        anchors_q = []

        inplane_angles = np.linspace(0, 2 * np.pi, self.n_inplane, endpoint=False)

        for direction in self.s2_directions:
            R_dir = self._direction_to_rotation(direction)

            for angle in inplane_angles:
                R_inp = self._inplane_rotation(angle)
                R_anchor = R_dir @ R_inp

                # Verify it's a valid rotation matrix
                assert np.allclose(R_anchor @ R_anchor.T, np.eye(3), atol=1e-6), \
                    "Generated invalid rotation matrix"
                assert np.isclose(np.linalg.det(R_anchor), 1.0, atol=1e-6), \
                    "Rotation matrix has wrong determinant"

                anchors_R.append(R_anchor)

                # Convert to quaternion [w, x, y, z]
                rot = Rotation.from_matrix(R_anchor)
                q = rot.as_quat()  # scipy returns [x, y, z, w]
                q_wxyz = np.array([q[3], q[0], q[1], q[2]])  # reorder to [w,x,y,z]
                anchors_q.append(q_wxyz)

        self.anchor_rotations = np.array(anchors_R)     # (K, 3, 3)
        self.anchor_quaternions = np.array(anchors_q)   # (K, 4)

        print(f"Hopf Grid built: {self.total_anchors} anchors "
              f"({self.n_elevation} directions x {self.n_inplane} in-plane)")

    def geodesic_distance(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """
        Geodesic distance between two rotation matrices on SO(3).
        
        dist = arccos((trace(R1^T @ R2) - 1) / 2)
        
        This is the angle of the relative rotation R1^T @ R2,
        ranging from 0 (identical) to pi (opposite).
        
        Args:
            R1, R2: (3, 3) rotation matrices
        Returns:
            angle in radians
        """
        R_rel = R1.T @ R2
        # Clamp for numerical safety (trace can exceed 3 due to float errors)
        trace_val = np.clip((np.trace(R_rel) - 1) / 2, -1.0, 1.0)
        return np.arccos(trace_val)

    def find_nearest_anchor(self, R_query: np.ndarray) -> tuple:
        """
        Find the closest anchor to a query rotation matrix.
        
        This is used during training to find the ground-truth anchor
        for the classification loss.
        
        Args:
            R_query: (3, 3) rotation matrix
        Returns:
            anchor_idx: int, index of nearest anchor
            distance: float, geodesic distance in radians
            R_nearest: (3, 3) nearest anchor rotation matrix
        """
        distances = np.array([
            self.geodesic_distance(R_query, R_anchor)
            for R_anchor in self.anchor_rotations
        ])

        anchor_idx = np.argmin(distances)
        return anchor_idx, distances[anchor_idx], self.anchor_rotations[anchor_idx]

    def compute_tangent_offset(self, R_anchor: np.ndarray, R_target: np.ndarray) -> np.ndarray:
        """
        Compute the Lie algebra (tangent space) offset from anchor to target.
        
        R_target = R_anchor @ expm(skew(v))
        => skew(v) = logm(R_anchor^T @ R_target)
        => v = unskew(logm(R_anchor^T @ R_target))
        
        This 3-vector v is what the refinement head predicts.
        
        Args:
            R_anchor: (3, 3) anchor rotation matrix
            R_target: (3, 3) target rotation matrix
        Returns:
            v: (3,) Lie algebra offset vector
        """
        R_rel = R_anchor.T @ R_target
        rot = Rotation.from_matrix(R_rel)
        # rotvec IS the Lie algebra element (axis * angle)
        return rot.as_rotvec()  # (3,)

    def apply_tangent_offset(self, R_anchor: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Apply Lie algebra offset to anchor to get refined rotation.
        
        R_refined = R_anchor @ expm(skew(v))
        
        Used at inference time to reconstruct final pose from
        (predicted anchor index, predicted offset).
        
        Args:
            R_anchor: (3, 3) anchor rotation matrix
            v: (3,) Lie algebra offset vector
        Returns:
            R_refined: (3, 3) refined rotation matrix
        """
        R_offset = Rotation.from_rotvec(v).as_matrix()
        return R_anchor @ R_offset

    def compute_coverage_stats(self) -> dict:
        """
        Compute statistics about how well the grid covers SO(3).
        
        Key metric: max_gap = maximum geodesic distance from any random
        rotation to its nearest anchor. Smaller = better coverage.
        
        Returns:
            dict with coverage statistics
        """
        # Sample random rotations and find their nearest anchor distance
        n_samples = 5000
        random_rots = Rotation.random(n_samples).as_matrix()

        gaps = []
        for R in random_rots:
            _, dist, _ = self.find_nearest_anchor(R)
            gaps.append(dist)

        gaps = np.array(gaps)
        return {
            "total_anchors": self.total_anchors,
            "mean_gap_deg": np.degrees(gaps.mean()),
            "max_gap_deg": np.degrees(gaps.max()),
            "median_gap_deg": np.degrees(np.median(gaps)),
            "std_gap_deg": np.degrees(gaps.std()),
        }

    def visualize_s2_coverage(self, save_path: str = None):
        """
        Visualize the S2 direction distribution.
        Good coverage = points look evenly spread on sphere,
        no clusters at poles.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        d = self.s2_directions
        ax.scatter(d[:, 0], d[:, 1], d[:, 2],
                   c=d[:, 2],  # color by z (elevation)
                   cmap='RdYlGn', s=30, alpha=0.8)

        # Draw wireframe sphere for reference
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        sx = np.outer(np.cos(u), np.sin(v))
        sy = np.outer(np.sin(u), np.sin(v))
        sz = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(sx, sy, sz, alpha=0.1, color='gray')

        ax.set_title(f'S2 Fibonacci Lattice Coverage\n'
                     f'{self.n_elevation} directions (no polar clustering)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved to {save_path}")
        plt.show()

    def visualize_gap_distribution(self, save_path: str = None):
        """
        Histogram of geodesic distances from random rotations
        to their nearest anchor. Shows coverage quality.
        Tighter distribution = better coverage.
        """
        print("Computing coverage (sampling 5000 random rotations)...")
        n_samples = 5000
        random_rots = Rotation.random(n_samples).as_matrix()

        gaps = []
        for R in random_rots:
            _, dist, _ = self.find_nearest_anchor(R)
            gaps.append(np.degrees(dist))

        gaps = np.array(gaps)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(gaps, bins=60, color='steelblue', edgecolor='white', alpha=0.85)
        ax.axvline(gaps.mean(), color='red', linestyle='--',
                   label=f'Mean: {gaps.mean():.2f}°')
        ax.axvline(gaps.max(), color='orange', linestyle='--',
                   label=f'Max: {gaps.max():.2f}°')

        ax.set_xlabel('Geodesic Distance to Nearest Anchor (degrees)')
        ax.set_ylabel('Count')
        ax.set_title(f'SO(3) Coverage Gap Distribution\n'
                     f'{self.total_anchors} anchors '
                     f'({self.n_elevation} dirs x {self.n_inplane} in-plane)')
        ax.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()