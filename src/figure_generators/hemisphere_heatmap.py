import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from hemisphere_with_robot import add_robot_to_hemisphere_3d
from PIL import Image, ImageChops
import os
import tempfile
from matplotlib.ticker import MaxNLocator

class HemisphereHeatmap:
    def __init__(self, n_cameras=100, radius=1.2):
        """
        Initialize the hemisphere heatmap visualizer.

        Args:
            n_cameras: Number of camera positions
            radius: Radius of the hemisphere (default 1.2 meters - physical size)
        """
        self.n_cameras = n_cameras
        self.radius = radius
        self.camera_positions = None
        self.mse_values=None
        # self.accuracy_values = None  # REMOVED - using MSE directly

    def generate_camera_positions(self, distribution='uniform'):
        """
        Generate camera positions on a hemisphere.

        Args:
            distribution: 'uniform', 'random', or 'clustered'
        """
        if distribution == 'uniform':
            # Generate uniform distribution using Fibonacci spiral
            indices = np.arange(0, self.n_cameras, dtype=float) + 0.5
            theta = np.arccos(1 - indices/self.n_cameras)  # Polar angle (0 to pi/2 for hemisphere)
            phi = np.pi * (1 + 5**0.5) * indices  # Azimuthal angle (golden angle)

        elif distribution == 'random':
            # Random distribution
            phi = np.random.uniform(0, 2*np.pi, self.n_cameras)
            u = np.random.uniform(0, 1, self.n_cameras)
            theta = np.arccos(1 - u)  # For hemisphere

        else:  # clustered
            # Create clusters of cameras
            n_clusters = 5
            cluster_centers = np.random.uniform(0, 2*np.pi, n_clusters)
            phi = []
            theta = []
            for center in cluster_centers:
                n_in_cluster = self.n_cameras // n_clusters
                phi.extend(np.random.normal(center, 0.3, n_in_cluster))
                theta.extend(np.random.uniform(0, np.pi/2, n_in_cluster))
            phi = np.array(phi[:self.n_cameras])
            theta = np.array(theta[:self.n_cameras])

        # Convert spherical to Cartesian coordinates
        x = self.radius * np.sin(theta) * np.cos(phi)
        y = self.radius * np.sin(theta) * np.sin(phi)
        z = self.radius * np.cos(theta)

        self.camera_positions = np.column_stack([x, y, z])
        return self.camera_positions

    # REMOVED: generate_accuracy_values method - no longer needed for MSE-based visualization
    # Use real MSE values from your data instead
    # def generate_accuracy_values(self, pattern='gradient'):
    #     """
    #     Generate synthetic accuracy values for each camera position.
    #
    #     Args:
    #         pattern: 'gradient', 'hotspots', 'radial', or 'sinusoidal'
    #     """
    #     if self.camera_positions is None:
    #         self.generate_camera_positions()
    #
    #     x, y, z = self.camera_positions.T
    #
    #     if pattern == 'gradient':
    #         # Linear gradient from one side to another
    #         self.accuracy_values = 0.5 + 0.5 * (x + y) / (2 * self.radius)
    #
    #     elif pattern == 'hotspots':
    #         # Multiple hotspots of high accuracy
    #         hotspots = np.array([[0.5, 0.5, 0.7], [-0.5, 0.5, 0.7], [0, -0.7, 0.5]])
    #         accuracy = np.zeros(self.n_cameras)
    #         for hotspot in hotspots:
    #             dist = np.sqrt(np.sum((self.camera_positions - hotspot)**2, axis=1))
    #             accuracy += np.exp(-2 * dist**2)
    #         self.accuracy_values = accuracy / np.max(accuracy)
    #
    #     elif pattern == 'radial':
    #         # Radial pattern from center
    #         r = np.sqrt(x**2 + y**2)
    #         self.accuracy_values = np.exp(-2 * r**2 / self.radius**2)
    #
    #     else:  # sinusoidal
    #         # Sinusoidal pattern
    #         self.accuracy_values = 0.5 + 0.5 * np.sin(3*x) * np.cos(3*y)
    #
    #     # Add some noise for realism
    #     self.accuracy_values += np.random.normal(0, 0.05, self.n_cameras)
    #     self.accuracy_values = np.clip(self.mse_values, 0, 1)
    #
    #     return self.accuracy_values


    def create_hemisphere_mesh(self, resolution=50):
        """
        Create a hemisphere mesh for visualization.

        Args:
            resolution: Grid resolution for the mesh
        """
        theta = np.linspace(0, np.pi/2, resolution)
        phi = np.linspace(0, 2*np.pi, resolution)
        THETA, PHI = np.meshgrid(theta, phi)

        X = self.radius * np.sin(THETA) * np.cos(PHI)
        Y = self.radius * np.sin(THETA) * np.sin(PHI)
        Z = self.radius * np.cos(THETA)

        return X, Y, Z, THETA, PHI

    def interpolate_mse(self, X, Y, Z):
        """
        Interpolate MSE values across the hemisphere surface.

        Args:
            X, Y, Z: Mesh grid coordinates
        """
        # Flatten mesh points
        points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])

        # Ensure mse_values is a 1D array and reshape for RBFInterpolator
        mse_vals = np.asarray(self.mse_values).flatten()

        # RBFInterpolator expects (n_points, n_values) shape
        # For 1D output, reshape to (n_points, 1)
        mse_vals_2d = mse_vals.reshape(-1, 1)

        # Use RBF interpolation for smooth results
        rbf = RBFInterpolator(self.camera_positions, mse_vals_2d,
                             smoothing=0.1, kernel='gaussian', epsilon=2.0)
        interpolated = rbf(points)

        # Reshape to match mesh and flatten back to 1D per point
        return interpolated.flatten().reshape(X.shape)

    def compute_gradient(self, mse_mesh, X, Y, Z):
        """
        Compute the gradient of MSE across the hemisphere.

        The gradient naturally points in the direction of INCREASING MSE (worse).
        We NEGATE it so arrows point toward DECREASING MSE (better performance).

        Args:
            mse_mesh: Interpolated MSE values
            X, Y, Z: Mesh coordinates
        """
        # Compute gradients in mesh space
        # dy, dx because numpy.gradient returns row-gradient first
        dy, dx = np.gradient(mse_mesh)

        # CRITICAL: Negate gradient to point toward improvement (lower MSE)
        # This makes arrows point toward better (lower MSE) areas
        dx = -dx
        dy = -dy

        # Scale gradients appropriately
        scale = 0.1
        dx *= scale
        dy *= scale

        # Project gradients to 3D surface
        # This is a simplified projection - for accurate surface gradients,
        # we would need to compute tangent vectors
        return dx, dy

    def plot_hemisphere_heatmap(self, show_cameras=True, show_gradient=True,
                                colormap='hot', save_path=None):
        """
        Create the complete visualization with heatmap and optional gradient arrows.

        Args:
            show_cameras: Whether to show camera positions
            show_gradient: Whether to show gradient arrows
            colormap: Matplotlib colormap name
            save_path: Path to save the figure
        """
        # Create mesh
        X, Y, Z, THETA, PHI = self.create_hemisphere_mesh(resolution=50)

        # Interpolate accuracy
        mse_mesh = self.interpolate_mse(X, Y, Z)

        # Create figure with subplots
        fig = plt.figure(figsize=(20/2, 8/2))
        # fig.patch.set_facecolor("#101014")


        # 3D Hemisphere view
        ax1 = fig.add_subplot(131, projection='3d')        #


        ## Remove 3D box/grid for ax1 but keep the hemisphere mesh
        ax1.grid(False)
        for axis in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
            axis._axinfo["grid"]["linewidth"] = 0  # hide lattice lines




        # Apply Gaussian filter for smoother visualization
        mse_smooth = gaussian_filter(mse_mesh, sigma=1)

        # Plot hemisphere with heatmap
        # --- Plot hemisphere with heatmap ---
        surf = ax1.plot_surface(
            X, Y, Z,
            facecolors=cm.get_cmap(colormap + 'hot_r')(mse_smooth),
            rstride=1, cstride=1,
            linewidth=0, antialiased=False,
            shade=False,
            alpha=0.6   # make surface semi-transparent
        )

        # --- Add camera positions more visibly ---
        if show_cameras and self.camera_positions is not None:
            cam = self.camera_positions.copy()
            mse = getattr(self, "mse_values", np.ones(len(cam)))

            # normalize+offset EVEN IF caller forgot
            cam = cam / np.linalg.norm(cam, axis=1, keepdims=True) * (self.radius * 1)

            ax1.scatter(
                cam[:, 0], cam[:, 1], cam[:, 2],
                s=10,                      # bigger
                c=mse, cmap=colormap + 'hot_r',
                edgecolors='black',
                linewidths=0.9,
                depthshade=False,          # don't let 3d shading dim them
                alpha=1.0,
                zorder=10                  # draw on top
            )
        # if show_cameras and self.camera_positions is not None:
        #     cam = self.camera_positions.copy()
        #     mse = getattr(self, "mse_values", np.ones(len(cam)))

        #     # Slightly push cameras outward so they sit above the surface
        #     cam = cam / np.linalg.norm(cam, axis=1, keepdims=True) * (self.radius * 1.02)

        #     ax1.scatter(
        #         cam[:, 0], cam[:, 1], cam[:, 2],
        #         s=70,                        # bigger points
        #         c=mse, cmap=colormap + '_r',
        #         edgecolors='black',          # dark outline for visibility
        #         linewidths=0.8,
        #         depthshade=False,            # ensures bright color
        #         alpha=1.0,
        #         zorder=5
        #     )
        # Add gradient arrows
        if show_gradient:
            dx, dy = self.compute_gradient(mse_smooth, X, Y, Z)

            # Sample points for arrows (every nth point)
            step = 5
            X_sample = X[::step, ::step]
            Y_sample = Y[::step, ::step]
            Z_sample = Z[::step, ::step]
            dx_sample = dx[::step, ::step]
            dy_sample = dy[::step, ::step]

            # Create tangent vectors for arrows
            for i in range(X_sample.shape[0]):
                for j in range(X_sample.shape[1]):
                    if Z_sample[i, j] > 0.05:  # Only show arrows on visible part
                        # Compute approximate tangent vectors
                        start_point = [X_sample[i, j], Y_sample[i, j], Z_sample[i, j]]

                        # Gradient direction in 3D (simplified)
                        grad_x = dx_sample[i, j] * np.cos(PHI[i*step, j*step])
                        grad_y = dy_sample[i, j] * np.sin(PHI[i*step, j*step])
                        grad_z = 0

                        ax1.quiver(start_point[0], start_point[1], start_point[2],
                                 grad_x, grad_y, grad_z,
                                 color='black', alpha=0.5, arrow_length_ratio=0.3,
                                 length=0.05)

        ax1.set_xlabel('X (m)', fontsize=9)
        ax1.set_ylabel('Y (m)', fontsize=9)
        ax1.set_zlabel('Z (m)', fontsize=9)
        # Title removed
        ax1.set_box_aspect([1,1,0.5])
        ax1.view_init(elev=30, azim=45)
        ax1.tick_params(labelsize=8)

        # Top-down view (circular heatmap)
        ax2 = fig.add_subplot(132)

        # Project to 2D circle
        r = np.sqrt(X**2 + Y**2)
        mask = r <= self.radius

        ax2 = fig.add_subplot(132)
        # ax2.set_facecolor("#101014")

        # Create circular heatmap
        contourf = ax2.contourf(X, Y, mse_smooth, levels=20, cmap=colormap)
        ax2.contour(X, Y, mse_smooth, levels=10, colors='black',
                   linewidths=0.5, alpha=0.3)

        if show_cameras:
            ax2.scatter(self.camera_positions[:, 0],
                       self.camera_positions[:, 1],
                       c=self.mse_values, cmap=colormap,
                       s=30, edgecolors='black', linewidths=0.5)

        # Add gradient arrows in 2D
        if show_gradient:
            ax2.quiver(X[::step, ::step], Y[::step, ::step],
                      dx[::step, ::step], dy[::step, ::step],
                      alpha=0.5, width=0.003)

        ax2.set_xlabel('X (m)', fontsize=9)
        ax2.set_ylabel('Y (m)', fontsize=9)
        # Title removed
        ax2.set_aspect('equal')
        ax2.tick_params(labelsize=8)
        # ax2.set_xlim([-self.radius*1.1, self.radius*1.1])
        # ax2.set_ylim([-self.radius*1.1, self.radius*1.1])
        ax2.set_xlim([-self.radius*1, self.radius*1])
        ax2.set_ylim([-self.radius*1, self.radius*1])

        # Add circle boundary
        # circle = plt.Circle((0, 0), self.radius, fill=False, edgecolor='black', linewidth=2)
        # ax2.add_patch(circle)
        # thick faint ring
        # ax2.add_patch(plt.Circle((0,0), 1.0, fill=False, edgecolor="white", linewidth=2.0, alpha=0.25, zorder=1))
        # # sharp dark ring on top
        # ax2.add_patch(plt.Circle((0,0), 1.0, fill=False, edgecolor="black", linewidth=0.8, zorder=30))


        # Side view with gradient magnitude

        ax3 = fig.add_subplot(133, projection='3d')

        # Remove 3D box/grid for ax3 but keep the hemisphere mesh
        ax3.grid(False)
        for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            axis._axinfo["grid"]["linewidth"] = 0


        # Compute gradient magnitude
        grad_magnitude = np.sqrt(dx**2 + dy**2)

        # --- Plot hemisphere surface (slightly transparent) ---
        surf2 = ax3.plot_surface(
            X, Y, Z,
            facecolors=cm.get_cmap('hot')(grad_magnitude),
            rstride=1, cstride=1,
            linewidth=0, antialiased=False,
            shade=False,
            alpha=0.6
        )

            # normalize+offset


        # --- Add camera positions ---
        if show_cameras and self.camera_positions is not None:
            cam = self.camera_positions.copy()
            mse = getattr(self, "mse_values", np.ones(len(cam)))

            # Push cameras outward slightly from the hemisphere
            cam = cam / np.linalg.norm(cam, axis=1, keepdims=True) * (self.radius * 1)

            # ax3.scatter(
            #     cam[:, 0], cam[:, 1], cam[:, 2],
            #     s=70,
            #     c=acc, cmap='turbo',         # you can change to any color map (e.g., 'hot' or 'inferno')
            #     edgecolors='black',
            #     linewidths=0.8,
            #     depthshade=False,
            #     alpha=1.0,
            #     zorder=5
            # )
            # ax3.scatter(
            #     cam[:, 0], cam[:, 1], cam[:, 2],
            #     s=10,                      # bigger
            #     c=acc, cmap='hot',
            #     edgecolors='black',
            #     linewidths=0.9,
            #     depthshade=False,          # don't let 3d shading dim them
            #     alpha=1.0,
            #     zorder=10                  # draw on top
            # )
        ax3.set_xlabel('X (m)', fontsize=9)
        ax3.set_ylabel('Y (m)', fontsize=9)
        ax3.set_zlabel('Z (m)', fontsize=9)
        # Title removed
        ax3.set_box_aspect([1,1,0.5])
        ax3.view_init(elev=30, azim=45)
        ax3.tick_params(labelsize=8)

        # ax3 = fig.add_subplot(133, projection='3d')

        # # Compute gradient magnitude
        # grad_magnitude = np.sqrt(dx**2 + dy**2)

        # # Plot with gradient magnitude as color
        # surf2 = ax3.plot_surface(X, Y, Z, facecolors=cm.get_cmap('viridis')(grad_magnitude),
        #                         alpha=0.9, shade=True, antialiased=True)

        # ax3.set_xlabel('X')
        # ax3.set_ylabel('Y')
        # ax3.set_zlabel('Z')
        # # ax3.set_title('Gradient Magnitude Visualization')  # Title removed
        # ax3.set_box_aspect([1,1,0.5])
        # ax3.view_init(elev=30, azim=135)

        # Add colorbars
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.05, right=0.95, wspace=0.3)

        # Accuracy colorbar
        cbar_ax1 = fig.add_axes([0.12, 0.05, 0.25, 0.02])
        cbar1 = plt.colorbar(contourf, cax=cbar_ax1, orientation='horizontal')
        cbar1.set_label('Accuracy', fontsize=10)

        # Gradient magnitude colorbar
        cbar_ax2 = fig.add_axes([0.68, 0.05, 0.25, 0.02])
        norm = Normalize(vmin=grad_magnitude.min(), vmax=grad_magnitude.max())
        sm = cm.ScalarMappable(norm=norm, cmap='hot')
        cbar2 = plt.colorbar(sm, cax=cbar_ax2, orientation='horizontal')
        cbar2.set_label('Gradient Magnitude', fontsize=10)

        plt.suptitle('Camera Position Accuracy Analysis on Hemisphere', fontsize=16, y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # plt.show()

        return fig


    def plot_paper_topdown(
        self,
        save_path,
        fig_width_in=3.25,
        dpi=300,
        cmap="hot",
        label=None,     # e.g. "height"
        vmin=None,      # Manual min for colorbar
        vmax=None,      # Manual max for colorbar
        use_power_norm=True,  # Use PowerNorm for small ranges
    ):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import Normalize, PowerNorm

        if self.camera_positions is None or self.mse_values is None:
            raise ValueError("Need camera_positions + mse_values")

        X, Y, Z, THETA, PHI = self.create_hemisphere_mesh(resolution=90)
        mse_mesh = self.interpolate_mse(X, Y, Z)

        dx_raw, dy_raw = self.compute_gradient(mse_mesh, X, Y, Z)
        grad_mag = np.sqrt(dx_raw**2 + dy_raw**2)
        max_mag = np.max(grad_mag)
        if max_mag < 1e-12:
            ux = np.zeros_like(dx_raw)
            uy = np.zeros_like(dy_raw)
        else:
            ux = dx_raw / (max_mag + 1e-12)
            uy = dy_raw / (max_mag + 1e-12)

        fig, ax = plt.subplots(figsize=(fig_width_in/2, fig_width_in/2), dpi=dpi)

        # Determine colorbar limits
        if vmin is None:
            vmin = mse_mesh.min()
        if vmax is None:
            vmax = mse_mesh.max()

        # Use PowerNorm for better contrast when MSE range is small
        mse_range = vmax - vmin
        if use_power_norm and mse_range < 0.10:  # Small range threshold
            # PowerNorm with gamma < 1 stretches small differences
            norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
            print(f"  Using PowerNorm (gamma=0.5) for small MSE range: {mse_range:.4f}")
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        cf = ax.contourf(X, Y, mse_mesh, levels=30, cmap=cmap , norm=norm)


        # normalize magnitudes to [0, 1]
        grad_norm = grad_mag / max_mag

        # choose a maximum arrow length in *hemisphere units*
        L_max = 0.5 * self.radius  # tweak this

        # final components used for quiver
        dx = ux * grad_norm * L_max
        dy = uy * grad_norm * L_max

        # ax.scatter(
        #     # self.camera_positions[:, 0],
        #     # self.camera_positions[:, 1],
        #     c=self.mse_values,
        #     cmap=cmap,
        #     s=18,
        #     # edgecolors="black",
        #     linewidths=0.4,
        #     zorder=10,
        # )

        # compute gradient of accuracy in x and y
        # dx, dy = np.gradient(acc_mesh, X[0, :], Y[:, 0])

        # # gradient magnitude
        # grad_mag = np.sqrt(dx**2 + dy**2)
        # eps = 1e-8

        # # unit direction vectors
        # ux = dx / (grad_mag + eps)
        # uy = dy / (grad_mag + eps)

        # # scale magnitudes into a nice length range (0 .. L_max)
        # grad_norm = grad_mag / (grad_mag.max() + eps)
        # L_max = 0.15 * self.radius         # tweak to make arrows longer/shorter
        # U = ux * grad_norm * L_max         # x-component of arrow
        # V = uy * grad_norm * L_max         # y-component of arrow

        step = 8  # subsample so it’s not too busy
        ax.quiver(
            X[::step, ::step],
            Y[::step, ::step],
            dx[::step, ::step],
            dy[::step, ::step],
            color="tab:blue",          # blue arrows
            angles="xy",
            scale_units="xy",
            scale=1.0,                 # use U, V directly in data units
            width=0.004,
            headwidth=5,
            headlength=6,
            alpha=0.9,
            zorder=50,
        )

        # step = 8
        # ax.quiver(
        #     X[::step, ::step],
        #     Y[::step, ::step],
        #     dx[::step, ::step],
        #     dy[::step, ::step],
        #     color="blue",
        #     width=0.005,
        #     headwidth=6.5,
        #     headlength=7.0,
        #     scale=16,
        #     alpha=0.8,
        #     zorder=50,
        # )


        circ = plt.Circle((0, 0), self.radius, fill=False, edgecolor="black", linewidth=1.1, zorder=25)
        ax.add_patch(circ)

        ax.set_aspect("equal")
        ax.set_xlim(-self.radius*1.05, self.radius*1.05)
        ax.set_ylim(-self.radius*1.05, self.radius*1.05)
        ax.set_xlabel("X (m)", fontsize=7)
        ax.set_ylabel("Y (m)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_axis_off()


        # vmin, vmax = contourf.get_clim()
        # ticks = np.linspace(vmin, vmax, 6)
        # cbar = plt.colorbar(contourf, ax=ax, ticks=ticks, shrink=0.8, orientation='horizontal')

        cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03, orientation='horizontal')
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("MSE", fontsize=7)

        # Caption at bottom - REMOVED to keep images clean
        # if label is not None:
        #     fig.text(0.5, 0.01, label, ha="center", va="bottom", fontsize=7)

        fig.tight_layout(pad=0.4)
        fig.savefig(save_path, dpi=dpi)
        if save_path.lower().endswith(".png"):
            fig.savefig(save_path.replace(".png", ".pdf"))
        plt.close(fig)

    # def plot_paper_3d(
    #     self,
    #     save_path,
    #     fig_width_in=3.25,
    #     dpi=300,
    #     cmap="hot",
    #     label=None,
    #     robot_image_path=None,
    #     size=0.01,

    # ):
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D  # noqa
    #     import numpy as np

    #     from scipy.ndimage import gaussian_filter  # add this import near the top of the function

    #     X, Y, Z, THETA, PHI = self.create_hemisphere_mesh(resolution=80)  # a bit higher res
    #     acc_mesh = self.interpolate_mse(X, Y, Z)

    #     # Smooth the accuracy map for a softer look
    #     acc_smooth = gaussian_filter(acc_mesh, sigma=1.0)

    #     # X, Y, Z, THETA, PHI = self.create_hemisphere_mesh(resolution=60)
    #     # acc_mesh = self.interpolate_mse(X, Y, Z)

    #     fig = plt.figure(figsize=(fig_width_in/2, fig_width_in * 0.85/2), dpi=dpi)
    #     ax = fig.add_subplot(111, projection="3d")

    #     light = (Y - Y.min()) / (Y.max() - Y.min() + 1e-6)
    #     colors = plt.cm.hot(acc_smooth)  # use smoothed values
    #     colors[..., :3] = colors[..., :3] * (0.6 + 0.4*light[..., None])
    #     # light = (Y - Y.min()) / (Y.max() - Y.min() + 1e-6)
    #     # colors = plt.cm.hot(acc_mesh)  # (n,m,4)
    #     # colors[..., :3] = colors[..., :3] * (0.6 + 0.4*light[..., None])  # brighten
    #     ax.grid(False)
    #     for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    #         axis._axinfo["grid"]["linewidth"] = 0



    #     ax.plot_surface(
    #         X, Y, Z,
    #         facecolors=colors,
    #         # facecolors=plt.cm.get_cmap(cmap)(acc_mesh),
    #         rstride=1, cstride=1,
    #         linewidth=0, antialiased=True,
    #         shade=False, alpha=0.6,
    #     )

    #     cam = self.camera_positions.copy()
    #     # cam = cam / np.linalg.norm(cam, axis=1, keepdims=True) * (self.radius * 1)
    #     # ax.scatter(
    #     #     cam[:, 0], cam[:, 1], cam[:, 2],
    #     #     s=10,
    #     #     c=self.mse_values,
    #     #     cmap=cmap,
    #     #     edgecolors="black",
    #     #     linewidths=0.4,
    #     #     depthshade=False,
    #     #     zorder=5,
    #     # )

    #     ax.set_xlabel("X (m)", fontsize=7)
    #     ax.set_ylabel("Y (m)", fontsize=7)
    #     ax.set_zlabel("Z (m)", fontsize=7, labelpad=4)
    #     ax.tick_params(labelsize=6)
    #     ax.set_box_aspect([1, 1, 0.5])
    #     ax.view_init(elev=28, azim=40)

    #     # move Z label to the left so it doesn't sit on the colorbar
    #     ax.zaxis.set_rotate_label(False)
    #     ax.zaxis.set_label_coords(-0.08, 0.5)   # (x, y) in axes coords

    #     mappable = plt.cm.ScalarMappable(cmap=cmap)
    #     mappable.set_array(acc_smooth)

    #     # mappable = plt.cm.ScalarMappable(cmap=cmap)
    #     # mappable.set_array(acc_mesh)
    #     cbar = fig.colorbar(
    #         mappable,
    #         ax=ax,
    #         fraction=0.046,
    #         pad=0.12,          # was 0.03 → push to the right
    #         orientation='horizontal',
    #     )
    #     cbar.ax.tick_params(labelsize=6)
    #     cbar.set_label("MSE", fontsize=5)

    #     # Bottom label removed to keep images clean
    #     # if label is not None:
    #     #     fig.text(0.5, 0.01, label, ha="center", va="bottom", fontsize=7)

    #     # Add robot image at the center, if provided
    #     if robot_image_path:
    #         try:
    #             add_robot_to_hemisphere_3d(
    #                 ax,
    #                 robot_image_path,
    #                 position=(-0.3, 0, 0.1*self.radius),   # scale with radius
    #                 size=0.01                           # proper proportional size
    #             )
    #         except Exception as e:
    #             print(f"Could not add robot image in plot_paper_3d: {e}")


    #     fig.tight_layout(pad=0.2)
    #     fig.savefig(save_path, dpi=dpi)
    #     if save_path.lower().endswith(".png"):
    #         fig.savefig(save_path.replace(".png", ".pdf"))
    #     plt.close(fig)

#################################################

    def plot_paper_3d(
            self,
            save_path,
            fig_width_in=3.25,
            dpi=300,
            cmap="hot_r",
            label=None,
            robot_image_path=None,
            size=0.02,
            vmin=None,      # Manual min for colorbar
            vmax=None,      # Manual max for colorbar
            use_power_norm=True,
        ):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        import numpy as np
        from matplotlib.colors import Normalize, PowerNorm
        from PIL import Image
        import tempfile
        import os

        # mesh + mse
        X, Y, Z, THETA, PHI = self.create_hemisphere_mesh(resolution=60)
        mse_mesh = self.interpolate_mse(X, Y, Z)

        # Larger canvas so margins are less significant after crop
        fig = plt.figure(figsize=(4.0, 4.0), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")

        # make axes occupy the full figure (no inner margins)
        ax.set_position([0.0, 0.0, 1.0, 1.0])

        # Determine colorbar limits
        if vmin is None:
            vmin = mse_mesh.min()
        if vmax is None:
            vmax = mse_mesh.max()

        # Use PowerNorm for better contrast when MSE range is small
        mse_range = vmax - vmin
        if use_power_norm and mse_range < 0.10:
            norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        # Get colormap and apply normalization (NO LIGHTING for consistency)
        cmap_obj = plt.cm.get_cmap(cmap)
        normalized_mse = norm(mse_mesh)
        colors = cmap_obj(normalized_mse)

        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["linewidth"] = 0

        # hemisphere surface
        ax.plot_surface(
            X, Y, Z,
            facecolors=colors,
            rstride=1, cstride=1,
            linewidth=0,
            antialiased=True,
            shade=False,
            alpha=0.85,
        )

        # make the sphere fill the axes
        ax.set_xlim(-self.radius * 1.05, self.radius * 1.05)
        ax.set_ylim(-self.radius * 1.05, self.radius * 1.05)
        ax.set_zlim(0, self.radius * 1.05)
        ax.set_box_aspect([1, 1, 0.6])
        ax.view_init(elev=28, azim=45)

        # NO AXES at all – just the sphere
        ax.set_axis_off()

        # OPTIONAL robot overlay
        if robot_image_path:
            from hemisphere_with_robot import add_robot_to_hemisphere_3d
            try:
                add_robot_to_hemisphere_3d(
                    ax,
                    robot_image_path,
                    position=(-0.3, 0, 0.1 * self.radius),
                    size=size,
                )
            except Exception as e:
                print(f"Could not add robot image in plot_paper_3d: {e}")

        # 1) Make figure fill space tightly - NO margins at all
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        # Disable automatic padding
        ax.margins(0, 0, 0)

        # Save to temporary location first (WITHOUT colorbar)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_path = tmp.name

        fig.savefig(temp_path, dpi=dpi, pad_inches=0.0, bbox_inches=None)
        plt.close(fig)

        # 2) VERY AGGRESSIVE cropping - remove all white/near-white pixels
        img = Image.open(temp_path).convert("RGB")
        img_array = np.array(img)

        # Use a more aggressive threshold - only skip nearly pure white
        threshold = 254
        is_white = (
            (img_array[:, :, 0] >= threshold) &
            (img_array[:, :, 1] >= threshold) &
            (img_array[:, :, 2] >= threshold)
        )

        non_white_rows = np.where(~is_white.all(axis=1))[0]
        non_white_cols = np.where(~is_white.all(axis=0))[0]

        if len(non_white_rows) > 0 and len(non_white_cols) > 0:
            # ZERO padding: as tight as possible
            top = non_white_rows.min()
            bottom = non_white_rows.max() + 1
            left = non_white_cols.min()
            right = non_white_cols.max() + 1
            img_cropped = img.crop((left, top, right, bottom))
        else:
            img_cropped = img

        # Clean up temp file
        os.unlink(temp_path)

        # 3) Now add colorbar as a separate element at the bottom
        # Create a new small figure just for the colorbar
        cbar_fig = plt.figure(figsize=(8, 0.5))
        cbar_ax = cbar_fig.add_axes([0.1, 0.3, 0.8, 0.4])

        # Use the same normalization as the hemisphere plot
        # sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        # sm.set_array([])

        # cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')

        # # Let matplotlib automatically determine optimal tick placement
        # # This ensures evenly-spaced, nicely-rounded tick values
        # cbar.ax.tick_params(labelsize=30)

        mappable = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        mappable.set_array(mse_mesh)   # <--- key change: attach your data

        cbar = plt.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=30)

        # Save colorbar to temp
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_cbar_path = tmp.name
        cbar_fig.savefig(temp_cbar_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05, facecolor='white')
        plt.close(cbar_fig)

        # Load colorbar and crop it
        cbar_img = Image.open(temp_cbar_path).convert("RGB")
        cbar_array = np.array(cbar_img)

        # Crop colorbar aggressively too
        is_white_cbar = (
            (cbar_array[:, :, 0] >= 254) &
            (cbar_array[:, :, 1] >= 254) &
            (cbar_array[:, :, 2] >= 254)
        )
        non_white_rows_cbar = np.where(~is_white_cbar.all(axis=1))[0]
        non_white_cols_cbar = np.where(~is_white_cbar.all(axis=0))[0]

        if len(non_white_rows_cbar) > 0 and len(non_white_cols_cbar) > 0:
            cbar_img_cropped = cbar_img.crop((
                non_white_cols_cbar.min(),
                non_white_rows_cbar.min(),
                non_white_cols_cbar.max() + 1,
                non_white_rows_cbar.max() + 1
            ))
        else:
            cbar_img_cropped = cbar_img

        os.unlink(temp_cbar_path)

        # 4) Combine main plot and colorbar vertically
        # Resize colorbar to match main plot width
        main_width = img_cropped.width
        cbar_aspect = cbar_img_cropped.width / cbar_img_cropped.height
        new_cbar_height = int(main_width / cbar_aspect)
        cbar_img_resized = cbar_img_cropped.resize((main_width, new_cbar_height), Image.Resampling.LANCZOS)

        # Create combined image with minimal gap
        gap = 10  # Small gap between plot and colorbar
        combined_height = img_cropped.height + gap + cbar_img_resized.height
        combined = Image.new('RGB', (main_width, combined_height), 'white')

        # Paste main plot and colorbar
        combined.paste(img_cropped, (0, 0))
        combined.paste(cbar_img_resized, (0, img_cropped.height + gap))

        # Save final combined image
        combined.save(save_path)

        # Also save as PDF if PNG
        if save_path.lower().endswith(".png"):
            pdf_path = save_path.replace(".png", ".pdf")
            img_cropped.save(pdf_path)

#################################################

    # def plot_paper_3d(
    #     self,
    #     save_path,
    #     fig_width_in=3.25,
    #     dpi=300,
    #     cmap="hot_r",
    #     label=None,
    #     robot_image_path=None,
    #     size=0.02,
    #     vmin=None,      # Manual min for colorbar
    #     vmax=None,      # Manual max for colorbar
    #     use_power_norm=True,
    # ):
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D  # noqa
    #     import numpy as np
    #     from matplotlib.colors import Normalize, PowerNorm

    # # mesh + mse
    #     X, Y, Z, THETA, PHI = self.create_hemisphere_mesh(resolution=60)
    #     mse_mesh = self.interpolate_mse(X, Y, Z)

    #     # Larger canvas so margins are less significant after crop
    #     fig = plt.figure(figsize=(4.0, 4.0), dpi=dpi)  # Increased from 2.2 to 4.0
    #     ax = fig.add_subplot(111, projection="3d")

    #     # make axes occupy the full figure (no inner margins)
    #     ax.set_position([0.0, 0.0, 1.0, 1.0])

    #     # Determine colorbar limits
    #     if vmin is None:
    #         vmin = mse_mesh.min()
    #     if vmax is None:
    #         vmax = mse_mesh.max()

    #     # Use PowerNorm for better contrast when MSE range is small
    #     mse_range = vmax - vmin
    #     if use_power_norm and mse_range < 0.10:
    #         norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    #     else:
    #         norm = Normalize(vmin=vmin, vmax=vmax)

    #     # Get colormap and apply normalization (NO LIGHTING for consistency)
    #     cmap_obj = plt.cm.get_cmap(cmap)
    #     normalized_mse = norm(mse_mesh)
    #     colors = cmap_obj(normalized_mse)
    #     # Removed lighting to match top-down MSE colors exactly

    #     ax.grid(False)
    #     for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    #         axis._axinfo["grid"]["linewidth"] = 0

    #     # hemisphere surface
    #     ax.plot_surface(
    #         X, Y, Z,
    #         facecolors=colors,
    #         rstride=1, cstride=1,
    #         linewidth=0,
    #         antialiased=True,
    #         shade=False,
    #         alpha=0.85,
    #     )

    #     # make the sphere fill the axes
    #     ax.set_xlim(-self.radius * 1.05, self.radius * 1.05)
    #     ax.set_ylim(-self.radius * 1.05, self.radius * 1.05)
    #     ax.set_zlim(0, self.radius * 1.05)
    #     ax.set_box_aspect([1, 1, 0.6])
    #     ax.view_init(elev=28, azim=45)

    #     # NO AXES at all – just the sphere
    #     ax.set_axis_off()

    #     # OPTIONAL robot overlay
    #     if robot_image_path:
    #         from hemisphere_with_robot import add_robot_to_hemisphere_3d
    #         try:
    #             add_robot_to_hemisphere_3d(
    #                 ax,
    #                 robot_image_path,
    #                 position=(-0.3, 0, 0.1 * self.radius),
    #                 size=size,
    #             )
    #         except Exception as e:
    #             print(f"Could not add robot image in plot_paper_3d: {e}")



    #     # 1) Make figure fill space tightly - NO margins at all
    #     # Remove all padding/spacing completely
    #     fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    #     # Disable automatic padding
    #     ax.margins(0, 0, 0)

    #     # Save to temporary location first
    #     import tempfile
    #     with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
    #         temp_path = tmp.name

    #     fig.savefig(temp_path, dpi=dpi, pad_inches=0.0, bbox_inches=None)
    #     plt.close(fig)

    #     # 2) VERY AGGRESSIVE cropping - remove all white/near-white pixels
    #     img = Image.open(temp_path).convert("RGB")
    #     img_array = np.array(img)

    #     # Find non-white pixels (anything not pure white)
    #     # Use a more aggressive threshold - even slightly off-white gets included
    #     threshold = 254  # Very aggressive - only skip nearly pure white

    #     # Check all three channels - pixel must be white in ALL channels to be cropped
    #     is_white = (img_array[:, :, 0] >= threshold) & \
    #                (img_array[:, :, 1] >= threshold) & \
    #                (img_array[:, :, 2] >= threshold)

    #     # Find rows and columns with non-white content
    #     non_white_rows = np.where(~is_white.all(axis=1))[0]
    #     non_white_cols = np.where(~is_white.all(axis=0))[0]

    #     if len(non_white_rows) > 0 and len(non_white_cols) > 0:
    #         # Get bounding box with ZERO padding
    #         top = non_white_rows.min()
    #         bottom = non_white_rows.max() + 1
    #         left = non_white_cols.min()
    #         right = non_white_cols.max() + 1

    #         # Crop to this tight bounding box
    #         img_cropped = img.crop((left, top, right, bottom))
    #     else:
    #         img_cropped = img

    #     # Clean up temp file
    #     import os
    #     os.unlink(temp_path)

    #     # Save final cropped image
    #     img_cropped.save(save_path)

    #     # 3) Also save as PDF if PNG
    #     if save_path.lower().endswith(".png"):
    #         pdf_path = save_path.replace(".png", ".pdf")
    #         img_cropped.save(pdf_path)




        # Aggressively remove outer margins (no white frame)
        # fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        # fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)

        # # also save PDF version if PNG
        # if save_path.lower().endswith(".png"):
        #     fig.savefig(
        #         save_path.replace(".png", ".pdf"),
        #         dpi=dpi,
        #         bbox_inches="tight",
        #         pad_inches=0.0,
        # #     )
        # plt.close(fig)



    # def plot_paper_3d(
    #     self,
    #     save_path,
    #     fig_width_in=3.25,
    #     dpi=300,
    #     cmap="hot",
    #     label=None,
    #     robot_image_path=None,
    #     size=0.01,
    # ):
    #     import matplotlib.pyplot as plt
    #     from mpl_toolkits.mplot3d import Axes3D  # noqa
    #     import numpy as np

    #     # mesh + accuracy
    #     X, Y, Z, THETA, PHI = self.create_hemisphere_mesh(resolution=60)
    #     acc_mesh = self.interpolate_mse(X, Y, Z)

    #     # smaller, more square canvas – mostly sphere
    #     fig = plt.figure(figsize=(fig_width_in * 0.55, fig_width_in * 0.55), dpi=dpi)
    #     ax = fig.add_subplot(111, projection="3d")

    #     # coloring
    #     light = (Y - Y.min()) / (Y.max() - Y.min() + 1e-6)
    #     colors = plt.cm.hot(acc_mesh)
    #     colors[..., :3] = colors[..., :3] * (0.6 + 0.4 * light[..., None])

    #     ax.grid(False)
    #     for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    #         axis._axinfo["grid"]["linewidth"] = 0

    #     # hemisphere surface
    #     ax.plot_surface(
    #         X, Y, Z,
    #         facecolors=colors,
    #         rstride=1, cstride=1,
    #         linewidth=0,
    #         antialiased=True,
    #         shade=False,
    #         alpha=0.7,
    #     )

    #     # make the sphere fill the axes


    #     ax.set_xlim(-self.radius * 1.05, self.radius * 1.05)
    #     ax.set_ylim(-self.radius * 1.05, self.radius * 1.05)
    #     ax.set_zlim(0, self.radius * 1.05)
    #     ax.set_box_aspect([1, 1, 0.6])
    #     ax.view_init(elev=28, azim=40)

    #     # if want_ax:
    #     #             # very small axis labels & ticks
    #     #     ax.set_xlabel("X (m)", fontsize=5, labelpad=0)
    #     #     ax.set_ylabel("Y (m)", fontsize=5, labelpad=0)
    #     #     ax.set_zlabel("Z (m)", fontsize=5, labelpad=2)
    #     #     ax.tick_params(labelsize=4, pad=1)

    #     #     # move Z label left so it doesn’t clash with colorbar
    #     #     ax.zaxis.set_rotate_label(False)
    #     #     ax.zaxis.set_label_coords(-0.08, 0.5)

    #     ax.set_axis_off()




    #     # tiny horizontal colorbar
    #     mappable = plt.cm.ScalarMappable(cmap=cmap)
    #     mappable.set_array(acc_mesh)
    #     cbar = fig.colorbar(
    #         mappable,
    #         ax=ax,
    #         fraction=0.035,   # slimmer bar
    #         pad=0.02,         # closer to axes
    #         orientation="horizontal",
    #     )
    #     cbar.ax.tick_params(labelsize=4, pad=0)
    #     cbar.set_label("MSE", fontsize=5)

    #     # optional robot, but small
    #     if robot_image_path:
    #         from hemisphere_with_robot import add_robot_to_hemisphere_3d
    #         try:
    #             add_robot_to_hemisphere_3d(
    #                 ax,
    #                 robot_image_path,
    #                 position=(-0.3, 0, 0.1 * self.radius),
    #                 size=size,   # already tiny (~1%)
    #             )
    #         except Exception as e:
    #             print(f"Could not add robot image in plot_paper_3d: {e}")

    #     # aggressively remove outer margins
    #     fig.subplots_adjust(left=0.0, right=1.0, bottom=0.05, top=0.95)
    #     fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    #     if save_path.lower().endswith(".png"):
    #         fig.savefig(save_path.replace(".png", ".pdf"), dpi=dpi,
    #                     bbox_inches="tight", pad_inches=0.01)
    #     plt.close(fig)

    def plot_paper_gradient3d(
        self,
        save_path,
        fig_width_in=3.25,
        dpi=300,
        surf_cmap="hot_r",
        cam_cmap="hot_r",
        label=None,
        vmin=None,
        vmax=None,
        use_power_norm=True,
    ):
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        from matplotlib.colors import Normalize, PowerNorm

        if self.camera_positions is None or self.mse_values is None:
            raise ValueError("Need camera_positions + mse_values")

        # 1) mesh + gradient
        X, Y, Z, THETA, PHI = self.create_hemisphere_mesh(resolution=50)
        mse_mesh = self.interpolate_mse(X, Y, Z)
        dx, dy = self.compute_gradient(mse_mesh, X, Y, Z)
        grad_mag = np.sqrt(dx**2 + dy**2)

        # 2) figure
        fig = plt.figure(figsize=(fig_width_in/2, fig_width_in * 0.85/2), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["linewidth"] = 0

        # Determine colorbar limits
        if vmin is None:
            vmin = mse_mesh.min()
        if vmax is None:
            vmax = mse_mesh.max()

        # Use PowerNorm for better contrast when MSE range is small
        mse_range = vmax - vmin
        if use_power_norm and mse_range < 0.10:
            norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        # Apply normalization and colormap (NO LIGHTING for consistency with top-down)
        cmap_obj = plt.cm.get_cmap(surf_cmap)  # Fixed: was surf_cmap + 'hot_r'
        normalized_mse = norm(mse_mesh)
        colors = cmap_obj(normalized_mse)
        # Removed lighting to match top-down MSE view better
        # 3) surface

        ax.plot_surface(
            X, Y, Z,
            # facecolors=plt.cm.get_cmap(surf_cmap)(grad_mag),
            facecolors = colors,
            rstride=1, cstride=1,
            linewidth=0, antialiased=False,
            shade=False,
            alpha=0.6,
            zorder=1,
        )

        # # 4) cameras (pushed out)
        # cam = self.camera_positions.copy()
        # cam = cam / np.linalg.norm(cam, axis=1, keepdims=True) * (self.radius * 1)
        # ax.scatter(
        #     cam[:, 0], cam[:, 1], cam[:, 2],
        #     s=10,
        #     c=self.mse_values,
        #     cmap=cam_cmap,
        #     edgecolors="black",
        #     linewidths=0.4,
        #     depthshade=False,
        #     zorder=5,
        # )

        # basic axes
        ax.set_xlabel("X (m)", fontsize=7)
        ax.set_ylabel("Y (m)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_box_aspect([1, 1, 0.5])
        ax.view_init(elev=28, azim=45)
        ax.set_axis_off()


        # 5) HARD-HIDE the original Z axis so it can’t appear on the right
        ax.set_zlabel("")                    # blank
        try:
            ax.zaxis.label.set_visible(False)
        except Exception:
            pass
        # hide z ticks & tick labels
        try:
            for item in ax.zaxis.get_ticklines() + ax.zaxis.get_ticklabels():
                item.set_visible(False)
        except Exception:
            pass
        # in some mpl versions:
        if hasattr(ax, "w_zaxis"):
            ax.w_zaxis.line.set_lw(0.)
            ax.w_zaxis.set_pane_color((0, 0, 0, 0))

        # 6) colorbar on the right
        mappable = plt.cm.ScalarMappable(cmap=surf_cmap)
        mappable.set_array(grad_mag)

        cbar = fig.colorbar(
            mappable,
            ax=ax,
            fraction=0.046,
            pad=0.06,   # push out a bit
            orientation='horizontal'
        )
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label("Gradient Magnitude", fontsize=7)

        # 7) now draw OUR Z on the left of the figure -> this will not move
        fig.text(
            0.045, 0.55,  # left margin
            "Z (m)",
            rotation=90,
            ha="center",
            va="center",
            fontsize=7,
        )

        # Bottom label removed to keep images clean
        # if label is not None:
        #     fig.text(0.5, 0.01, label, ha="center", va="bottom", fontsize=7)

        fig.tight_layout(pad=0.4)
        fig.savefig(save_path, dpi=dpi)
        if save_path.lower().endswith(".png"):
            fig.savefig(save_path.replace(".png", ".pdf"))
        plt.close(fig)


    @staticmethod

    def make_cvpr_3x2_spheres(components_data, output_path, cmap="hot_r"):
        """
        CVPR-style 3x2 grid, but each panel is the 3D hemisphere surface.
        components_data: list of dicts:
            {
              "name": "height",
              "positions": (N,3) np.array,
              "accuracy":  (N,)   np.array
            }
        Uses first 6 entries.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        # 2-column CVPR width
        fig_w = 6.75
        fig_h = 7.0
        fig = plt.figure(figsize=(fig_w/2, fig_h/2), dpi=300)

        # only 6
        components_data = components_data[:6]

        # to get global min/max for shared colorbar
        all_vals = []

        for idx, comp in enumerate(components_data):
            # 3 rows x 2 cols
            ax = fig.add_subplot(3, 2, idx + 1, projection="3d")
            ax.grid(False)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis._axinfo["grid"]["linewidth"] = 0


            vis = HemisphereHeatmap(n_cameras=len(comp["positions"]))
            vis.camera_positions = comp["positions"]
            vis.mse_values = comp["mse"]
            vis.mse_values = comp["mse"]

            X, Y, Z, THETA, PHI = vis.create_hemisphere_mesh(resolution=60)
            mse_mesh = vis.interpolate_mse(X, Y, Z)
            from scipy.ndimage import gaussian_filter
            mse_smooth = gaussian_filter(mse_mesh, sigma=1.5)
            all_vals.append(mse_mesh)

            # Scale to 1.2m physical size for display
            X = X * 1.2
            Y = Y * 1.2
            Z = Z * 1.2


            # BETTER LIGHTING
            y_light = (Y - Y.min()) / (Y.max() - Y.min() + 1e-6)
            z_light = (Z / (Z.max() + 1e-6))
            light = 0.5 * y_light + 0.5 * z_light

            colors = plt.cm.hot_r(mse_smooth)
            colors[..., :3] = colors[..., :3] * (0.5 + 0.5*light[..., None])

            # BEAUTIFUL SURFACE
            ax.plot_surface(
                X, Y, Z,
                facecolors=colors,
                rstride=1, cstride=1,
                linewidth=0,
                antialiased=True,   # ← TRUE!
                shade=False,
                alpha=0.8,          # ← 0.8!
            )

            # light = (Y - Y.min()) / (Y.max() - Y.min() + 1e-6)
            # colors = plt.cm.hot(acc_mesh)  # (n,m,4)
            # colors[..., :3] = colors[..., :3] * (0.6 + 0.4*light[..., None])  # brighten

            # ax.plot_surface(
            #     X, Y, Z,
            #     # facecolors=plt.cm.get_cmap(cmap)(acc_mesh),
            #     facecolors = colors,
            #     rstride=1, cstride=1,
            #     linewidth=0, antialiased=False,
            #     shade=False,
            #     alpha=0.65,
            # )

            # ADD ROBOT IMAGE AT CENTER
            # try:
            #     from PIL import Image
            #     import matplotlib.patches as mpatches
            #     from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            #     # Load and rotate robot image
            #     robot_img = Image.open("C:\\Users\\rkhan\\Downloads\\robot.png")
            #     robot_img = robot_img.rotate(90, expand=True)  # Counter-clockwise

            #     if robot_img.mode != 'RGBA':
            #         robot_img = robot_img.convert('RGBA')

            #     # Make semi-transparent
            #     img_array = np.array(robot_img)
            #     if img_array.shape[2] == 4:
            #         img_array[:, :, 3] = (img_array[:, :, 3] * 0.6).astype(np.uint8)
            #     robot_img_transparent = Image.fromarray(img_array)

            try:
                ROBOT_IMAGE_PATH_3D = "C:\\Users\\rkhan\\Downloads\\robot.png"
                add_robot_to_hemisphere_3d(
                    ax,
                    ROBOT_IMAGE_PATH_3D,
                    position=(-0.3, 0.7, 0.2*1.2),   # scale with 1.2m radius
                    size=0.010                         # proper proportional size (40%)
                )
            except Exception as e:
                print(f"Could not add robot to panel {comp['name']}: {e}")

            #     # Create a vertical plane for robot at center
            #     robot_size = 0.3
            #     vertices = [
            #         [-robot_size/2, 0, 0],           # Bottom left
            #         [robot_size/2, 0, 0],            # Bottom right
            #         [robot_size/2, 0, robot_size],   # Top right
            #         [-robot_size/2, 0, robot_size]   # Top left
            #     ]

            #     # Add as a semi-transparent plane
            #     poly = Poly3DCollection([vertices], alpha=0.7, facecolor='lightgray',
            #                            edgecolor='white', linewidth=0.5, zorder=100)
            #     ax.add_collection3d(poly)

            #     # Add text indicator
            #     ax.text(0, 0, 0.1, '🤖', fontsize=20, ha='center', va='center',
            #            zorder=200, bbox=dict(boxstyle='round', facecolor='white',
            #            alpha=0.6, pad=0.2, edgecolor='none'))

            # except Exception as e:
            #     # Fallback: just add emoji
            #     ax.text(0, 0, 0.1, '🤖', fontsize=20, ha='center', va='center',
            #            zorder=200, bbox=dict(boxstyle='round', facecolor='white',
            #            alpha=0.6, pad=0.2, edgecolor='none'))

            # cameras → push out
            # cam = vis.camera_positions.copy()
            # cam = cam / np.linalg.norm(cam, axis=1, keepdims=True) * (vis.radius * 1)
            # ax.scatter(
            #     cam[:, 0], cam[:, 1], cam[:, 2],
            #     s=10,
            #     c=vis.accuracy_values,
            #     cmap=cmap,
            #     edgecolors="black",
            #     linewidths=0.35,
            #     depthshade=False,
            #     zorder=5,
            # )

            # ax.set_xlabel("X (m)", fontsize=6)
            # ax.set_ylabel("Y (m)", fontsize=6)
            # ax.set_zlabel("Z (m)", fontsize=6)
            # ax.tick_params(labelsize=5)
            ax.set_box_aspect([1, 1, 0.5])
            # ax.set_xlim([-1.3, 1.3])
            # ax.set_ylim([-1.3, 1.3])
            # ax.set_zlim([0, 1.3])
            # alternate views a bit so they don’t self-hide
            if idx % 2 == 0:
                ax.view_init(elev=26, azim=38)
            else:
                ax.view_init(elev=26, azim=135)

            # Label removed to keep images clean
            # ax.text2D(
            #     0.5, -0.18,
            #     comp["name"].replace("_", " "),
            #     transform=ax.transAxes,
            #     ha="center",
            #     va="top",
            #     fontsize=6.2,
            # )

        # shared colorbar
        all_vals = np.array(all_vals)
        vmin = float(all_vals.min())
        vmax = float(all_vals.max())

        cax = fig.add_axes([0.92, 0.10, 0.015, 0.80])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("MSE", fontsize=7)

        fig.subplots_adjust(
            left=0.03, right=0.90, top=0.97, bottom=0.05,
            wspace=0.05, hspace=0.25
        )

        fig.savefig(output_path, dpi=300)
        plt.close(fig)
    @staticmethod
    def save_shared_accuracy_colorbar(
        save_path,
        vmin=0.0,
        vmax=1.0,
        cmap="hot_r",
        fig_width_in=3.25,
        dpi=300,
    ):
        """
        Save a standalone horizontal accuracy colorbar to use under a 2x3 layout.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        fig, ax = plt.subplots(figsize=(fig_width_in, 0.35), dpi=dpi)

        norm = Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            fraction=0.8,
            pad=0.3,
        )
        cbar.set_label("MSE", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        # no actual axes content
        ax.set_axis_off()

        fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.3)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)

        if save_path.lower().endswith(".png"):
            fig.savefig(
                save_path.replace(".png", ".pdf"),
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.0,
            )

        plt.close(fig)


    @staticmethod
    def make_cvpr_2x3_grid(components_data, output_path, cam_cmap="hot_r"):
        """
        CVPR-style 3x2 figure (3 rows, 2 cols) for all components.
        Features:
        - No borders around subplots
        - Colorbar at bottom (horizontal)
        - Larger circles with ZERO margins between them
        - Ultra-tight layout

        components_data: list of dicts like
            {
            "name": "height",
            "positions": np.array shape (N,3),
            "mse": np.array shape (N,)
            }
        Only the first 6 items are used.
        """
        # Vertical figure for 3x2 layout (3 rows, 2 columns)
        fig_w = 7.0   # Width for 2 columns
        fig_h = 10.0  # Height for 3 rows

        # Create 3x2 grid (3 rows, 2 columns)
        fig, axes = plt.subplots(3, 2, figsize=(fig_w, fig_h), dpi=300)
        axes = axes.flatten()

        all_mesh_vals = []

        # Only use first 6 components
        components_data = components_data[:6]

        for idx, comp in enumerate(components_data):
            ax = axes[idx]

            vis = HemisphereHeatmap(n_cameras=len(comp["positions"]))
            vis.camera_positions = comp["positions"]
            vis.mse_values = comp["mse"]

            # Create mesh and interpolate
            X, Y, Z, THETA, PHI = vis.create_hemisphere_mesh(resolution=90)
            mse_mesh = vis.interpolate_mse(X, Y, Z)
            all_mesh_vals.append(mse_mesh)

            # Plot heatmap
            ax.contourf(X, Y, mse_mesh, levels=25, cmap=cam_cmap)

            # Add circle outline (black border only)
            circle = plt.Circle((0, 0), 1.2, fill=False, edgecolor="black", linewidth=2, zorder=30)
            ax.add_patch(circle)

            # Tidy axes - NO BORDERS, MINIMAL MARGINS
            ax.set_aspect("equal")
            ax.set_xlim(-1.2*1.01, 1.2*1.01)  # Almost zero margin (1% only)
            ax.set_ylim(-1.2*1.01, 1.2*1.01)
            ax.set_xticks([])
            ax.set_yticks([])

            # Remove all spines (borders)
            for spine in ax.spines.values():
                spine.set_visible(False)

            ##### comonents labels ###

            # # Component label below panel
            # ax.text(
            #     0.5, -0.05,
            #     comp["name"].replace("_", " "),
            #     ha="center", va="top",
            #     transform=ax.transAxes,
            #     fontsize=10,
            #     weight='normal'
            # )

        # Calculate colorbar range from all data
        all_mesh_vals = np.array(all_mesh_vals)
        vmin = float(all_mesh_vals.min())
        vmax = float(all_mesh_vals.max())

        # Add HORIZONTAL colorbar at BOTTOM
        # [left, bottom, width, height]
        cax = fig.add_axes([0.15, 0.02, 0.70, 0.015])  # Horizontal at bottom
        sm = plt.cm.ScalarMappable(cmap=cam_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        # ticks = np.linspace(vmin, vmax, 5)
        # cbar = plt.colorbar(contourf, ax=ax, ticks=ticks, shrink=0.8, pad=0.05,orientation='horizontal')
        # Instead of:
        # ticks = np.linspace(vmin, vmax, 5)
        # cb = fig.colorbar(sm, cax=cax, ticks=ticks, orientation='horizontal')

        # Use:
        cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=20)

        # cb = fig.colorbar(sm, cax=cax, ticks=ticks, orientation='horizontal')
        # cb.ax.tick_params(labelsize=20)
        # cb.set_label("MSE", fontsize=10, labelpad=3)

        # ULTRA-TIGHT layout - ALMOST ZERO spacing between subplots
        fig.subplots_adjust(
            left=0.01,      # Almost zero left margin
            right=0.99,     # Almost zero right margin
            top=0.99,       # Almost zero top margin
            bottom=0.06,    # Small space for colorbar at bottom
            wspace=0.01,    # ALMOST ZERO horizontal space between plots
            hspace=0.08,    # MINIMAL vertical space between plots (just for labels)
        )

        fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

        print(f"Saved CVPR 3x2 grid to {output_path}")

    @staticmethod
    def make_height_cvpr_2x3_from_folders(
        model_dirs,
        output_path,
        csv_name="height_hemisphere_data.csv",
        cmap="hot_r",
    ):
        """
        Render a CVPR-style 2x3 of HEIGHT (top-down) maps with ONE shared
        horizontal colorbar (no per-panel bars, no gradients/arrows), using
        the same style as make_cvpr_2x3_grid.

        Args:
            model_dirs (list[str|Path]): six folders (in display order). Each
                folder must contain a CSV with columns x,y,z,mse (csv_name).
            output_path (str|Path): destination .png (and .pdf sibling).
            csv_name (str): filename of the per-model hemisphere CSV.
            cmap (str): Matplotlib colormap (default 'hot_r').

        Expected CSV columns: x,y,z,mse  (camera positions + MSE per camera).
        """
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path
        from matplotlib.ticker import MaxNLocator

        # --- load data from the six folders into the structure used by make_cvpr_2x3_grid ---
        components = []
        for d in model_dirs[:6]:
            d = Path(d)
            df = pd.read_csv(d / csv_name)
            # required columns
            for col in ("x", "y", "z", "mse"):
                if col not in df.columns:
                    raise ValueError(f"Missing column '{col}' in {d/csv_name}")
            positions = df[["x", "y", "z"]].to_numpy(dtype=float)
            mse = df["mse"].to_numpy(dtype=float)
            components.append({"name": d.name, "positions": positions, "mse": mse})

        # --- render identical to make_cvpr_2x3_grid (flat circles + one bottom colorbar) ---
        fig_w, fig_h = 7.0, 10.0
        fig, axes = plt.subplots(3, 2, figsize=(fig_w, fig_h), dpi=300)
        axes = axes.flatten()

        # we’ll compute a shared vmin/vmax over all six meshes
        all_mesh_vals = []

        # draw panels
        for idx, comp in enumerate(components):
            ax = axes[idx]

            vis = HemisphereHeatmap(n_cameras=len(comp["positions"]))
            vis.camera_positions = comp["positions"]
            vis.mse_values = comp["mse"]

            # same mesh+contourf style as make_cvpr_2x3_grid
            X, Y, Z, THETA, PHI = vis.create_hemisphere_mesh(resolution=90)
            mse_mesh = vis.interpolate_mse(X, Y, Z)
            all_mesh_vals.append(mse_mesh)

            ax.contourf(X, Y, mse_mesh, levels=25, cmap=cmap)

            # circular boundary only
            circle = plt.Circle((0, 0), 1.2, fill=False, edgecolor="black", linewidth=2, zorder=30)
            ax.add_patch(circle)

            # clean axes
            ax.set_aspect("equal")
            ax.set_xlim(-1.2*1.01, 1.2*1.01)
            ax.set_ylim(-1.2*1.01, 1.2*1.01)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

        # shared colorbar range
        all_mesh_vals = np.array(all_mesh_vals)
        vmin = float(all_mesh_vals.min())
        vmax = float(all_mesh_vals.max())

        # single horizontal colorbar at bottom
        cax = fig.add_axes([0.15, 0.02, 0.70, 0.015])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.locator = MaxNLocator(nbins=5)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=20)

        # ultra-tight layout like make_cvpr_2x3_grid
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.06, wspace=0.01, hspace=0.08)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        # save PDF sibling
        if output_path.suffix.lower() == ".png":
            fig = None  # suppress reuse
            # write a quick PDF by reusing the saved PNG if needed externally
            # users usually want a PNG+PDF pair like other helpers
            try:
                import PIL.Image as Image
                im = Image.open(output_path)
                im.save(output_path.with_suffix(".pdf"))
            except Exception:
                pass
        print(f"Saved HEIGHT CVPR 2×3 to {output_path}")



    @staticmethod
    def make_height_cvpr_2x3_from_models(
        base_dir,
        out_path,
        pattern="output/height_robot_topdown_cvpr.png",
        titles=(
            "VAE-128", "VAE-256",
            "VGG19-128", "VGG19-256",
            "ResNet50-128", "ResNet50-256",
        ),
        model_order=(
            "vp_conv_vae_128", "vp_conv_vae_256",
            "vp_vgg19_128",    "vp_vgg19_256",
            "vp_resnet50_128", "vp_resnet50_256",
        ),
        dpi=300,
        cmap="hot_r",
        vmin=None,
        vmax=None,
        tight_crop_white=True,
    ):
        """
        Build a CVPR-style 2x3 grid using only the HEIGHT panels
        from six model folders (128/256 for VAE, VGG19, ResNet50).

        Args:
            base_dir (str|Path): parent dir containing the model folders.
            out_path (str|Path): where to save the combined PNG (and a PDF sibling).
            pattern (str): relative path to the height image inside each model folder.
                           Change if your per-model height panel has another name.
            titles (tuple[str]): six short labels shown under each circle (optional).
            model_order (tuple[str]): prefixes used to find the folders in order.
            vmin, vmax (float|None): optional shared color scale labels for the bar.
                                     If None, we still draw a bar with automatic ticks.
            tight_crop_white (bool): aggressively crop white borders of source images.

        Notes:
            - We do not try to re-color the images (they’re already colored);
              the bottom colorbar is created with the same colormap purely for
              consistent presentation. If you want the bar to match your data
              range numerically, set vmin/vmax to your global MSE min/max.
        """
        base_dir = Path(base_dir)
        six_images = []

        # Allow a couple of common filenames just in case
        candidate_names = [pattern,
                           "output/height_topdown_cvpr.png",
                           "output/height_cvpr.png",
                           "height_robot_topdown_cvpr.png",
                           "height_topdown_cvpr.png"]

        # Resolve model subfolders in the exact order requested
        resolved_dirs = []
        for prefix in model_order:
            matches = sorted([p for p in base_dir.iterdir()
                              if p.is_dir() and p.name.startswith(prefix)])
            if not matches:
                raise FileNotFoundError(f"No folder starting with '{prefix}' in {base_dir}")
            resolved_dirs.append(matches[0])

        # Load each height panel
        for mdir in resolved_dirs:
            img_path = None
            for cand in candidate_names:
                trial = mdir / cand
                if trial.exists():
                    img_path = trial
                    break
            if img_path is None:
                raise FileNotFoundError(
                    f"Could not find a height panel in '{mdir}'. "
                    f"Tried: {candidate_names}"
                )

            img = Image.open(img_path).convert("RGB")
            if tight_crop_white:
                arr = np.array(img)
                # Crop nearly-white borders
                thr = 252
                is_white = (arr[..., 0] >= thr) & (arr[..., 1] >= thr) & (arr[..., 2] >= thr)
                rows = np.where(~is_white.all(axis=1))[0]
                cols = np.where(~is_white.all(axis=0))[0]
                if rows.size and cols.size:
                    img = img.crop((cols.min(), rows.min(), cols.max()+1, rows.max()+1))
            six_images.append(img)

        # Layout constants
        nrows, ncols = 3, 2
        w, h = six_images[0].size
        pad = int(0.07 * w)        # padding between panels
        title_h = int(0.18 * h)    # space for tiny title under each panel
        bar_h = int(0.22 * h)      # colorbar height
        bar_gap = int(0.12 * h)

        canvas_w = ncols * w + (ncols - 1) * pad
        canvas_h = nrows * (h + title_h) + (nrows - 1) * pad + bar_gap + bar_h

        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

        # Paste six images
        idx = 0
        draw_positions = []
        for r in range(nrows):
            for c in range(ncols):
                x = c * (w + pad)
                y = r * (h + title_h + pad)
                canvas.paste(six_images[idx], (x, y))
                draw_positions.append((x, y))
                idx += 1

        # Add tiny titles (optional)
        if titles is not None:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("arial.ttf", size=max(12, h // 18))
            except Exception:
                font = ImageFont.load_default()
            for i, (x, y) in enumerate(draw_positions):
                label = titles[i] if i < len(titles) else ""
                tw, th = draw.textbbox((0, 0), label, font=font)[2:]
                tx = x + (w - tw) // 2
                ty = y + h + max(2, title_h // 5)
                draw.text((tx, ty), label, fill="black", font=font)

        # Create a separate matplotlib colorbar to append at the bottom
        # (uses the same colormap for visual consistency)
        fig = plt.figure(figsize=(8, 0.6), dpi=dpi)
        ax = fig.add_axes([0.08, 0.35, 0.84, 0.3])
        if vmin is None or vmax is None:
            # Arbitrary 0..1 range if not specified (purely for appearance)
            vmin_, vmax_ = 0.0, 1.0
        else:
            vmin_, vmax_ = vmin, vmax
        norm = Normalize(vmin=vmin_, vmax=vmax_)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=35)
        # cbar.set_label("MSE", fontsize=20)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            bar_path = tmp.name
        fig.savefig(bar_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)

        bar_img = Image.open(bar_path).convert("RGB")
        os.remove(bar_path)

        # Crop the bar tightly
        barr = np.array(bar_img)
        thr = 252
        mask = ~((barr[...,0]>=thr)&(barr[...,1]>=thr)&(barr[...,2]>=thr))
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size and cols.size:
            bar_img = bar_img.crop((cols.min(), rows.min(), cols.max()+1, rows.max()+1))

        # Resize colorbar to full canvas width and paste
        bar_img = bar_img.resize((canvas_w, bar_h), Image.Resampling.LANCZOS)
        canvas.paste(bar_img, (0, canvas_h - bar_h))

        # Save outputs
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        if out_path.suffix.lower() == ".png":
            canvas.save(out_path.with_suffix(".pdf"))
    # @staticmethod
    # def make_cvpr_2x3_grid(components_data, output_path, cam_cmap="hot_r"):
    #     """
    #     CVPR-style 3x2 figure (3 rows, 2 cols) for all components.

    #     components_data: list of dicts like
    #         {
    #         "name": "height",
    #         "positions": np.array shape (N,3),
    #         "accuracy": np.array shape (N,)
    #         }
    #     Only the first 6 items are used.
    #     """
    #     # CVPR 2-column width ≈ 6.75 in
    #     fig_w = 6.75
    #     fig_h = 7.0   # taller than 2x3 to fit 3 rows

    #     # make 3x2 axes
    #     fig, axes = plt.subplots(3, 2, figsize=(fig_w/2, fig_h/2), dpi=300)
    #     axes = axes.reshape(3, 2)
    #     # fig.patch.set_facecolor("#101014")
    #     # for ax in axes.ravel():
    #     #     ax.set_facecolor("#101014")

    #     all_mesh_vals = []

    #     from hemisphere_heatmap import HemisphereHeatmap  # if this is the same file, it's fine

    #     # we only need 6
    #     components_data = components_data[:6]

    #     for idx, comp in enumerate(components_data):
    #         row = idx // 2
    #         col = idx % 2
    #         ax = axes[row, col]

    #         vis = HemisphereHeatmap(n_cameras=len(comp["positions"]))
    #         vis.camera_positions = comp["positions"]
    #         vis.mse_values = comp["mse"]
    #         vis.mse_values = comp["mse"]

    #         # mesh + interp
    #         X, Y, Z, THETA, PHI = vis.create_hemisphere_mesh(resolution=90)
    #         mse_mesh = vis.interpolate_mse(X, Y, Z)
    #         all_mesh_vals.append(mse_mesh)

    #         # heatmap
    #         ax.contourf(X, Y, mse_mesh, levels=25, cmap=cam_cmap)

    #         # # cameras
    #         # ax.scatter(
    #         #     vis.camera_positions[:, 0],
    #         #     vis.camera_positions[:, 1],
    #         #     c=vis.accuracy_values,
    #         #     cmap=cam_cmap,
    #         #     s=10,
    #         #     edgecolors="black",
    #         #     linewidths=0.25,
    #         #     zorder=5,
    #         # )

    #         # circle outline
    #         # thick faint ring
    #         ax.add_patch(plt.Circle((0,0), 1.2, fill=False, edgecolor="white", linewidth=2.0, alpha=0.25, zorder=1))
    #         # sharp dark ring on top
    #         ax.add_patch(plt.Circle((0,0), 1.2, fill=False, edgecolor="black", linewidth=0.8, zorder=30))

    #         # circ = plt.Circle((0, 0), 1.0, fill=False, edgecolor="black", linewidth=0.7)
    #         # ax.add_patch(circ)

    #         # tidy axes
    #         ax.set_aspect("equal")
    #         # ax.set_xlim(-1.05, 1.05)
    #         # ax.set_ylim(-1.05, 1.05)
    #         ax.set_xlim(-1.2*1.05, 1.2*1.05)
    #         ax.set_ylim(-1.2*1.05, 1.2*1.05)

    #         ax.set_xticks([])
    #         ax.set_yticks([])

    #         # component label under panel
    #         ax.text(
    #             0.5, -0.10,
    #             comp["name"].replace("_", " "),
    #             ha="center", va="top",
    #             transform=ax.transAxes,
    #             fontsize=6.5,
    #         )

    #     # shared colorbar on right
    #     all_mesh_vals = np.array(all_mesh_vals)
    #     vmin = float(all_mesh_vals.min())
    #     vmax = float(all_mesh_vals.max())

    #     # [left, bottom, width, height]
    #     cax = fig.add_axes([0.92, 0.10, 0.015, 0.80])
    #     sm = plt.cm.ScalarMappable(cmap=cam_cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    #     cb = fig.colorbar(sm, cax=cax)
    #     cb.ax.tick_params(labelsize=6)
    #     cb.set_label("MSE", fontsize=7)

    #     fig.subplots_adjust(
    #         left=0.05,
    #         right=0.90,
    #         top=0.97,
    #         bottom=0.05,
    #         wspace=0.08,
    #         hspace=0.20,
    #     )
    #     fig.savefig(output_path, dpi=300)
    #     plt.close(fig)

    def generate_report(self):
        """Generate a statistical report of the accuracy distribution."""
        if self.mse_values is None:
            print("No MSE values available yet!")
            return

        print("\n" + "="*50)
        print("MSE ANALYSIS REPORT")
        print("="*50)
        print(f"Number of camera positions: {self.n_cameras}")
        print(f"Hemisphere radius: {self.radius}")
        print(f"\MSE Statistics:")
        print(f"  Mean MSE: {np.mean(self.mse_values):.3f}")
        print(f"  Std deviation: {np.std(self.mse_values):.3f}")
        print(f"  Min MSE: {np.min(self.mse_values):.3f}")
        print(f"  Max MSE: {np.max(self.mse_values):.3f}")

        # Find best camera position
        best_idx = np.argmin(self.mse_values)  # Lower MSE is better
        best_pos = self.camera_positions[best_idx]
        print(f"\nBest camera position:")
        print(f"  Index: {best_idx}")
        print(f"  Position: ({best_pos[0]:.3f}, {best_pos[1]:.3f}, {best_pos[2]:.3f})")
        print(f"  MSE: {self.mse_values[best_idx]:.3f}")

        # Find worst camera position
        worst_idx = np.argmax(self.mse_values)  # Higher MSE is worse
        worst_pos = self.camera_positions[worst_idx]
        print(f"\nWorst camera position:")
        print(f"  Index: {worst_idx}")
        print(f"  Position: ({worst_pos[0]:.3f}, {worst_pos[1]:.3f}, {worst_pos[2]:.3f})")
        print(f"  MSE: {self.mse_values[worst_idx]:.3f}")
        print("="*50 + "\n")


# Main execution
if __name__ == "__main__":
    # Create visualizer
    visualizer = HemisphereHeatmap(n_cameras=100)

    # Generate camera positions (try 'uniform', 'random', or 'clustered')
    visualizer.generate_camera_positions(distribution='uniform')

    # Generate accuracy values (try 'gradient', 'hotspots', 'radial', or 'sinusoidal')
    visualizer.generate_accuracy_values(pattern='hotspots')

    # Generate report
    visualizer.generate_report()

    # Create visualization
    fig = visualizer.plot_hemisphere_heatmap(
        show_cameras=False,
        show_gradient=True,
        colormap='hot_r',  # Try also: 'viridis', 'plasma', 'coolwarm', 'seismic'
        save_path='C:\\Users\\rkhan\\Downloads\\hemisphere_heatmap.png'
    )

    print("Visualization saved to hemisphere_heatmap.png")