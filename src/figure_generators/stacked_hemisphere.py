#!/usr/bin/env python3
"""
Corrected Three-Category Stacked Heatmap Visualization
Creates vertically stacked interpolated heatmaps showing all three categories
on top of each other for each component.
"""

import os
import numpy as np
import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from PIL import Image
import tempfile


class StackedThreeCategoryHeatmap:
    """Create stacked heatmaps with three categories vertically stacked"""

    def __init__(self, base_folder, camera_placements_file, radius=1.2):
        """
        Initialize the processor

        Args:
            base_folder: Path to folder containing all device folders
            camera_placements_file: Path to camera_placements.txt
            radius: Hemisphere radius (default 1.2m)
        """
        self.base_folder = Path(base_folder)
        self.camera_placements_file = camera_placements_file
        self.radius = radius

        # Define categories with their folder patterns
        self.categories = {
            'Simulated': r'VisualProprioception_flow_00dev(\d+)$',      # No R or SR
            'Real': r'VisualProprioception_flow_00devR(\d+)$',          # Has R
            'Sim-to-Real': r'VisualProprioception_flow_00devSR(\d+)$'   # Has SR
        }

        # Define components
        self.components = ['height', 'distance', 'heading',
                          'wrist_angle', 'wrist_rotation', 'gripper']

        # Storage: category -> component -> camera_name -> mse_value
        self.camera_positions = {}
        self.mse_data = {cat: {comp: {} for comp in self.components}
                        for cat in self.categories.keys()}

    def parse_camera_placements(self):
        """Parse camera positions from placements file"""
        print("Parsing camera positions...")

        with open(self.camera_placements_file, 'r') as f:
            lines = f.readlines()

        camera_pattern = r'Position of (camera\d{3}) is: x: ([-\d.e-]+) y: ([-\d.e-]+) z: ([-\d.e-]+)'

        for line in lines:
            match = re.search(camera_pattern, line)
            if match:
                camera_name = match.group(1)
                x = float(match.group(2))
                y = float(match.group(3))
                z = float(match.group(4))

                self.camera_positions[camera_name] = {'x': x, 'y': y, 'z': z}

        print(f"  Found {len(self.camera_positions)} camera positions")
        return self.camera_positions

    def categorize_folder(self, folder_name):
        """
        Determine which category a folder belongs to

        Returns:
            (category_name, camera_number) or (None, None)
        """
        for category, pattern in self.categories.items():
            match = re.match(pattern, folder_name)
            if match:
                camera_num = int(match.group(1))
                return category, camera_num
        return None, None

    def read_all_mse_values(self):
        """Read MSE values from all folders"""
        print("\nScanning for camera folders...")

        if not self.base_folder.exists():
            print(f"  Error: Base folder does not exist: {self.base_folder}")
            return

        # Get all subdirectories
        all_folders = [f for f in self.base_folder.iterdir() if f.is_dir()]

        category_counts = {cat: 0 for cat in self.categories.keys()}

        for folder in all_folders:
            # Determine category and camera number
            category, camera_num = self.categorize_folder(folder.name)

            if category is None:
                continue

            camera_name = f"camera{camera_num:03d}"

            # Check if camera exists in placements
            if camera_name not in self.camera_positions:
                print(f"  Warning: {camera_name} from {folder.name} not in placements")
                continue

            # Build path to MSE file - CORRECTED PATH
            mse_path = (folder / "result" / "visual_proprioception" /
                       "vp_comp_flow_all" / "msecomparison_values.csv")

            if mse_path.exists():
                category_counts[category] += 1

                try:
                    # Read CSV
                    df = pd.read_csv(mse_path)

                    # Debug: print first few rows of first file
                    if category_counts[category] == 1:
                        print(f"\n  Sample CSV from {category} ({folder.name}):")
                        print(f"    Columns: {df.columns.tolist()}")
                        print(f"    First few rows:\n{df.head()}")

                    # Extract MSE values for each component
                    # Try different column access methods
                    for idx, row in df.iterrows():
                        # Try to get component name from first column
                        if 'Title' in df.columns:
                            component = row['Title']
                        elif len(df.columns) > 0:
                            component = row.iloc[0]
                        else:
                            continue

                        if component in self.components:
                            # Try to get MSE value from second column
                            if len(df.columns) > 1:
                                mse_value = float(row.iloc[1])
                                self.mse_data[category][component][camera_name] = mse_value

                except Exception as e:
                    print(f"  Error reading {mse_path}: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"  Missing MSE file: {folder.name} -> {mse_path}")

        print("\n" + "="*70)
        print("MSE FILES FOUND PER CATEGORY:")
        print("="*70)
        for category, count in category_counts.items():
            print(f"  {category}: {count} cameras")
            # Show which cameras have data for first component
            first_comp = self.components[0]
            cameras_with_data = list(self.mse_data[category][first_comp].keys())
            if cameras_with_data:
                print(f"    Cameras: {', '.join(sorted(cameras_with_data))}")
        print("="*70)

    def create_hemisphere_mesh(self, resolution=100):
        """Create hemisphere mesh for interpolation"""
        theta = np.linspace(0, np.pi/2, resolution)
        phi = np.linspace(0, 2*np.pi, resolution)
        THETA, PHI = np.meshgrid(theta, phi)

        X = self.radius * np.sin(THETA) * np.cos(PHI)
        Y = self.radius * np.sin(THETA) * np.sin(PHI)
        Z = self.radius * np.cos(THETA)

        return X, Y, Z

    def interpolate_mse(self, camera_positions_array, mse_values_array, X, Y, Z):
        """Interpolate MSE values across hemisphere"""
        points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        mse_vals = np.asarray(mse_values_array).flatten().reshape(-1, 1)

        rbf = RBFInterpolator(camera_positions_array, mse_vals,
                             smoothing=0.1, kernel='gaussian', epsilon=2.0)
        interpolated = rbf(points)

        return interpolated.flatten().reshape(X.shape)

    def create_stacked_heatmap_for_component(self, component, output_path, resolution=100):
        """
        Create flat 2D stacked circular heatmaps (top-down view) with camera positions

        Args:
            component: Component name (e.g., 'height', 'distance')
            output_path: Path to save the figure
            resolution: Grid resolution
        """
        print(f"\nCreating stacked heatmap for {component}...")

        # Create circular mesh for top-down view
        X, Y, Z = self.create_hemisphere_mesh(resolution=resolution)

        # Prepare data for each category
        category_meshes = []
        category_positions = []
        category_mse_values = []
        category_labels = []
        category_order = ['Simulated', 'Real', 'Sim-to-Real']  # Bottom to top

        for category in category_order:
            camera_mse = self.mse_data[category][component]

            if len(camera_mse) == 0:
                print(f"  Warning: No data for {category} - {component}")
                continue

            positions = []
            mse_values = []

            for camera_name, mse_value in camera_mse.items():
                if camera_name in self.camera_positions:
                    pos = self.camera_positions[camera_name]
                    positions.append([pos['x'], pos['y'], pos['z']])
                    mse_values.append(mse_value)

            if len(positions) == 0:
                print(f"  Warning: No valid positions for {category} - {component}")
                continue

            positions = np.array(positions)
            mse_values = np.array(mse_values)

            # Interpolate MSE across hemisphere
            mse_mesh = self.interpolate_mse(positions, mse_values, X, Y, Z)

            # Apply Gaussian smoothing
            mse_mesh = gaussian_filter(mse_mesh, sigma=1.0)

            category_meshes.append(mse_mesh)
            category_positions.append(positions)
            category_mse_values.append(mse_values)
            category_labels.append(category)

            print(f"  {category}: {len(positions)} cameras, "
                  f"MSE range [{mse_values.min():.4f}, {mse_values.max():.4f}]")

        if len(category_meshes) == 0:
            print(f"  Error: No data available for {component}")
            return

        # Create SQUARE figure
        fig = plt.figure(figsize=(8, 8.8))

        # Fill ENTIRE figure with 3D plot - absolutely NO margins
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection='3d')
        ax.set_proj_type('ortho')
        z_spacing=0.8                    # your spacing (keep whatever you want)
        z_exag=1.2                  # try 2–5 to make gaps obvious
        gap=z_spacing * z_exag
        base = -0.55 * gap                 # shift the whole stack down (tweak 0.45–0.70)
        ax.set_box_aspect((1, 1, z_exag))  # stretch Z relative to X/Y


        # Global normalization across all categories for this component
        all_mesh_values = np.concatenate([mesh.flatten() for mesh in category_meshes])
        vmin = all_mesh_values.min()
        vmax = all_mesh_values.max()

        # Create colormap
        cmap = cm.get_cmap('hot_r')
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Vertical spacing between layers
        z_spacing=0.8
        z_exag=1.2
        gap=z_spacing * z_exag
        base = -0.55 * gap                 # shift the whole stack down (tweak 0.45–0.70)

        # bottom → top lifts (tweak numbers if you like)
        lift_ratios = [0.05, 0.05, 0.05]               # 32%, 26%, 18% of the gap

        # never let a lift exceed half the gap (so it can't touch the next layer)
        max_ratio = 0.49
        lift_ratios = [min(r, max_ratio) for r in lift_ratios]
        all_cam_x, all_cam_y, all_cam_z, all_mse = [], [], [], []


        # Plot each category as a flat circular layer at different heights
        for idx, (mse_mesh, positions, mse_vals, label) in enumerate(
            zip(category_meshes, category_positions, category_mse_values, category_labels)):

            # Height for this layer
            # layer_height = idx * z_spacing

            # Create flat Z array at this height
            # Z_flat = np.full_like(X, layer_height)

            layer_height = idx * z_spacing * z_exag
            Z_flat = np.full_like(X, layer_height)

            alphas = [0.85, 0.85, 0.85]   # bottom, middle, top
            layer_alpha = alphas[idx]

            # Plot flat circular heatmap
            surf = ax.plot_surface(X, Y, Z_flat,
                                  facecolors=cmap(norm(mse_mesh)),
                                  alpha=layer_alpha,
                                  antialiased=True,
                                  linewidth=0.1,
                                  rasterized=True,
                                  shade=False)

            # Plot camera positions as HIGHLY VISIBLE scatter points

            camera_x = positions[:, 0]
            camera_y = positions[:, 1]
            # lift=0.05 * z_exag
            # lift=0.35 * z_spacing * z_exag    # try 0.25–0.45
            lift = lift_ratios[idx] * gap
            if idx<2:
                camera_z = np.full(len(positions), layer_height + lift)
            else:
                camera_z = np.full(len(positions), layer_height + lift)

            all_cam_x.append(camera_x)
            all_cam_y.append(camera_y)
            all_cam_z.append(camera_z)
            all_mse.append(mse_vals)


            # Add THICK circle outline for this layer
            theta_circle = np.linspace(0, 2*np.pi, 100)
            circle_x = self.radius * np.cos(theta_circle)
            circle_y = self.radius * np.sin(theta_circle)
            circle_z = np.full(100, layer_height)
            ax.plot(circle_x, circle_y, circle_z, 'k-', linewidth=4.0, zorder=5)

        all_cam_x = np.concatenate(all_cam_x)
        all_cam_y = np.concatenate(all_cam_y)
        all_cam_z = np.concatenate(all_cam_z)
        all_mse  = np.concatenate(all_mse)

        # Draw camera dots with STRONG white borders for visibility on dark backgrounds
        # Layer 1: Thick black outline (for maximum contrast)
        ax.scatter(all_cam_x, all_cam_y, all_cam_z,
                s=100, c='black', edgecolors='none', linewidths=0,
                marker='o', alpha=1.0, zorder=99, depthshade=False)

        # Layer 2: White background (visible on all colors)
        ax.scatter(all_cam_x, all_cam_y, all_cam_z,
                s=90, c='white', edgecolors='black', linewidths=1.5,
                marker='o', alpha=1.0, zorder=100, depthshade=False)

        # Layer 3: Colored center based on MSE value
        ax.scatter(all_cam_x, all_cam_y, all_cam_z,
                s=40, c=all_mse, cmap=cmap, norm=norm,
                marker='o', alpha=1.0, zorder=101, depthshade=False)



        # COMPLETELY REMOVE ALL AXES ELEMENTS
        ax.set_axis_off()
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Make panes completely invisible
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')

        # Turn off grid
        ax.grid(False)

        # VERY TIGHT axis limits - absolutely minimal extra space
        max_range = self.radius * 1.02  # Almost no extra space
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        # ax.set_zlim([-0.02, len(category_meshes) * z_spacing + 0.08])  # Minimal vertical padding
        ax.set_zlim([-0.02, len(category_meshes) * z_spacing * z_exag + 0.08])


        # Set viewing angle
        ax.view_init(elev=25, azim=45)

        # Add horizontal colorbar RIGHT AT THE BOTTOM - no gap
        # Make it span almost full width
        # cbar_ax = fig.add_axes([0.1, 0.01, 0.8, 0.03])  # Very bottom, wider
        # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        # # cbar.set_label('MSE', fontsize=18, labelpad=5)
        # cbar.ax.tick_params(labelsize=14)



        # 1) Make figure fill space tightly - NO margins at all
        # Remove all padding/spacing completely
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

        # Disable automatic padding
        ax.margins(0, 0, 0)

        # Save to temporary location first
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_path = tmp.name

        fig.savefig(temp_path, dpi=300, pad_inches=0.0, bbox_inches=None)
        plt.close(fig)

        # 2) VERY AGGRESSIVE cropping - remove all white/near-white pixels
        img = Image.open(temp_path).convert("RGB")
        img_array = np.array(img)

        # Find non-white pixels (anything not pure white)
        # Use a more aggressive threshold - even slightly off-white gets included
        threshold = 254  # Very aggressive - only skip nearly pure white

        # Check all three channels - pixel must be white in ALL channels to be cropped
        is_white = (img_array[:, :, 0] >= threshold) & \
                   (img_array[:, :, 1] >= threshold) & \
                   (img_array[:, :, 2] >= threshold)

        # Find rows and columns with non-white content
        non_white_rows = np.where(~is_white.all(axis=1))[0]
        non_white_cols = np.where(~is_white.all(axis=0))[0]

        if len(non_white_rows) > 0 and len(non_white_cols) > 0:
            # Get bounding box with ZERO padding
            top = non_white_rows.min()
            bottom = non_white_rows.max() + 1
            left = non_white_cols.min()
            right = non_white_cols.max() + 1

            # Crop to this tight bounding box
            img_cropped = img.crop((left, top, right, bottom))
        else:
            img_cropped = img

        # Clean up temp file
        os.unlink(temp_path)

        # Save final cropped image
        img_cropped.save(output_path)

        print(f"  ✓ Saved to {output_path}")

        # 3) Now add colorbar as a separate element at the bottom
        # Create a new small figure just for the colorbar
        cbar_fig = plt.figure(figsize=(8, 0.5))
        cbar_ax = cbar_fig.add_axes([0.1, 0.3, 0.8, 0.4])

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')

        # ============ CUSTOMIZE COLORBAR TICKS HERE ============

        # Option 1: Automatic ticks (default - already happens)
        cbar.ax.tick_params(labelsize=30)

        # Option 2: Set specific number of ticks (e.g., 5 ticks)
        # import numpy as np
        # ticks = np.linspace(norm.vmin, norm.vmax, 5)
        # cbar.set_ticks(ticks)
        # cbar.ax.tick_params(labelsize=14)

        # Option 3: Set custom tick positions
        # cbar.set_ticks([0.22, 0.25, 0.28, 0.30, 0.34])  # Your specific values
        # cbar.ax.tick_params(labelsize=14)

        # Option 4: Set ticks with custom labels
        # cbar.set_ticks([0.22, 0.26, 0.30, 0.34])
        # cbar.set_ticklabels(['Low', 'Med', 'High', 'Very High'])
        # cbar.ax.tick_params(labelsize=14)

        # Option 5: More ticks for detail (e.g., 10 ticks)
        # ticks = np.linspace(norm.vmin, norm.vmax, 10)
        # cbar.set_ticks(ticks)
        # cbar.ax.tick_params(labelsize=14)

        # ========================================================

        # Save colorbar to temp
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_cbar_path = tmp.name
        cbar_fig.savefig(temp_cbar_path, dpi=300, bbox_inches='tight', pad_inches=0.05, facecolor='white')
        plt.close(cbar_fig)

        # Load colorbar and crop it
        cbar_img = Image.open(temp_cbar_path).convert("RGB")
        cbar_array = np.array(cbar_img)

        # Crop colorbar aggressively too
        is_white_cbar = (cbar_array[:, :, 0] >= 254) & \
                        (cbar_array[:, :, 1] >= 254) & \
                        (cbar_array[:, :, 2] >= 254)
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

        # Create combined image with minimal gap (just a few pixels)
        gap = 10  # Small gap between plot and colorbar
        combined_height = img_cropped.height + gap + cbar_img_resized.height
        combined = Image.new('RGB', (main_width, combined_height), 'white')

        # Paste main plot and colorbar
        combined.paste(img_cropped, (0, 0))
        combined.paste(cbar_img_resized, (0, img_cropped.height + gap))

        # Save final combined image
        combined.save(output_path)

        print(f"  ✓ Saved combined plot with colorbar to {output_path}")




    def create_all_stacked_heatmaps(self, output_dir):
        """Create both flat and 3D hemisphere stacked heatmaps for all components"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("CREATING STACKED HEATMAPS (Both Flat and 3D Versions)")
        print("="*70)

        for component in self.components:
            # Create flat version (top-down circles)
            flat_file = output_path / f"{component}_stacked_flat_heatmaps.png"
            self.create_stacked_heatmap_for_component(component, flat_file)

            # Create 3D hemisphere version
            # hemisphere_file = output_path / f"{component}_stacked_3d_hemispheres.png"
            # self.create_3d_hemisphere_stacked(component, hemisphere_file)

        print("\n" + "="*70)
        print("STACKED HEATMAPS COMPLETE")
        print("="*70)

    def generate_summary_statistics(self, output_dir):
        """Generate summary statistics"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        summary_data = []

        for category in self.categories.keys():
            for component in self.components:
                camera_mse = self.mse_data[category][component]

                if len(camera_mse) > 0:
                    mse_values = list(camera_mse.values())

                    summary_data.append({
                        'Category': category,
                        'Component': component,
                        'N_Cameras': len(camera_mse),
                        'Mean_MSE': np.mean(mse_values),
                        'Std_MSE': np.std(mse_values),
                        'Min_MSE': np.min(mse_values),
                        'Max_MSE': np.max(mse_values)
                    })

        df = pd.DataFrame(summary_data)
        summary_file = output_path / "summary_statistics.csv"
        df.to_csv(summary_file, index=False)

        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)
        print(f"\nSaved to: {summary_file}")

        return df


def main():


    # ========================================================================
    # UPDATE THESE PATHS
    # ========================================================================
    BASE_FOLDER = "C:\\Users\\rkhan\\Downloads\\sim_real_sim2real_10cams_and_all_sims_only_vvgg"
    CAMERA_PLACEMENTS_FILE = "C:\\Users\\rkhan\\Downloads\\camera_placements.txt"
    OUTPUT_DIR = "C:\\Users\\rkhan\\Downloads\\stacked_heatmaps_sim_tested_on_sim"

    # Expected folder structure in BASE_FOLDER:
    # VisualProprioception_flow_00dev015/
    #   └── result/visual_proprioception/vp_comp_flow_all/all_msecomparison_values.csv
    # VisualProprioception_flow_00dev039/
    #   └── result/visual_proprioception/vp_comp_flow_all/all_msecomparison_values.csv
    # VisualProprioception_flow_00devR015/
    #   └── result/visual_proprioception/vp_comp_flow_all/all_msecomparison_values.csv
    # VisualProprioception_flow_00devR039/
    #   └── result/visual_proprioception/vp_comp_flow_all/all_msecomparison_values.csv
    # VisualProprioception_flow_00devSR015/
    #   └── result/visual_proprioception/vp_comp_flow_all/all_msecomparison_values.csv
    # VisualProprioception_flow_00devSR039/
    #   └── result/visual_proprioception/vp_comp_flow_all/all_msecomparison_values.csv
    # etc. (10 of each type)

    print("\n" + "="*70)
    print("STACKED THREE-CATEGORY HEATMAP VISUALIZATION")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Base folder: {BASE_FOLDER}")
    print(f"  Camera placements: {CAMERA_PLACEMENTS_FILE}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("\nExpected:")
    print("  - 10 Simulated cameras (format: dev###)")
    print("  - 10 Real cameras (format: devR###)")
    print("  - 10 Sim-to-Real cameras (format: devSR###)")

    # Create processor
    processor = StackedThreeCategoryHeatmap(BASE_FOLDER, CAMERA_PLACEMENTS_FILE)

    # Parse camera positions
    processor.parse_camera_placements()

    # Read MSE values
    processor.read_all_mse_values()

    # Generate outputs
    processor.generate_summary_statistics(OUTPUT_DIR)
    processor.create_all_stacked_heatmaps(OUTPUT_DIR)

    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nGenerated files in {OUTPUT_DIR}:")
    print("\nFlat circular versions (with cameras):")
    print("  - height_stacked_flat_heatmaps.png")
    print("  - distance_stacked_flat_heatmaps.png")
    print("  - heading_stacked_flat_heatmaps.png")
    print("  - wrist_angle_stacked_flat_heatmaps.png")
    print("  - wrist_rotation_stacked_flat_heatmaps.png")
    print("  - gripper_stacked_flat_heatmaps.png")
    print("\n3D hemisphere versions (with cameras):")
    print("  - height_stacked_3d_hemispheres.png")
    print("  - distance_stacked_3d_hemispheres.png")
    print("  - heading_stacked_3d_hemispheres.png")
    print("  - wrist_angle_stacked_3d_hemispheres.png")
    print("  - wrist_rotation_stacked_3d_hemispheres.png")
    print("  - gripper_stacked_3d_hemispheres.png")
    print("\nOther files:")
    print("  - summary_statistics.csv")
    print("\nEach heatmap shows 3 vertically stacked layers:")
    print("  Bottom: Simulated cameras")
    print("  Middle: Real cameras")
    print("  Top: Sim-to-Real cameras")


if __name__ == "__main__":
    main()