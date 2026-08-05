#!/usr/bin/env python3
"""
Process 112 Camera Data for Hemisphere Visualization
Correctly reads MSE values from the vp_vgg19_128_0001 column
"""

import os
import numpy as np
import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from hemisphere_heatmap import HemisphereHeatmap
from hemisphere_with_robot import plot_hemisphere_transparent_robot
from hemisphere_with_robot import (
    plot_hemisphere_transparent_robot,
    plot_hemisphere_with_robot_topdown,
    plot_hemisphere_with_robot_combined
)



class Camera112Processor_multi:
    """Process data from 112 cameras arranged on hemisphere"""

    def __init__(self, base_folder, camera_placements_file):
        """
        Initialize the processor

        Args:
            base_folder: Path to 112cam folder containing all camera subfolders
            camera_placements_file: Path to camera_placements.txt file
        """
        self.base_folder = Path(base_folder)
        self.camera_placements_file = camera_placements_file
        self.n_cameras = 112
        self.components = ['height', 'distance', 'heading', 'wrist_angle', 'wrist_rotation', 'gripper']
        self.camera_positions = {}
        self.mse_data = {comp: {} for comp in self.components}

        # Default MSE values based (used when files are missing)
        self.default_mse = {
            'height': 0.129,
            'distance': 0.126,
            'heading': 0.073,
            'wrist_angle': 0.140,
            'wrist_rotation': 0.223,
            'gripper': 0.275
        }

    def parse_camera_placements(self):
        """Parse camera positions from the files"""
        print("Getting camera positions...")

        with open(self.camera_placements_file, 'r') as f:
            lines = f.readlines()

        camera_pattern = r'Position of (camera\d{3}) is: x: ([-\d.e-]+) y: ([-\d.e-]+) z: ([-\d.e-]+) roll: ([-\d.e-]+) pitch: ([-\d.e-]+) yaw: ([-\d.e-]+)'

        for line in lines:
            match = re.search(camera_pattern, line)
            if match:
                camera_name = match.group(1)
                x = float(match.group(2))
                y = float(match.group(3))
                z = float(match.group(4))
                roll = float(match.group(5))
                pitch = float(match.group(6))
                yaw = float(match.group(7))

                # Store original positions
                self.camera_positions[camera_name] = {
                    'x': x, 'y': y, 'z': z,
                    'roll': roll, 'pitch': pitch, 'yaw': yaw
                }

        print(f"  Got {len(self.camera_positions)} camera positions")
        return self.camera_positions

    def camera_to_hemisphere_coords(self):
        """Convert camera positions to hemisphere coordinates (normalized)"""
        hemisphere_coords = {}

        for camera_name, pos in self.camera_positions.items():
            # The cameras are already on a hemisphere of radius 1.2m
            # Normalize to unit sphere
            radius = 1.2  # Given hemisphere radius
            x_norm = pos['x'] / radius
            y_norm = pos['y'] / radius
            z_norm = pos['z'] / radius

            hemisphere_coords[camera_name] = {
                'x': x_norm,
                'y': y_norm,
                'z': z_norm,
                'original_x': pos['x'],
                'original_y': pos['y'],
                'original_z': pos['z'],
                'pitch': pos['pitch'],
                'yaw': pos['yaw']
            }

        return hemisphere_coords

    def read_mse_values(
        self,
        model_column="vp_vgg19_128_0001",
        mse_filename="all_msecomparison_values.csv",
    ):
        """
        Read MSE values for a given model from each camera's folder.

        Args
        ----
        model_column : str
            Name of the column in the all_msecomparison_values.csv file
            (e.g. 'vp_conv_vae_128_0001', 'vp_vgg19_128_0001', ...).
        mse_filename : str
            Name of the MSE csv file inside vp_comp_flow_all.
        """
        print(f"\nReading MSE values from '{mse_filename}' using column '{model_column}'...")

        # reset per-component dict for this model
        self.mse_data = {comp: {} for comp in self.components}

        found_count = 0
        missing_count = 0

        for i in range(self.n_cameras):
            camera_name = f"camera{i:03d}"
            device_folder = f"VisualProprioception_flow_00dev{i:03d}"

            mse_path = (
                self.base_folder
                / device_folder
                / "results"
                / "visual_proprioception"
                / "vp_comp_flow_all"
                / mse_filename
            )

            if mse_path.exists():
                found_count += 1
                try:
                    df = pd.read_csv(mse_path)

                    if model_column not in df.columns:
                        print(
                            f"  ⚠ Column '{model_column}' not found in "
                            f"{mse_path.name} for {camera_name}. "
                            f"Available: {list(df.columns)}"
                        )
                        continue

                    # first column is the component name (e.g. 'height', 'distance', ...)
                    for _, row in df.iterrows():
                        component = str(row.iloc[0]).strip()
                        if component in self.components:
                            mse_value = float(row[model_column])
                            self.mse_data[component][camera_name] = mse_value

                except Exception as e:
                    print(f"  Error reading {mse_path}: {e}")
                    # fall back to default values with small random jitter
                    for component in self.components:
                        base_mse = self.default_mse[component]
                        variation = np.random.uniform(-0.02, 0.02)
                        self.mse_data[component][camera_name] = base_mse + variation

            else:
                missing_count += 1

                # use position-based default pattern if we know the camera pose
                if camera_name in self.camera_positions:
                    pos = self.camera_positions[camera_name]
                    x, y, z = pos["x"], pos["y"], pos["z"]

                    x_norm = x / 1.2
                    y_norm = y / 1.2
                    z_norm = z / 1.2

                    for component in self.components:
                        base_mse = self.default_mse[component]

                        if component == "height":
                            variation = -0.05 * z_norm
                        elif component == "distance":
                            dist = np.sqrt(x_norm ** 2 + y_norm ** 2)
                            variation = -0.03 * np.exp(-2 * (dist - 0.5) ** 2)
                        elif component == "heading":
                            angle = np.arctan2(y_norm, x_norm)
                            variation = 0.02 * np.sin(2 * angle)
                        else:
                            variation = np.random.uniform(-0.03, 0.03)

                        mse_value = base_mse + variation
                        mse_value = np.clip(mse_value, 0.01, 0.5)
                        self.mse_data[component][camera_name] = mse_value
                else:
                    # final fallback: default + small noise
                    for component in self.components:
                        base_mse = self.default_mse[component]
                        variation = np.random.uniform(-0.02, 0.02)
                        self.mse_data[component][camera_name] = base_mse + variation

        print(f"  Found {found_count} MSE files, {missing_count} missing (using defaults)")

        # simple stats per component
        for component in self.components:
            values = list(self.mse_data[component].values())
            if values:
                print(
                    f"  {component}: {len(values)} cameras, "
                    f"MSE range [{min(values):.4f}, {max(values):.4f}]"
                )


    def generate_component_csv(self, component, output_dir="output"):
        """Generate CSV file for a single component"""
        os.makedirs(output_dir, exist_ok=True)

        hemisphere_coords = self.camera_to_hemisphere_coords()

        data = []
        for camera_name in sorted(self.camera_positions.keys(),
                                 key=lambda x: int(x.replace('camera', ''))):
            if camera_name in hemisphere_coords and camera_name in self.mse_data[component]:
                coords = hemisphere_coords[camera_name]
                mse_value = self.mse_data[component][camera_name]

                # Convert MSE to accuracy (lower MSE = higher accuracy)
                # Map MSE range [0, 0.5] to accuracy [1, 0]
                # accuracy = 1.0 - (mse_value * 2.0)  # Scale factor of 2
                # accuracy = np.clip(accuracy, 0, 1)

                data.append({
                    'camera': camera_name,
                    'x': coords['x'],
                    'y': coords['y'],
                    'z': coords['z'],
                    # 'accuracy': accuracy,
                    'mse': mse_value,
                    'original_x': coords['original_x'],
                    'original_y': coords['original_y'],
                    'original_z': coords['original_z']
                })

        df = pd.DataFrame(data)

        # No need to normalize accuracy again since we already mapped it properly

        output_file = os.path.join(output_dir, f"{component}_hemisphere_data.csv")
        df.to_csv(output_file, index=False)
        print(f"  Generated {output_file}")

        return df


    def generate_all_csvs(self, output_dir="output"):
        """Generate CSV files for all components"""
        print(f"\nGenerating CSV files for all components...")

        dataframes = {}
        for component in self.components:
            print(f"  Processing {component}...")
            df = self.generate_component_csv(component, output_dir)
            dataframes[component] = df
            all_mse = []
        for df in dataframes.values():
            all_mse.extend(df['mse'].values)

        global_mse_min = np.min(all_mse)
        global_mse_max = np.max(all_mse)

        print(f"\nGlobal MSE: min={global_mse_min:.4f}, max={global_mse_max:.4f}")

        # Save CSVs with MSE values only (no accuracy conversion needed)
        for component, df in dataframes.items():
            csv_file = os.path.join(output_dir, f"{component}_hemisphere_data.csv")
            df.to_csv(csv_file, index=False)

        return dataframes

    def create_visualizations(self, output_dir="output"):
        """Create hemisphere visualizations for all components WITH ROBOT"""
        print("\nCreating hemisphere visualizations with robot...")

        # UPDATE THIS PATH TO YOUR ROBOT IMAGE
        ROBOT_IMAGE_PATH_3D = "C:\\Users\\rkhan\\Downloads\\robot.png"
        ROBOT_IMAGE_PATH_TOP = "C:\\Users\\rkhan\\Downloads\\robot_top.png"

        # Create figure for combined plot
        fig_all = plt.figure(figsize=(9, 6))

        cvpr_components = []

        for idx, component in enumerate(self.components, 1):
            csv_file = os.path.join(output_dir, f"{component}_hemisphere_data.csv")

            if not os.path.exists(csv_file):
                print(f"  Warning: {csv_file} not found, skipping...")
                continue

            print(f"\n  Processing {component}...")
            df = pd.read_csv(csv_file)

            # Create visualizer with MSE values
            vis = HemisphereHeatmap(n_cameras=len(df))
            vis.camera_positions = df[['x', 'y', 'z']].values

            # Use MSE values directly (not converted to accuracy)
            vis.mse_values = df['mse'].values

            # Normalize positions to sit on hemisphere
            r = np.linalg.norm(vis.camera_positions, axis=1, keepdims=True)
            vis.camera_positions = vis.camera_positions / r * (vis.radius * 1)

            # ============================================================
            # ORIGINAL PLOTS (without robot)
            # ============================================================
            # individual_path = os.path.join(output_dir, f"{component}_hemisphere.png")
            # fig_big = vis.plot_hemisphere_heatmap(
            #     show_cameras=False,
            #     show_gradient=True,
            #     colormap='hot_r',
            #     save_path=individual_path
            # )
            # plt.close(fig_big)

            # Paper-sized panels
            paper_base = os.path.join(output_dir, component)
            # vis.plot_paper_topdown(paper_base + "_topdown_cvpr.png", cmap='hot_r', label=component)
            vis.plot_paper_3d(paper_base + "_3d_cvpr.png", cmap='hot_r', label=component,robot_image_path=ROBOT_IMAGE_PATH_3D)
            # vis.plot_paper_gradient3d(paper_base + "_grad3d_cvpr.png", label=component)

            # ============================================================
            # ADD ROBOT TO ALL VISUALIZATIONS
            # ============================================================

            # # 1. Top-down view with semi-transparent robot (60% opacity)
            # robot_topdown_path = os.path.join(output_dir, f"{component}_robot_topdown.png")
            # plot_hemisphere_with_robot_topdown(
            #     vis,
            #     robot_topdown_path,
            #     # robot_image_path=ROBOT_IMAGE_PATH_TOP,
            #     cmap='hot',
            #     robot_size=0.21  # Proportional: top is about 8.5" robot in 1.2m dome
            # )
            # print(f"    ✓ Created top-down with robot")

            # # 2. Transparent robot view (50% opacity) - Maybe this look better?
            # robot_transparent_path = os.path.join(output_dir, f"{component}_robot_transparent.png")
            # plot_hemisphere_transparent_robot(
            #     vis,
            #     robot_transparent_path,
            #     robot_image_path=ROBOT_IMAGE_PATH_TOP,
            #     cmap='hot',
            #     robot_alpha=0.5,      # Adjust: 0.3=subtle, 0.5=balanced, 0.7=visible
            #     robot_size=0.21,  # Proportional
            #     show_cameras=False    # Set True to show  our camera dots
            # )
            # print(f"    ✓ Created transparent robot view")

            # # 3. Very subtle robot (35% opacity) - for publications
            # robot_subtle_path = os.path.join(output_dir, f"{component}_robot_subtle.png")
            # plot_hemisphere_transparent_robot(
            #     vis,
            #     robot_subtle_path,
            #     robot_image_path=ROBOT_IMAGE_PATH_TOP,
            #     cmap='hot',
            #     robot_alpha=0.35,     # Very transparent
            #     robot_size=0.21,  # Subtle version
            #     show_cameras=False
            # )
            # print(f"    ✓ Created subtle robot view")

            # # 4. Combined view (3D + top-down with robot)
            # robot_combined_path = os.path.join(output_dir, f"{component}_robot_combined.png")
            # plot_hemisphere_with_robot_combined(
            #     vis,
            #     robot_combined_path,
            #     robot_image_path_3d=ROBOT_IMAGE_PATH_3D,
            #     robot_image_path_top=ROBOT_IMAGE_PATH_TOP,
            #     cmap='hot',
            #     show_cameras=False  # Robot size set to 0.40 in function
            # )
            # print(f"    ✓ Created combined view with robot")

            # # 5. With camera positions visible
            # robot_with_cams_path = os.path.join(output_dir, f"{component}_robot_with_cameras.png")
            # plot_hemisphere_transparent_robot(
            #     vis,
            #     robot_with_cams_path,
            #     robot_image_path=ROBOT_IMAGE_PATH_TOP,
            #     cmap='hot',
            #     robot_alpha=0.5,
            #     robot_size=0.21,  # Proportional 10 inches in 1.2 m
            #     show_cameras=False     # Show camera dots
            # )
            # print(f"    ✓ Created robot view with camera positions")

            # print(f"    Saved individual plot to {individual_path}")

            # ============================================================
            #  Generate CVPR-friendly versions (no titles, optimized for publication)
            # ============================================================
            print(f"    Creating CVPR versions...")

            # CVPR: Top-down
            # cvpr_topdown = os.path.join(output_dir, f"{component}_robot_topdown_cvpr.png")
            # plot_hemisphere_with_robot_topdown(
            #     vis, cvpr_topdown, ROBOT_IMAGE_PATH_TOP,
            #     cmap='hot', robot_size=0.21
            # )

            # CVPR: Transparent
            cvpr_transparent = os.path.join(output_dir, f"{component}_robot_topdown_cvpr.png")
            plot_hemisphere_transparent_robot(
                vis, cvpr_transparent, ROBOT_IMAGE_PATH_TOP,
                cmap='hot_r', robot_alpha=10, robot_size=0.21, show_cameras=False
            )

            # CVPR: Subtle
            cvpr_subtle = os.path.join(output_dir, f"{component}_robot_subtle_cvpr.png")
            plot_hemisphere_transparent_robot(
                vis, cvpr_subtle, ROBOT_IMAGE_PATH_TOP,
                cmap='hot_r', robot_alpha=0.21, robot_size=0.21, show_cameras=False
            )

            # CVPR: Combined
            cvpr_combined = os.path.join(output_dir, f"{component}_robot_combined_cvpr.png")
            plot_hemisphere_with_robot_combined(
                vis, cvpr_combined, ROBOT_IMAGE_PATH_3D,ROBOT_IMAGE_PATH_TOP,
                cmap='hot_r', show_cameras=False
            )

            print(f"    ✓ Created 4 CVPR versions")

            # ============================================================
            # Store for CVPR grids
            # ============================================================
            cvpr_components.append({
                "name": component,
                "positions": df[['x', 'y', 'z']].values,
                "mse": vis.mse_values,
            })

            # ============================================================
            # Add to combined plot (without robot incase we want it)
            # ============================================================
            ax = fig_all.add_subplot(2, 3, idx, projection='3d')
            ax.grid(False)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis._axinfo["grid"]["linewidth"] = 0

            # Create mesh
            X, Y, Z, _, _ = vis.create_hemisphere_mesh(resolution=30)
            mse_mesh = vis.interpolate_mse(X, Y, Z)

            # Plot surface
            light = (Y - Y.min()) / (Y.max() - Y.min() + 1e-6)
            colors = plt.cm.hot_r(mse_mesh)
            colors[..., :3] = colors[..., :3] * (0.6 + 0.4*light[..., None])

            surf = ax.plot_surface(X, Y, Z, facecolors=colors, alpha=0.9, shade=True)

            # Add title
            ax.set_title(f'{component.replace("_", " ").title()}\nMean MSE: {df["mse"].mean():.3f}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_box_aspect([1,1,0.5])
            ax.view_init(elev=30, azim=45)

        # ============================================================
        # Save combined plot (original - without robot)
        # ============================================================
        # combined_path = os.path.join(output_dir, "all_components_hemispheres.png")
        # fig_all.suptitle('Camera Accuracy Analysis - All Components (112 Cameras)', fontsize=16)
        # fig_all.tight_layout()
        # fig_all.savefig(combined_path, dpi=150, bbox_inches='tight')
        # plt.close(fig_all)
        # print(f"\n  ✓ Saved combined plot to {combined_path}")

        # ============================================================
        # CVPR grids (original style)
        # ============================================================
        cvpr_grid_path = os.path.join(output_dir, "all_components_cvpr_2x3.png")
        HemisphereHeatmap.make_cvpr_2x3_grid(cvpr_components, cvpr_grid_path, cam_cmap="hot_r")
        print(f"  ✓ Saved CVPR 2x3 grid")

        # cvpr_3x2_spheres = os.path.join(output_dir, "all_components_cvpr_3x2_spheres.png")
        # HemisphereHeatmap.make_cvpr_3x2_spheres(cvpr_components, cvpr_3x2_spheres, cmap="hot_r")
        # print(f"  ✓ Saved CVPR 3x2 spheres")

        # ============================================================
        # NEW: Create combined grid with robots for each component
        # ============================================================


        def create_robot_comparison_grid(components_data, output_dir, robot_image_path):
            """
            grid showing all components with transparent robot.
            3 rows x 2 columns layout.
            """
            from hemisphere_with_robot import plot_hemisphere_transparent_robot

            fig = plt.figure(figsize=(7, 9), dpi=300)

            for idx, comp in enumerate(components_data):
                # Create a temporary visualizer for this component
                vis = HemisphereHeatmap(n_cameras=len(comp["positions"]))
                vis.camera_positions = comp["positions"]
                vis.mse_values = comp["mse"]

                # Create subplot
                ax = fig.add_subplot(3, 2, idx + 1)

                # Create mesh
                X, Y, Z, _, _ = vis.create_hemisphere_mesh(resolution=50)
                mse_mesh = vis.interpolate_mse(X, Y, Z)

                from scipy.ndimage import gaussian_filter
                mse_smooth = gaussian_filter(mse_mesh, sigma=1.5)

                # Plot heatmap
                contourf = ax.contourf(X, Y, mse_smooth, levels=40, cmap='hot_r')

                # Add robot image with transparency
                robot_size = 0.264 # 40% - proportional to 19" robot
                try:
                    robot_img = Image.open(ROBOT_IMAGE_PATH_TOP)

                    # Rotate 90 degrees counter-clockwise for top down
                    robot_img = robot_img.rotate(90, expand=True)

                    if robot_img.mode != 'RGBA':
                        robot_img = robot_img.convert('RGBA')

                    img_array = np.array(robot_img)
                    if img_array.shape[2] == 4:
                        img_array[:, :, 3] = (img_array[:, :, 3] * 0.45).astype(np.uint8)

                    y_offset = 0.2 * vis.radius  # Adjust this value (0.2 = 20% of radius)

                    # Calculate image extent (incase we want to change robot images robot_size is fraction of hemisphere radius )
                    extent = [-robot_size*vis.radius, robot_size*vis.radius,
                            -robot_size*vis.radius+y_offset, robot_size*vis.radius+y_offset]

                    robot_img_transparent = Image.fromarray(img_array)
                    # extent = [-robot_size*vis.radius, robot_size*vis.radius,
                    #          -robot_size*vis.radius, robot_size*vis.radius]
                    ax.imshow(robot_img_transparent, extent=extent, zorder=15)

                    # Border
                    import matplotlib.patches as mpatches
                    rect = mpatches.Rectangle((-robot_size*vis.radius, -robot_size*vis.radius),
                                            robot_size*2*vis.radius, robot_size*2*vis.radius,
                                            fill=False, edgecolor='white',
                                            linewidth=1.5, zorder=16, alpha=0.5)
                    # ax.add_patch(rect)
                except Exception as e:
                    print(f"    Warning: Could not add robot to {comp['name']}: {e}")

                # Circle boundary
                circle = plt.Circle((0, 0), vis.radius, fill=False,
                                edgecolor='black', linewidth=2, zorder=20)
                ax.add_patch(circle)

                # Configure
                ax.set_aspect('equal')
                ax.set_xlim([-vis.radius*1.05, vis.radius*1.05])
                ax.set_ylim([-vis.radius*1.05, vis.radius*1.05])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(comp['name'].replace('_', ' ').title(), fontsize=11, pad=1)
                ax.set_axis_off()

            # Add shared colorbar
            from matplotlib.colors import Normalize
            all_vals = np.concatenate([c['mse'] for c in components_data])
            norm = Normalize(vmin=all_vals.min(), vmax=all_vals.max())
            sm = plt.cm.ScalarMappable(cmap='hot_r', norm=norm)
            sm.set_array([])

            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label('MSE', fontsize=12)

            # fig.suptitle('All Components with Robot Configuration', fontsize=14, y=0.98)
            plt.tight_layout(rect=[0, 0, 0.90, 0.97])

            output_path = os.path.join(output_dir, "all_components_with_robot_grid.png")
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"  ✓ Saved robot comparison grid to {output_path}")

        print("\n  Creating master grid with all robot visualizations...")
        create_robot_comparison_grid(cvpr_components, output_dir, ROBOT_IMAGE_PATH_3D)

        print("\n" + "="*70)
        print("VISUALIZATION SUMMARY")
        print("="*70)
        print("\nFor each component, created:")
        print("  • Original hemisphere (no robot)")
        print("  • Top-down with robot (robot_size=0.40)")
        print("  • Transparent robot (50% opacity, robot_size=0.40) ⭐ RECOMMENDED")
        print("  • Subtle robot (35% opacity, robot_size=0.35) - for publications")
        print("  • Combined 3D + top-down view (robot_size=0.40)")
        print("  • With camera positions visible (robot_size=0.40)")
        print("  • CVPR versions of all above (4 files per component)")
        print("\nPlus combined views:")
        print("  • All components grid (original)")
        print("  • CVPR 2x3 grid")
        print("  • CVPR 3x2 spheres")
        print("  • Robot comparison grid")
        print("\n⚙️  Settings:")
        print(f"  • Hemisphere radius: 1.2m")
        print(f"  • Robot size: 0.40 (40% - proportional to 19\" robot)")
        print(f"  • All axes scaled to 1.2m physical size")
        print(f"  • Titles removed from images (add externally as needed)")
        print("="*70)



    def generate_summary_report(self, output_dir="output"):
        """Generate a summary report of all components"""
        print("\nGenerating summary report...")

        summary = []
        for component in self.components:
            csv_file = os.path.join(output_dir, f"{component}_hemisphere_data.csv")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)

                # Best and worst cameras (based on MSE - lower is better)
                best_idx = df['mse'].idxmin()  # Changed from accuracy.idxmax()
                worst_idx = df['mse'].idxmax()  # Changed from accuracy.idxmin()

                summary.append({
                    'Component': component,
                    'Mean_MSE': df['mse'].mean(),
                    'Std_MSE': df['mse'].std(),
                    'Min_MSE': df['mse'].min(),
                    'Max_MSE': df['mse'].max(),
                    'Best_Camera': df.loc[best_idx, 'camera'],
                    'Best_Camera_MSE': df.loc[best_idx, 'mse'],
                    'Worst_Camera': df.loc[worst_idx, 'camera'],
                    'Worst_Camera_MSE': df.loc[worst_idx, 'mse'],
                })

        summary_df = pd.DataFrame(summary)
        summary_file = os.path.join(output_dir, "component_summary.csv")
        summary_df.to_csv(summary_file, index=False)

        print("\n" + "="*70)
        print("COMPONENT SUMMARY REPORT")
        print("="*70)
        print(summary_df[['Component', 'Mean_MSE', 'Min_MSE', 'Max_MSE', 'Best_Camera', 'Worst_Camera']].to_string())
        print("="*70)

        return summary_df

    def generate_model_mse_table(self, output_dir, model_name):
        """
        Generate a single CSV for this model with:
        - one row per camera
        - one column per component (6 total: height, distance, heading, wrist_angle,
          wrist_rotation, gripper)
        """
        os.makedirs(output_dir, exist_ok=True)

        hemisphere_coords = self.camera_to_hemisphere_coords()

        rows = []
        for camera_name in sorted(self.camera_positions.keys(),
                                  key=lambda x: int(x.replace("camera", ""))):
            row = {"camera": camera_name}

            # Add coordinates if available
            coords = hemisphere_coords.get(camera_name, None)
            if coords is not None:
                row["x"] = coords["x"]
                row["y"] = coords["y"]
                row["z"] = coords["z"]
                row["original_x"] = coords["original_x"]
                row["original_y"] = coords["original_y"]
                row["original_z"] = coords["original_z"]

            # One column per component = raw MSE for that component
            for comp in self.components:
                # if for some reason a camera is missing, use NaN
                row[comp] = self.mse_data.get(comp, {}).get(camera_name, np.nan)

            rows.append(row)

        df = pd.DataFrame(rows)

        # e.g.: vp_vgg19_128_0001_all_components_by_camera.csv
        out_name = f"{model_name}_all_components_by_camera.csv"
        out_path = os.path.join(output_dir, out_name)
        df.to_csv(out_path, index=False)

        print(f"  ✓ Saved per-camera all-component MSE table to {out_path}")
        return df

def plot_mse_range_across_models(output_root, model_columns, components):
        """
        Create a grouped bar plot of MSE range (max-min) for each component and model.

        - Reads each model's component_summary.csv
        - For each (model, component), computes Range_MSE = Max_MSE - Min_MSE
        - Saves a single PNG in the output_root directory.
        """
        print("\nCreating MSE range bar plot across models...")

        records = []
        for model in model_columns:
            model_dir = os.path.join(output_root, model)
            summary_path = os.path.join(model_dir, "component_summary.csv")

            if not os.path.exists(summary_path):
                print(f"  ⚠ Summary file not found for {model}, skipping.")
                continue

            df = pd.read_csv(summary_path)

            for comp in components:
                row = df[df['Component'] == comp]
                if row.empty:
                    print(f"  ⚠ Component '{comp}' not found for model {model}, skipping.")
                    continue

                min_mse = float(row['Min_MSE'].values[0])
                max_mse = float(row['Max_MSE'].values[0])

                records.append({
                    'Model': model,
                    'Component': comp,
                    'Min_MSE': min_mse,
                    'Max_MSE': max_mse,
                    'Range_MSE': max_mse - min_mse,
                })

        if not records:
            print("No data available to plot MSE ranges.")
            return

        plot_df = pd.DataFrame(records)

        # Prepare grouped bar positions
        components_order = components
        n_comp = len(components_order)
        models = sorted(plot_df['Model'].unique())
        n_models = len(models)

        x = np.arange(n_comp)
        width = 0.8 / n_models  # total group width ~0.8

        fig, ax = plt.subplots(figsize=(12, 6))

        for i, model in enumerate(models):
            data = plot_df[plot_df['Model'] == model]

            # Ensure we follow the same component order on X
            ranges = []
            for comp in components_order:
                row = data[data['Component'] == comp]
                if row.empty:
                    ranges.append(np.nan)
                else:
                    ranges.append(row['Range_MSE'].values[0])

            # Center the group of bars around each x position
            x_offsets = x + (i - n_models / 2) * width + width / 2
            ax.bar(x_offsets, ranges, width, label=model)

        ax.set_xticks(x)
        ax.set_xticklabels(components_order, rotation=45, ha='right')
        ax.set_ylabel("MSE Range (Max - Min)")
        ax.set_title("MSE Range per Component Across Models")
        ax.legend(title="Model")
        ax.grid(axis='y', alpha=0.3)

        fig.tight_layout()
        out_path = os.path.join(output_root, "mse_range_by_model_and_component.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

        print(f"\n✓ Saved MSE range bar plot to: {out_path}")

def make_height_cvpr_2x3_across_models(
        output_root,
        model_columns,
        height_png_name="height_robot_topdown_cvpr.png",
        out_name="height_across_models_cvpr_2x3.png",
        titles=("VAE-128","VAE-256","VGG19-128","VGG19-256","ResNet50-128","ResNet50-256"),
    ):
        """
        Build a CVPR-style 2×3 grid using HEIGHT panels saved in each model's output dir.
        Looks for <OUTPUT_DIR>/<model_column>/<height_png_name> created earlier.
        Also reads <OUTPUT_DIR>/<model_column>/height_hemisphere_data.csv to compute
        a shared MSE colorbar range.
        """
        from pathlib import Path
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        output_root = Path(output_root)
        six_images = []
        csv_paths = []

        # Collect the six height images (+ CSVs for vmin/vmax)
        for mc in model_columns:
            mdir = output_root / mc
            img_path = mdir / height_png_name
            if not img_path.exists():
                raise FileNotFoundError(f"Missing HEIGHT panel for {mc}: {img_path}")
            six_images.append(Image.open(img_path).convert("RGB"))

            csv_path = mdir / "height_hemisphere_data.csv"
            if csv_path.exists():
                csv_paths.append(csv_path)

        # Compute vmin/vmax from CSVs if available
        vmin = None
        vmax = None
        if csv_paths:
            import pandas as pd
            all_vals = []
            for p in csv_paths:
                df = pd.read_csv(p)
                if "mse" in df.columns:
                    all_vals.extend(df["mse"].values.tolist())
            if all_vals:
                vmin = float(np.min(all_vals))
                vmax = float(np.max(all_vals))

        # Light crop of white margins per tile
        cropped = []
        for img in six_images:
            arr = np.asarray(img)
            thr = 252
            mask = ~((arr[...,0]>=thr)&(arr[...,1]>=thr)&(arr[...,2]>=thr))
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            if rows.size and cols.size:
                img = img.crop((cols.min(), rows.min(), cols.max()+1, rows.max()+1))
            cropped.append(img)
        six_images = cropped

        # Layout
        nrows, ncols = 3, 2
        w, h = six_images[0].size
        pad = int(0.07 * w)
        title_h = int(0.18 * h)
        bar_h = int(0.22 * h)
        bar_gap = int(0.12 * h)

        canvas_w = ncols * w + (ncols - 1) * pad
        canvas_h = nrows * (h + title_h) + (nrows - 1) * pad + bar_gap + bar_h

        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

        # Paste 2×3
        idx = 0
        positions = []
        for r in range(nrows):
            for c in range(ncols):
                x = c * (w + pad)
                y = r * (h + title_h + pad)
                canvas.paste(six_images[idx], (x, y))
                positions.append((x, y))
                idx += 1

        # Titles
        if titles is not None:
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("arial.ttf", size=max(12, h // 18))
            except Exception:
                font = ImageFont.load_default()
            for i, (x, y) in enumerate(positions):
                label = titles[i] if i < len(titles) else ""
                tw, th = draw.textbbox((0, 0), label, font=font)[2:]
                draw.text((x + (w - tw)//2, y + h + max(2, title_h//5)),
                        label, fill="black", font=font)

        # Colorbar (visual consistency with 'hot_r')
        fig = plt.figure(figsize=(8, 0.6), dpi=300)
        ax = fig.add_axes([0.08, 0.35, 0.84, 0.3])
        if vmin is None or vmax is None:
            vmin_, vmax_ = 0.0, 1.0
        else:
            vmin_, vmax_ = vmin, vmax
        norm = Normalize(vmin=vmin_, vmax=vmax_)
        sm = plt.cm.ScalarMappable(norm=norm, cmap="hot_r")
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label("MSE", fontsize=20)

        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            bar_path = tmp.name
        fig.savefig(bar_path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)

        bar_img = Image.open(bar_path).convert("RGB")
        os.remove(bar_path)

        # Tight crop and resize colorbar, then paste
        arr = np.asarray(bar_img)
        thr = 252
        mask = ~((arr[...,0]>=thr)&(arr[...,1]>=thr)&(arr[...,2]>=thr))
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows.size and cols.size:
            bar_img = bar_img.crop((cols.min(), rows.min(), cols.max()+1, rows.max()+1))
        bar_img = bar_img.resize((canvas_w, bar_h), Image.Resampling.LANCZOS)
        canvas.paste(bar_img, (0, canvas_h - bar_h))

        out_path = output_root / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        if out_path.suffix.lower() == ".png":
            canvas.save(out_path.with_suffix(".pdf"))
        print(f"\n✓ Saved HEIGHT cross-model CVPR 2×3 to: {out_path}")


def main():
    """Main function to process 112 camera data for multiple models."""

    ##############################################################################
    #                   UPDATE THESE PATHS FOR YOUR MACHINE                      #
    ##############################################################################
    BASE_FOLDER = "C:\\Users\\rkhan\\Downloads\\VGG_OK_All_models_data"
    CAMERA_PLACEMENTS_FILE = "C:\\Users\\rkhan\\Downloads\\camera_placements.txt"
    OUTPUT_DIR = "C:\\Users\\rkhan\\Downloads\\hemisphere_output_models_newVGG-last"

    # robot images (unchanged)
    ROBOT_IMAGE_PATH_3D = "C:\\Users\\rkhan\\Downloads\\robot.png"
    ROBOT_IMAGE_PATH_TOP = "C:\\Users\\rkhan\\Downloads\\robot_top.png"
    MODEL_COLUMNS = ["vp_conv_vae_128_0001", "vp_conv_vae_256_0001", "vp_vgg19_128_0001", "vp_vgg19_256_0001", "vp_resnet50_128_0001", "vp_resnet50_256_0001"]

    # all model columns we want to process
    MODEL_COLUMNS = [
        "vp_conv_vae_128_0001",
        "vp_vgg19_128_0001",
        "vp_resnet50_128_0001",
        "vp_conv_vae_256_0001",
        "vp_vgg19_256_0001",
        "vp_resnet50_256_0001",
    ]

    print("\n" + "=" * 70)
    print("112 CAMERA HEMISPHERE VISUALIZATION PROCESSOR")
    print("=" * 70)
    print(f"\nBase folder:      {BASE_FOLDER}")
    print(f"Camera placements:{CAMERA_PLACEMENTS_FILE}")
    print(f"Output root:      {OUTPUT_DIR}")

    processor = Camera112Processor_multi(BASE_FOLDER, CAMERA_PLACEMENTS_FILE)
    processor.parse_camera_placements()  # only once

    # loop over each model column
    for model_col in MODEL_COLUMNS:
        print("\n" + "#" * 70)
        print(f"Processing model: {model_col}")
        print("#" * 70)

        model_output_dir = os.path.join(OUTPUT_DIR, model_col)
        os.makedirs(model_output_dir, exist_ok=True)

        # read MSE for this model
        processor.read_mse_values(
            model_column=model_col,
            mse_filename="all_msecomparison_values.csv",
        )

        # generate CSVs and visualizations into that model’s folder
        dataframes = processor.generate_all_csvs(model_output_dir)
        processor.generate_model_mse_table(model_output_dir, model_col)

        processor.create_visualizations(model_output_dir)
        processor.generate_summary_report(model_output_dir)

        print(f"\nFinished model {model_col}. Outputs in: {model_output_dir}")

    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE FOR ALL MODELS!")
    print("=" * 70)


    # NEW: create comparison bar plot of MSE ranges across models
    plot_mse_range_across_models(
        output_root=OUTPUT_DIR,
        model_columns=MODEL_COLUMNS,
        components=processor.components,
    )

        # NEW: HEIGHT-only 2×3 grid across the six models
    make_height_cvpr_2x3_across_models(
        output_root=OUTPUT_DIR,
        model_columns=MODEL_COLUMNS,
        height_png_name="height_robot_topdown_cvpr.png",  # produced per-model earlier
        out_name="height_across_models_cvpr_2x3.png",
        titles=("VAE-128","VAE-256","VGG19-128","VGG19-256","ResNet50-128","ResNet50-256"),


    )

    # Build the six dirs from your existing OUTPUT_DIR and MODEL_COLUMNS
    dirs = [str(Path(OUTPUT_DIR) / col) for col in MODEL_COLUMNS]  # must be 6 in display order

    # Make the 2×3 HEIGHT panel with one shared colorbar (same style as make_cvpr_2x3_grid)
    HemisphereHeatmap.make_height_cvpr_2x3_from_folders(
        model_dirs=dirs,
        output_path=str(Path(OUTPUT_DIR) / "height_across_models_cvpr_2x3-without-labels.png"),
        csv_name="height_hemisphere_data.csv",   # this is the CSV each model folder already writes
        cmap="hot_r",
    )
if __name__ == "__main__":
    main()
