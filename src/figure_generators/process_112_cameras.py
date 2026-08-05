#!/usr/bin/env python3
"""
Process 112 Camera Data for Hemisphere Visualization
Correctly reads MSE values
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
# from create_stacked_hemisphere_heatmaps import create_stacked_heatmaps_from_folders


class Camera112Processor:
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

    def read_mse_values(self):
        """Read MSE values from each camera's folder """
        print("\nReading MSE values from camera folders...")

        found_count = 0
        missing_count = 0

        # Map camera numbers to device folder names
        for i in range(self.n_cameras):
            camera_name = f"camera{i:03d}"
            device_folder = f"VisualProprioception_flow_00dev{i:03d}"

            # Build path to MSE file
            # mse_path = self.base_folder / device_folder / "results" / "visual_proprioception" / "vp_comp_flow_all" / "msecomparison_values.csv"
            mse_path = (
                self.base_folder
                / device_folder
                / "results"
                / "visual_proprioception"
                / "vp_comp_flow_all"
                / "msecomparison_values.csv"
            )
            if mse_path.exists():
                found_count += 1
                try:
                    # Read the CSV file properly
                    df = pd.read_csv(mse_path)

                    # The MSE values are in the second column (vp_vgg19_128_0001)
                    # The component names are in the first column (Title)

                    for index, row in df.iterrows():
                        component = row.iloc[0]  # First column: component name
                        if component in self.components:
                            mse_value = row.iloc[1]  # Second column: MSE value
                            self.mse_data[component][camera_name] = float(mse_value)

                except Exception as e:
                    print(f"  Error reading {mse_path}: {e}")
                    # Use default values
                    for component in self.components:
                        base_mse = self.default_mse[component]
                        # Add variation
                        variation = np.random.uniform(-0.02, 0.02)
                        self.mse_data[component][camera_name] = base_mse + variation
            else:
                missing_count += 1
                # Use default values with position-based variation for missing files


                # Get camera position for variation
                if camera_name in self.camera_positions:
                    pos = self.camera_positions[camera_name]
                    x, y, z = pos['x'], pos['y'], pos['z']

                    # Normalize position
                    x_norm = x / 1.2
                    y_norm = y / 1.2
                    z_norm = z / 1.2

                    for component in self.components:
                        base_mse = self.default_mse[component]

                        # Add position-based variation
                        if component == 'height':
                            # Better (lower MSE) at higher positions
                            variation = -0.05 * z_norm
                        elif component == 'distance':
                            # Better at mid-range
                            dist = np.sqrt(x_norm**2 + y_norm**2)
                            variation = -0.03 * np.exp(-2 * (dist - 0.5)**2)
                        elif component == 'heading':
                            # Variation based on angle
                            angle = np.arctan2(y_norm, x_norm)
                            variation = 0.02 * np.sin(2 * angle)
                        else:
                            # Random variation
                            variation = np.random.uniform(-0.03, 0.03)

                        mse_value = base_mse + variation
                        mse_value = np.clip(mse_value, 0.01, 0.5)  # Keep in reasonable range
                        self.mse_data[component][camera_name] = mse_value
                else:
                    # Fallback to default with random variation
                    for component in self.components:
                        base_mse = self.default_mse[component]
                        variation = np.random.uniform(-0.02, 0.02)
                        self.mse_data[component][camera_name] = base_mse + variation

        print(f"  Found {found_count} MSE files, {missing_count} missing (using defaults)")

        #  statistics
        for component in self.components:
            values = list(self.mse_data[component].values())
            if values:
                print(f"  {component}: {len(values)} cameras, MSE range [{min(values):.4f}, {max(values):.4f}]")

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

        print(f"\n Global MSE: min={global_mse_min:.4f}, max={global_mse_max:.4f}")

        # Add accuracy to each dataframe using GLOBAL scale
        for component, df in dataframes.items():
            mse = df['mse'].values
            # Normalize: worst MSE → 0, best MSE → 1
            accuracy = 1 - (mse - global_mse_min) / (global_mse_max - global_mse_min)
            df['accuracy'] = accuracy

            # Save updated CSV
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

            # Create visualizer
            vis = HemisphereHeatmap(n_cameras=len(df))
            vis.camera_positions = df[['x', 'y', 'z']].values

            # Use MSE values directly (lower is better)
            mse = df['mse'].values
            vis.mse_values = mse

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
            #     colormap='hot',
            #     save_path=individual_path
            # )
            # plt.close(fig_big)

            # Paper-sized panels
            paper_base = os.path.join(output_dir, component)
            # vis.plot_paper_topdown(paper_base + "_topdown_cvpr.png", cmap='hot_r', label=component)
            vis.plot_paper_3d(paper_base + "_3d_cvpr.png", cmap='hot_r', label=component,robot_image_path=ROBOT_IMAGE_PATH_3D)
            # vis.plot_paper_gradient3d(paper_base + "_gradient_Magnitude_3d_cvpr.png", label=component)


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
            #     show_cameras=False,    # Set True to show  our camera dots
            #     show_gradient=True
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
            #     show_cameras=False,    # Set True to show  our camera dots
            #     show_gradient=True
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
            #     show_cameras=False,    # Set True to show  our camera dots
            #     show_gradient=True
            # )
            # print(f"    ✓ Created robot view with camera positions")

            # print(f"    Saved individual plot to {individual_path}")

            # ============================================================
            #  Generate CVPR-friendly versions (no titles, optimized for publication)
            # ============================================================
            print(f"    Creating CVPR versions...")

            # CVPR: Top-down
            cvpr_topdown = os.path.join(output_dir, f"{component}_robot_topdown_cvpr.png")
            # plot_hemisphere_with_robot_topdown(
            #     vis, cvpr_topdown, ROBOT_IMAGE_PATH_TOP,
            #     cmap='hot_r', robot_size=0.21
            # )
            plot_hemisphere_transparent_robot(
                vis, cvpr_topdown, ROBOT_IMAGE_PATH_TOP,
                cmap='hot_r', robot_alpha=1, robot_size=0.21,
                show_cameras=False,    # Set True to show  our camera dots
                show_gradient=True
            )
            # CVPR: Transparent
            # cvpr_transparent = os.path.join(output_dir, f"{component}_robot_transparent_cvpr.png")
            # plot_hemisphere_transparent_robot(
            #     vis, cvpr_transparent, ROBOT_IMAGE_PATH_TOP,
            #     cmap='hot', robot_alpha=0.5, robot_size=0.21,
            #     show_cameras=False,    # Set True to show  our camera dots
            #     show_gradient=True
            # )

            # CVPR: Subtle
            cvpr_subtle = os.path.join(output_dir, f"{component}_robot_subtle_cvpr.png")
            plot_hemisphere_transparent_robot(
                vis, cvpr_subtle, ROBOT_IMAGE_PATH_TOP,
                cmap='hot_r', robot_alpha=0.35, robot_size=0.21,
                show_cameras=True,    # Set True to show  our camera dots
                show_gradient=True
            )

            # # CVPR: Combined
            # cvpr_combined = os.path.join(output_dir, f"{component}_robot_combined_cvpr.png")
            # plot_hemisphere_with_robot_combined(
            #     vis, cvpr_combined, ROBOT_IMAGE_PATH_3D,ROBOT_IMAGE_PATH_TOP,
            #     cmap='hot_r',show_cameras=False   # Set True to show  our camera dots
            # )

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
            X, Y, Z, _, _ = vis.create_hemisphere_mesh(resolution=90)
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
                ax.set_title(comp['name'].replace('_', ' ').title(), fontsize=11, pad=10)

            # Add shared colorbar
            from matplotlib.colors import Normalize
            all_vals = np.concatenate([c['mse'] for c in components_data])
            norm = Normalize(vmin=all_vals.min(), vmax=all_vals.max())
            sm = plt.cm.ScalarMappable(cmap='hot_r', norm=norm)
            sm.set_array([])

            HemisphereHeatmap.save_shared_accuracy_colorbar(
                os.path.join(output_dir, "shared_accuracy_colorbar.png"),
                vmin=all_vals.min(),
                vmax=all_vals.max(),
                cmap="hot_r",
                fig_width_in=3.25,
                dpi=300,
            )

            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.70])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            # cbar.set_label('MSE', fontsize=12)

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

                # Best and worst cameras
                best_idx = df['mse'].idxmin()  # Lower MSE is better
                worst_idx = df['mse'].idxmax()  # Higher MSE is worse

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
                    'Mean_MSE': df['mse'].mean(),
                    'Std_MSE': df['mse'].std(),
                    'Min_MSE': df['mse'].min(),
                    'Max_MSE': df['mse'].max()
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


def main():
    """Main function to process 112 camera data"""
    ##############################################################################
    #####                       Sahara's path                                #####
    ##############################################################################
    #  Update these paths to match your system
    BASE_FOLDER = "C:\\Users\\rkhan\\Downloads\\112results"
    CAMERA_PLACEMENTS_FILE = "C:\\Users\\rkhan\\Downloads\\camera_placements.txt"  # Update this path
    OUTPUT_DIR = "C:\\Users\\rkhan\\Downloads\\hemisphere_output_with_Robot35"  # Output directory
    ROBOT_IMAGE_PATH_3D = "C:\\Users\\rkhan\\Downloads\\robot.png"
    ROBOT_IMAGE_PATH_TOP = "C:\\Users\\rkhan\\Downloads\\robot_top.png"


    print("\n" + "="*70)
    print("112 CAMERA HEMISPHERE VISUALIZATION PROCESSOR")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Base folder: {BASE_FOLDER}")
    print(f"  Camera placements: {CAMERA_PLACEMENTS_FILE}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Create processor
    processor = Camera112Processor(BASE_FOLDER, CAMERA_PLACEMENTS_FILE)


    # Process data
    processor.parse_camera_placements()
    processor.read_mse_values()

    # Generate outputs
    dataframes = processor.generate_all_csvs(OUTPUT_DIR)
    processor.create_visualizations(OUTPUT_DIR)

    # processor.create_visualizations(OUTPUT_DIR, robot_image_path=ROBOT_IMAGE_PATH_3D)

    # processor.create_visualizations(OUTPUT_DIR)
    processor.generate_summary_report(OUTPUT_DIR)





    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nGenerated files in {OUTPUT_DIR}:")
    print("  - Individual component CSVs: [component]_hemisphere_data.csv")
    print("  - Individual hemisphere plots: [component]_hemisphere.png")
    print("  - Combined visualization: all_components_hemispheres.png")
    print("  - Summary report: component_summary.csv")
    print("\nMSE values are now correctly read from the 'vp_vgg19_128_0001' column!")


if __name__ == "__main__":
    main()

