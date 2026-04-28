#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
# ROS 2 Humble port of sharedAutonomyController.py
#
# This node implements a shared autonomy controller for a mobile robot.
# It blends human joystick input with repulsive potential fields derived
# from LiDAR scan data to provide obstacle avoidance assistance.
# -------------------------------------------------------------------------------------------------

import math
import numpy as np

from scipy.spatial import ConvexHull

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from tf2_ros import Buffer, TransformListener
import tf_transformations

from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan, PointCloud2, Joy
from sensor_msgs_py import point_cloud2

# -------------------------------------------------------------------------------------------------
# Global constants
# -------------------------------------------------------------------------------------------------

# Repulsive force computation mode:
#   ALL_OBSTACLES  — sum repulsion vectors from every detected obstacle cluster
#   CLOSEST_OBSTACLE — use only the strongest (closest) repulsion vector
ALL_OBSTACLES = 0
CLOSEST_OBSTACLE = 1

# Normalization divisors applied to repulsion, resultant, and final velocity vectors.
# Set to 1 to pass values through unchanged.
NORM_FACTOR_0 = 1   # Per-obstacle repulsion vector
NORM_FACTOR_1 = 1   # Repulsive resultant vector
NORM_FACTOR_2 = 1   # Final blended velocity signal

# Repulsive potential field gain constant.
# Higher values produce stronger obstacle avoidance forces.
K_REP = 0.375


class SharedAutonomyController(Node):
    def __init__(self):
        super().__init__('shared_autonomy_controller_node')

        # ----------------------------------------------------------------------------
        # ROS 2 parameters — can be overridden via YAML config or CLI arguments
        # ----------------------------------------------------------------------------
        self.declare_parameter('scan_topic', '/scan_raw')
        self.declare_parameter('joy_topic', '/ps5/joy')
        self.declare_parameter('output_joy_topic', 'sac/joy')
        self.declare_parameter('loop_frequency', 250.0)         # Hz — main control loop rate
        self.declare_parameter('marker_threshold_range', 2.0)   # m — max range to visualize obstacles
        self.declare_parameter('scan_threshold_range', 5.0)     # m — max range to include scan points
        self.declare_parameter('rho_0', 1.5)    # m — influence radius; repulsion is zero beyond this
        self.declare_parameter('rho_cap', 0.5)  # m — saturation radius; repulsion is clamped inside this
        self.declare_parameter('rep_from', CLOSEST_OBSTACLE)    # Which obstacles contribute to repulsion

        self.loop_frequency         = float(self.get_parameter('loop_frequency').value)
        self.marker_threshold_range = float(self.get_parameter('marker_threshold_range').value)
        self.scan_threshold_range   = float(self.get_parameter('scan_threshold_range').value)
        self.rho_0                  = float(self.get_parameter('rho_0').value)
        self.rho_cap                = float(self.get_parameter('rho_cap').value)
        self.rep_from               = int(self.get_parameter('rep_from').value)

        # Marker lifetime is tied to the loop period so stale markers auto-expire
        self.marker_lifetime = 1.0 / self.loop_frequency
        self.k_rep = K_REP

        # ----------------------------------------------------------------------------
        # LiDAR scan metadata — populated on first scan message
        # ----------------------------------------------------------------------------
        self.front_minAngle     = 0.0
        self.front_maxAngle     = 0.0
        self.front_angIncrement = 0.0
        self.front_minRange     = 0.0
        self.front_maxRange     = 0.0
        self.front_ranges       = []
        self.front_FoV          = 0.0   # Field of view (radians)
        self.front_noOfScans    = 0.0   # Total number of scan beams per sweep

        # ----------------------------------------------------------------------------
        # Working data — reset every control loop iteration
        # ----------------------------------------------------------------------------
        self.base_r_values      = []   # Radial distances of scan points in base_link frame
        self.base_theta_values  = []   # Angles of scan points in base_link frame
        self.roi1_ranges        = []   # Region of Interest 1: outer boundary (expanded outward from obstacle)
        self.roi2_ranges        = []   # Region of Interest 2: inner boundary (contracted toward centroid)
        self.centroids          = []   # Geometric centroids of each obstacle cluster
        self.closestPoints      = []   # Nearest point within each obstacle cluster to the robot
        self.rep_points         = []   # Computed repulsive force vectors, one per obstacle

        # ----------------------------------------------------------------------------
        # Velocity vectors used in the blending pipeline
        # ----------------------------------------------------------------------------
        self.rep_resultant  = Point(x=0.0, y=0.0, z=0.0)  # Net repulsive force from obstacles
        self.ref_signal     = Point(x=0.0, y=0.0, z=0.0)  # Raw joystick input from the human operator
        self.vfinal_signal  = Point(x=0.0, y=0.0, z=0.0)  # Blended output: human intent + repulsion

        # ----------------------------------------------------------------------------
        # Joystick / operator state
        # ----------------------------------------------------------------------------
        self.deadman_switch  = 0        # Button that must be held for motion to be enabled
        self.autonomy_switch = 0        # (Reserved) toggle for autonomy level
        self.vfinal_joy      = Joy()    # Output Joy message, modified from the incoming one

        # ----------------------------------------------------------------------------
        # TF2 — used to transform LiDAR points from lidar_frame into base_link
        # ----------------------------------------------------------------------------
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ----------------------------------------------------------------------------
        # Publishers
        # ----------------------------------------------------------------------------
        self.pcld2_pub       = self.create_publisher(PointCloud2,  '/transformed_pcld2', 10)
        self.marker_pub      = self.create_publisher(Marker,       '/obstacle_marker',   10)
        self.centroid_pub    = self.create_publisher(Marker,       '/centroid_marker',   10)
        self.scan1_pub       = self.create_publisher(LaserScan,    '/ROI1_laserScan',    10)
        self.scan2_pub       = self.create_publisher(LaserScan,    '/ROI2_laserScan',    10)
        self.repForce_pub    = self.create_publisher(MarkerArray,  '/rep_marker',        10)
        self.resForce_pub    = self.create_publisher(Marker,       '/res_marker',        10)
        self.refSignal_pub   = self.create_publisher(Marker,       '/refSignal_marker',  10)
        self.vfinal_marker_pub = self.create_publisher(Marker,     '/vfinal_marker',     10)

        output_joy_topic = self.get_parameter('output_joy_topic').value
        self.vfinal_joy_pub = self.create_publisher(Joy, output_joy_topic, 10)

        # ----------------------------------------------------------------------------
        # Subscribers
        # ----------------------------------------------------------------------------
        scan_topic = self.get_parameter('scan_topic').value
        joy_topic  = self.get_parameter('joy_topic').value

        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.frontScan_callback, 10
        )
        self.joy_sub = self.create_subscription(
            Joy, joy_topic, self.joy_bs_callback, 10
        )

        # ----------------------------------------------------------------------------
        # Main control loop timer — replaces the ROS 1 while-loop + rate.sleep() pattern
        # ----------------------------------------------------------------------------
        self.main_timer = self.create_timer(1.0 / self.loop_frequency, self.main_loop)

        self.displayLaserSpecs_once = False  # Guard so specs are logged only on startup

    # =============================================================================
    # Callbacks
    # =============================================================================

    def frontScan_callback(self, scan_msg: LaserScan):
        """Cache the latest LiDAR scan metadata and range array."""
        self.front_minAngle     = scan_msg.angle_min
        self.front_maxAngle     = scan_msg.angle_max
        self.front_angIncrement = scan_msg.angle_increment
        self.front_FoV          = self.front_maxAngle - self.front_minAngle

        if self.front_angIncrement != 0.0:
            self.front_noOfScans = math.degrees(self.front_FoV) / math.degrees(self.front_angIncrement)
        else:
            self.front_noOfScans = 0.0

        self.front_minRange = scan_msg.range_min
        self.front_maxRange = scan_msg.range_max
        self.front_ranges   = list(scan_msg.ranges)

    def joy_bs_callback(self, joy_bs_msg: Joy):
        """
        Cache the latest joystick message and extract the operator's velocity intent.

        Axis mapping (PS5 controller default):
          axes[0] — left stick horizontal (→ positive = left)
          axes[1] — left stick vertical   (→ positive = forward)
        Button mapping:
          buttons[6] — L2 trigger used as deadman switch
        """
        self.vfinal_joy = joy_bs_msg  # Keep full message to forward axes/buttons unchanged

        # Map left-stick axes to the reference signal (forward = x, lateral = y)
        if len(joy_bs_msg.axes) >= 2:
            self.ref_signal.x = joy_bs_msg.axes[1]
            self.ref_signal.y = joy_bs_msg.axes[0]
        else:
            self.ref_signal.x = 0.0
            self.ref_signal.y = 0.0
        self.ref_signal.z = 0.0

        # Deadman switch must be held for the robot to move
        self.deadman_switch = joy_bs_msg.buttons[6] if len(joy_bs_msg.buttons) > 6 else 0

    # =============================================================================
    # Helpers
    # =============================================================================

    def getAngle(self, range_index: int) -> float:
        """Convert a scan array index to the corresponding beam angle (radians)."""
        return self.front_minAngle + (range_index * self.front_angIncrement)

    def getTransform(self, target_frame: str, source_frame: str):
        """
        Look up the latest TF2 transform between two frames.

        Returns (translation, rotation) tuples on success, or (None, None) on failure.
        Translation is (x, y, z); rotation is a quaternion (x, y, z, w).
        """
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, now, timeout=Duration(seconds=0.5)
            )
            trans = (
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            )
            rot = (
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            )
            return trans, rot
        except Exception as e:
            self.get_logger().error(
                f'Transform lookup from {source_frame} to {target_frame} failed: {e}'
            )
            return None, None

    def publish_transformed_pointCloud(self):
        """
        Transform all valid LiDAR scan points from lidar_frame into base_link,
        then publish them as a PointCloud2 message.

        Also populates self.base_r_values and self.base_theta_values, which are
        used downstream by publish_obstacles() to detect and cluster obstacles.
        """
        trans, rot = self.getTransform('base_link', 'lidar_frame')
        if trans is None or rot is None:
            self.get_logger().warn('Frame transformation skipped.')
            return

        transformed_points    = []
        self.base_r_values    = []
        self.base_theta_values = []

        for i, r in enumerate(self.front_ranges):
            # Skip invalid readings (out-of-range or sensor noise)
            if self.front_minRange < r < self.front_maxRange:
                theta = self.getAngle(i)

                # Convert polar scan point to Cartesian in lidar_frame
                x_lidar = r * math.cos(theta)
                y_lidar = r * math.sin(theta)
                z_lidar = 0.0
                lidar_point = np.array([x_lidar, y_lidar, z_lidar, 1.0])  # Homogeneous coords

                # Build the 4×4 homogeneous transform: T = Translation × Rotation
                Rq = tf_transformations.quaternion_matrix(rot)
                Tr = np.array([
                    [1, 0, 0, trans[0]],
                    [0, 1, 0, trans[1]],
                    [0, 0, 1, trans[2]],
                    [0, 0, 0, 1       ],
                ])
                Tx = np.dot(Tr, Rq)

                # Apply transform to get the point in base_link frame
                base_point = np.dot(Tx, lidar_point)
                x_base = float(base_point[0])
                y_base = float(base_point[1])
                z_base = float(base_point[2])
                transformed_points.append((x_base, y_base, z_base))

                # Convert back to polar for threshold-based obstacle detection
                r_base     = math.sqrt(x_base**2 + y_base**2)
                theta_base = math.atan2(y_base, x_base)
                self.base_r_values.append(r_base)
                self.base_theta_values.append(theta_base)

        header           = Header()
        header.stamp     = self.get_clock().now().to_msg()
        header.frame_id  = 'base_link'
        pcld2_msg = point_cloud2.create_cloud_xyz32(header, transformed_points)
        self.pcld2_pub.publish(pcld2_msg)

    def displayLaserSpecs(self):
        """Log LiDAR sensor specifications once at startup for diagnostics."""
        self.get_logger().info('*** ROSOrin+ Front Laser Specifications ***')
        self.get_logger().info(f'Minimum Angle in degrees: {math.degrees(self.front_minAngle)}')
        self.get_logger().info(f'Maximum Angle in degrees: {math.degrees(self.front_maxAngle)}')
        self.get_logger().info(f'Angle increment in degrees: {math.degrees(self.front_angIncrement)}')
        self.get_logger().info(f'FoV in degrees: {math.degrees(self.front_FoV)}')
        self.get_logger().info(f'Number of scans per sweep: {self.front_noOfScans}')
        self.get_logger().info(f'Min Range scanned in m: {self.front_minRange}')
        self.get_logger().info(f'Max Range scanned in m: {self.front_maxRange}')
        self.get_logger().info(f'Range array size: {len(self.front_ranges)}')
        self.get_logger().info('*** *** ***')

    def record_closestPoint(self, points):
        """
        Find the point in a cluster that is closest to the robot origin and
        append it to self.closestPoints.

        This point is later used to compute the repulsive force for the cluster.
        """
        cPoint = Point()
        min_distance = 5.5  # Initial sentinel — larger than any expected obstacle range
        for point in points:
            distance = math.sqrt(point.x**2 + point.y**2)
            if distance <= min_distance:
                min_distance = distance
                cPoint = point
        self.closestPoints.append(cPoint)

    def compute_centroid(self, points):
        """
        Compute the arithmetic mean (centroid) of a set of 2-D points and
        append it to self.centroids.

        The centroid is used to define the outward/inward directions for ROI expansion.
        """
        n = len(points)
        if n == 0:
            return Point(x=0.0, y=0.0, z=0.0)
        g_x = sum(p.x for p in points) / n
        g_y = sum(p.y for p in points) / n
        centroid = Point(x=g_x, y=g_y, z=0.0)
        self.centroids.append(centroid)
        return centroid

    def publish_centroids(self):
        """Publish all obstacle centroids as a single POINTS marker (green dots)."""
        centroid_marker = Marker()
        centroid_marker.header.frame_id  = 'base_link'
        centroid_marker.header.stamp     = self.get_clock().now().to_msg()
        centroid_marker.type             = Marker.POINTS
        centroid_marker.scale.x          = 0.05
        centroid_marker.scale.y          = 0.05
        centroid_marker.color.r          = 0.0
        centroid_marker.color.g          = 1.0
        centroid_marker.color.b          = 0.0
        centroid_marker.color.a          = 1.0
        centroid_marker.points           = self.centroids
        centroid_marker.pose.orientation.w = 1.0
        centroid_marker.lifetime         = Duration(seconds=self.marker_lifetime).to_msg()
        self.centroid_pub.publish(centroid_marker)

    def compute_convexhull(self, points):
        """
        Reduce an obstacle point cluster to its convex hull vertices.

        Returns the original point list unchanged if fewer than 3 points are present
        (ConvexHull requires at least 3 non-collinear points).
        """
        if len(points) < 3:
            return points
        points_np = np.array([(p.x, p.y) for p in points])
        cvxhull   = ConvexHull(points_np)
        return [Point(x=p[0], y=p[1], z=0.0) for p in points_np[cvxhull.vertices]]

    def compute_roi1(self, points, centroid):
        """
        Compute Region of Interest 1 — the outer safety boundary around an obstacle.

        For each hull point, a boundary point is projected outward (away from the
        centroid) by rho_0 metres. The resulting polar coordinates are written into
        self.roi1_ranges so the boundary can be published as a LaserScan.
        """
        for point in points:
            # Direction vector pointing outward from centroid through the hull point
            dir_x  = point.x - centroid.x
            dir_y  = point.y - centroid.y
            length = math.sqrt(dir_x**2 + dir_y**2)
            if length == 0.0:
                continue

            # Unit vector in the outward direction
            unit_x = dir_x / length
            unit_y = dir_y / length

            # Project rho_0 metres outward to get the ROI boundary point
            roi1_x = point.x + unit_x * self.rho_0
            roi1_y = point.y + unit_y * self.rho_0
            roi1_r     = math.sqrt(roi1_x**2 + roi1_y**2)
            roi1_theta = math.atan2(roi1_y, roi1_x)

            # Write into the scan range array at the corresponding angular index
            if self.front_minAngle <= roi1_theta <= self.front_maxAngle:
                index = int((roi1_theta - self.front_minAngle) / self.front_angIncrement)
                if 0 <= index < len(self.roi1_ranges):
                    self.roi1_ranges[index] = roi1_r

    def compute_roi2(self, points, centroid):
        """
        Compute Region of Interest 2 — the inner boundary (toward the obstacle interior).

        For each hull point, a boundary point is projected inward (toward the centroid)
        by rho_0 metres. Written into self.roi2_ranges as a LaserScan-compatible array.
        """
        for point in points:
            # Direction vector pointing inward (centroid minus hull point)
            dir_x  = centroid.x - point.x
            dir_y  = centroid.y - point.y
            length = math.sqrt(dir_x**2 + dir_y**2)
            if length == 0.0:
                continue

            unit_x = dir_x / length
            unit_y = dir_y / length

            # Project rho_0 metres inward
            roi2_x = point.x + unit_x * self.rho_0
            roi2_y = point.y + unit_y * self.rho_0
            roi2_r     = math.sqrt(roi2_x**2 + roi2_y**2)
            roi2_theta = math.atan2(roi2_y, roi2_x)

            if self.front_minAngle <= roi2_theta <= self.front_maxAngle:
                index = int((roi2_theta - self.front_minAngle) / self.front_angIncrement)
                if 0 <= index < len(self.roi2_ranges):
                    self.roi2_ranges[index] = roi2_r

    def publish_roi1(self):
        """Publish ROI 1 (outer boundary) as a LaserScan on /ROI1_laserScan."""
        scan1                  = LaserScan()
        scan1.header.frame_id  = 'base_link'
        scan1.header.stamp     = self.get_clock().now().to_msg()
        scan1.angle_min        = self.front_minAngle
        scan1.angle_max        = self.front_maxAngle
        scan1.angle_increment  = self.front_angIncrement
        scan1.range_min        = self.front_minRange
        scan1.range_max        = self.front_maxRange
        scan1.ranges           = self.roi1_ranges
        self.scan1_pub.publish(scan1)

    def publish_roi2(self):
        """Publish ROI 2 (inner boundary) as a LaserScan on /ROI2_laserScan."""
        scan2                  = LaserScan()
        scan2.header.frame_id  = 'base_link'
        scan2.header.stamp     = self.get_clock().now().to_msg()
        scan2.angle_min        = self.front_minAngle
        scan2.angle_max        = self.front_maxAngle
        scan2.angle_increment  = self.front_angIncrement
        scan2.range_min        = self.front_minRange
        scan2.range_max        = self.front_maxRange
        scan2.ranges           = self.roi2_ranges
        self.scan2_pub.publish(scan2)

    def publish_potentialFields(self):
        """
        Compute and publish repulsive potential field vectors for each obstacle cluster.

        The repulsive force follows the classical artificial potential field formula:

            F_rep = K_rep / rho^2 * (1/rho - 1/rho_0)   for rho_cap <= rho <= rho_0
            F_rep = K_rep / rho_cap^2 * (1/rho_cap - 1/rho_0)  for rho < rho_cap  (saturated)
            F_rep = 0                                             for rho > rho_0   (outside influence)

        where rho is the distance to the closest point of the obstacle cluster.
        Each vector is published as an ARROW marker pointing away from the obstacle.
        """
        rep_markerArray = MarkerArray()
        self.rep_points = []

        for i, cPoint in enumerate(self.closestPoints):
            rho_xy = math.sqrt(cPoint.x**2 + cPoint.y**2)
            if rho_xy == 0.0:
                continue  # Avoid division-by-zero for a point exactly at the robot origin

            # Unit vector pointing away from the obstacle (repulsion direction)
            rep_unit_x = -cPoint.x / rho_xy
            rep_unit_y = -cPoint.y / rho_xy

            if self.rho_cap <= rho_xy <= self.rho_0:
                # Normal repulsion zone: force increases as robot gets closer
                scale  = (self.k_rep / rho_xy**2) * ((1.0 / rho_xy) - (1.0 / self.rho_0))
                rep_x  = rep_unit_x * scale / NORM_FACTOR_0
                rep_y  = rep_unit_y * scale / NORM_FACTOR_0
            elif rho_xy < self.rho_cap:
                # Saturation zone: cap the force to avoid singularity as rho → 0
                scale  = (self.k_rep / self.rho_cap**2) * ((1.0 / self.rho_cap) - (1.0 / self.rho_0))
                rep_x  = rep_unit_x * scale / NORM_FACTOR_0
                rep_y  = rep_unit_y * scale / NORM_FACTOR_0
            else:
                # Outside influence radius — no repulsion
                rep_x, rep_y = 0.0, 0.0

            rep_point = Point(x=rep_x, y=rep_y, z=0.0)
            self.rep_points.append(rep_point)

            # Visualize as a purple arrow from the robot origin to the repulsion vector tip
            rep_marker                      = Marker()
            rep_marker.header.frame_id      = 'base_link'
            rep_marker.header.stamp         = self.get_clock().now().to_msg()
            rep_marker.type                 = Marker.ARROW
            rep_marker.id                   = i
            rep_marker.ns                   = 'repulsive_forces'
            rep_marker.scale.x              = 0.05
            rep_marker.scale.y              = 0.1
            rep_marker.scale.z              = 0.15
            rep_marker.color.r              = 0.8
            rep_marker.color.g              = 0.2
            rep_marker.color.b              = 0.8
            rep_marker.color.a              = 1.0
            rep_marker.pose.orientation.w   = 1.0
            rep_marker.points               = [Point(x=0.0, y=0.0, z=0.0), rep_point]
            rep_marker.lifetime             = Duration(seconds=self.marker_lifetime).to_msg()
            rep_markerArray.markers.append(rep_marker)

        self.repForce_pub.publish(rep_markerArray)

    def compute_resultant(self, points):
        """Sum a list of Point vectors component-wise and return the resultant Point."""
        resultant = Point(x=0.0, y=0.0, z=0.0)
        for point in points:
            resultant.x += point.x
            resultant.y += point.y
            resultant.z += point.z
        return resultant

    def publish_repulsiveResultant(self):
        """
        Compute the net repulsive vector and publish it as a red ARROW marker.

        In ALL_OBSTACLES mode the contributions from every cluster are summed.
        In CLOSEST_OBSTACLE mode only the vector with the largest magnitude is used
        (i.e. the obstacle that is imposing the strongest repulsion right now).
        """
        if self.rep_from == ALL_OBSTACLES:
            self.rep_resultant = self.compute_resultant(self.rep_points)
        else:  # CLOSEST_OBSTACLE
            if self.rep_points:
                # Select the repulsion vector with the greatest magnitude
                closest_rep_point     = self.rep_points[0]
                closest_rep_magnitude = math.sqrt(closest_rep_point.x**2 + closest_rep_point.y**2)
                for rep_point in self.rep_points:
                    rep_magnitude = math.sqrt(rep_point.x**2 + rep_point.y**2)
                    if rep_magnitude > closest_rep_magnitude:
                        closest_rep_point     = rep_point
                        closest_rep_magnitude = rep_magnitude
                self.rep_resultant = closest_rep_point
            else:
                self.rep_resultant = Point(x=0.0, y=0.0, z=0.0)

        # Apply normalization (currently a no-op with NORM_FACTOR_1 = 1)
        self.rep_resultant.x /= NORM_FACTOR_1
        self.rep_resultant.y /= NORM_FACTOR_1
        self.rep_resultant.z /= NORM_FACTOR_1

        # Publish the resultant as a red arrow for RViz visualization
        resultant_marker                    = Marker()
        resultant_marker.header.frame_id    = 'base_link'
        resultant_marker.header.stamp       = self.get_clock().now().to_msg()
        resultant_marker.type               = Marker.ARROW
        resultant_marker.ns                 = 'repulsive_resultant'
        resultant_marker.scale.x            = 0.05
        resultant_marker.scale.y            = 0.1
        resultant_marker.scale.z            = 0.15
        resultant_marker.color.r            = 1.0
        resultant_marker.color.g            = 0.0
        resultant_marker.color.b            = 0.0
        resultant_marker.color.a            = 1.0
        resultant_marker.pose.orientation.w = 1.0
        resultant_marker.points             = [Point(x=0.0, y=0.0, z=0.0), self.rep_resultant]
        self.resForce_pub.publish(resultant_marker)

    def publish_referenceSignal(self):
        """Publish the operator's raw joystick intent as a blue ARROW marker."""
        reference_marker                    = Marker()
        reference_marker.header.frame_id    = 'base_link'
        reference_marker.header.stamp       = self.get_clock().now().to_msg()
        reference_marker.type               = Marker.ARROW
        reference_marker.scale.x            = 0.05
        reference_marker.scale.y            = 0.1
        reference_marker.scale.z            = 0.15
        reference_marker.color.r            = 0.0
        reference_marker.color.g            = 0.0
        reference_marker.color.b            = 1.0
        reference_marker.color.a            = 1.0
        reference_marker.pose.orientation.w = 1.0
        reference_marker.points             = [Point(x=0.0, y=0.0, z=0.0), self.ref_signal]
        self.refSignal_pub.publish(reference_marker)

    def publish_finalVelocity_marker(self):
        """
        Blend the human joystick signal with the repulsive field resultant to produce
        the final commanded velocity, then publish it as a green ARROW marker.

        Blending strategy: simple vector addition (shared autonomy additive model).
        The output is clamped to [-1, 1] to stay within joystick axis range.
        """
        # Additive blend: human intent + obstacle repulsion
        self.vfinal_signal = self.compute_resultant([self.ref_signal, self.rep_resultant])

        # Normalization is a no-op here (NORM_FACTOR_2 = 1), but left for future tuning
        self.vfinal_signal.x /= NORM_FACTOR_2
        self.vfinal_signal.y /= NORM_FACTOR_2
        self.vfinal_signal.z /= NORM_FACTOR_2

        # Clamp to joystick axis range so downstream controllers are never over-driven
        self.vfinal_signal.x = max(-1.0, min(1.0, self.vfinal_signal.x))
        self.vfinal_signal.y = max(-1.0, min(1.0, self.vfinal_signal.y))

        # Publish green arrow for RViz
        vfinal_marker                    = Marker()
        vfinal_marker.header.frame_id    = 'base_link'
        vfinal_marker.header.stamp       = self.get_clock().now().to_msg()
        vfinal_marker.type               = Marker.ARROW
        vfinal_marker.scale.x            = 0.05
        vfinal_marker.scale.y            = 0.1
        vfinal_marker.scale.z            = 0.15
        vfinal_marker.color.r            = 0.0
        vfinal_marker.color.g            = 1.0
        vfinal_marker.color.b            = 0.0
        vfinal_marker.color.a            = 1.0
        vfinal_marker.pose.orientation.w = 1.0
        vfinal_marker.points             = [Point(x=0.0, y=0.0, z=0.0), self.vfinal_signal]
        self.vfinal_marker_pub.publish(vfinal_marker)

        # Log magnitudes for runtime debugging / tuning
        ref_mag    = math.sqrt(self.ref_signal.x**2    + self.ref_signal.y**2)
        rep_mag    = math.sqrt(self.rep_resultant.x**2 + self.rep_resultant.y**2)
        vfinal_mag = math.sqrt(self.vfinal_signal.x**2 + self.vfinal_signal.y**2)
        self.get_logger().info(
            f"\n"
            f"  ref    : ({self.ref_signal.x:+.3f}, {self.ref_signal.y:+.3f}) | mag={ref_mag:.3f}\n"
            f"  rep    : ({self.rep_resultant.x:+.3f}, {self.rep_resultant.y:+.3f}) | mag={rep_mag:.3f}\n"
            f"  vfinal : ({self.vfinal_signal.x:+.3f}, {self.vfinal_signal.y:+.3f}) | mag={vfinal_mag:.3f}"
        )

    def publish_vfinal_joy(self):
        """
        Overwrite the lateral (axes[0]) and longitudinal (axes[1]) components of the
        cached Joy message with the blended vfinal_signal, then republish.

        All other axes and buttons are forwarded unchanged so nothing is lost
        (e.g. rotation axis, triggers, bumpers).
        """
        vfinal_joy_axes = list(self.vfinal_joy.axes)
        if len(vfinal_joy_axes) < 2:
            # Pad if the message arrived with fewer axes than expected
            vfinal_joy_axes = [0.0, 0.0] + vfinal_joy_axes[2:]
        vfinal_joy_axes[0] = self.vfinal_signal.y   # Lateral  → axes[0]
        vfinal_joy_axes[1] = self.vfinal_signal.x   # Forward  → axes[1]
        self.vfinal_joy.axes = tuple(vfinal_joy_axes)
        self.vfinal_joy_pub.publish(self.vfinal_joy)

    def publish_obstacles(self):
        """
        Segment the transformed scan into obstacle clusters and process each one.

        Algorithm:
          1. Iterate over scan points sorted by angle.
          2. Accumulate consecutive points that fall within marker_threshold_range
             into a cluster (points list).
          3. When a point falls outside the range (or the scan ends), close the
             current cluster and:
               a. Publish it as a red LINE_STRIP marker.
               b. Record the closest point (for repulsion computation).
               c. Compute its centroid (for ROI boundary directions).
               d. Compute ROI 1 and ROI 2 boundaries.
          4. Reset roi1_ranges / roi2_ranges to large sentinel values each call
             so stale data from the previous cycle does not persist.
        """
        # Initialize ROI range arrays with large sentinel values (no obstacle assumed)
        self.roi1_ranges = [5.5] * len(self.base_r_values)
        self.roi2_ranges = [6.0] * len(self.base_r_values)

        points       = []   # Current cluster being built
        markerNumber = 0    # Unique ID for each obstacle marker

        for j in range(len(self.base_r_values)):
            if self.base_r_values[j] < self.marker_threshold_range:
                # Point is within the obstacle detection range — add to current cluster
                r     = self.base_r_values[j]
                theta = self.base_theta_values[j]
                marker_point   = Point()
                marker_point.x = r * math.cos(theta)
                marker_point.y = r * math.sin(theta)
                marker_point.z = 0.0
                points.append(marker_point)
            else:
                # Gap detected — close and process the cluster accumulated so far
                if points:
                    marker                      = Marker()
                    marker.header.frame_id      = 'base_link'
                    marker.header.stamp         = self.get_clock().now().to_msg()
                    marker.ns                   = 'thresholded_laserScan'
                    marker.id                   = markerNumber
                    marker.type                 = Marker.LINE_STRIP
                    marker.action               = Marker.ADD
                    marker.scale.x              = 0.02
                    marker.color.r              = 1.0
                    marker.color.g              = 0.0
                    marker.color.b              = 0.0
                    marker.color.a              = 1.0
                    marker.pose.orientation.w   = 1.0
                    marker.points               = points
                    marker.lifetime             = Duration(seconds=self.marker_lifetime).to_msg()
                    self.marker_pub.publish(marker)

                    self.record_closestPoint(points)
                    centroid = self.compute_centroid(points)
                    self.compute_roi1(points, centroid)
                    self.compute_roi2(points, centroid)
                    points        = []
                    markerNumber += 1

        # Handle a cluster that extends all the way to the last scan point
        if points:
            marker                      = Marker()
            marker.header.frame_id      = 'base_link'
            marker.header.stamp         = self.get_clock().now().to_msg()
            marker.ns                   = 'thresholded_laserScan'
            marker.id                   = markerNumber
            marker.type                 = Marker.LINE_STRIP
            marker.action               = Marker.ADD
            marker.scale.x              = 0.02
            marker.color.r              = 1.0
            marker.color.g              = 0.0
            marker.color.b              = 0.0
            marker.color.a              = 1.0
            marker.pose.orientation.w   = 1.0
            marker.points               = points
            marker.lifetime             = Duration(seconds=self.marker_lifetime).to_msg()
            self.marker_pub.publish(marker)

            self.record_closestPoint(points)
            centroid = self.compute_centroid(points)
            self.compute_roi1(points, centroid)
            self.compute_roi2(points, centroid)

    # =============================================================================
    # Main loop (250 Hz timer callback)
    # =============================================================================
    def main_loop(self):
        """
        Execute one full sensing → perception → control cycle:

          1. Log laser specs once on startup.
          2. Guard against running before any scan data arrives.
          3. Reset per-cycle data structures.
          4. Transform scan points into base_link and publish the point cloud.
          5. Detect and cluster obstacles; compute ROI boundaries.
          6. Publish all visualization markers.
          7. Compute potential fields and the blended final velocity.
          8. Republish the modified Joy message to the downstream controller.
        """
        # Print specs once as soon as the first scan arrives
        if not self.displayLaserSpecs_once and self.front_ranges:
            self.displayLaserSpecs()
            self.displayLaserSpecs_once = True

        # Wait until scan data is available before doing anything
        if not self.front_ranges:
            return

        # Reset per-iteration accumulators
        self.centroids    = []
        self.closestPoints = []

        # --- Sensing ---
        self.publish_transformed_pointCloud()

        # --- Perception ---
        self.publish_obstacles()       # Cluster scan → markers, centroids, ROIs

        # --- Visualization ---
        self.publish_centroids()
        self.publish_roi1()
        self.publish_roi2()

        # --- Potential field computation ---
        self.publish_potentialFields()        # Per-obstacle repulsion vectors
        self.publish_repulsiveResultant()     # Net repulsion (red arrow)

        # --- Reference signal ---
        self.publish_referenceSignal()        # Human joystick input (blue arrow)

        # --- Blending & output ---
        self.publish_finalVelocity_marker()   # Blended velocity (green arrow) + logging
        self.publish_vfinal_joy()             # Forward modified Joy to robot controller


def main(args=None):
    rclpy.init(args=args)
    node = SharedAutonomyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
