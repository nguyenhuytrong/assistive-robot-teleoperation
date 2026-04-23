#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
# ROS 2 Humble port of sharedAutonomyController_v14.py
# -------------------------------------------------------------------------------------------------

import math
import numpy as np

from scipy.spatial import ConvexHull

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from tf2_ros import Buffer, TransformListener
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2mat, mat2quat


from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan, PointCloud2, Joy
from sensor_msgs_py import point_cloud2

# -------------------------------------------------------------------------------------------------
# Global constants
# -------------------------------------------------------------------------------------------------
ALL_OBSTACLES = 0
CLOSEST_OBSTACLE = 1

NORM_FACTOR_0 = 1
NORM_FACTOR_1 = 1
NORM_FACTOR_2 = 1

K_REP = 0.6


class SharedAutonomyController(Node):
    def __init__(self):
        super().__init__('sac_node')

        # ----------------------------------------------------------------------------
        # Parameters (can be overridden from YAML / CLI)
        # ----------------------------------------------------------------------------
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('joy_topic', '/ps5/joy')
        self.declare_parameter('output_joy_topic', '/sac/joy')
        self.declare_parameter('loop_frequency', 250.0)
        self.declare_parameter('marker_threshold_range', 2.0)
        self.declare_parameter('scan_threshold_range', 5.0)
        self.declare_parameter('rho_0', 1.0)
        self.declare_parameter('rho_cap', 0.75)
        self.declare_parameter('rep_from', CLOSEST_OBSTACLE)

        # Add
        self.declare_parameter('stop_distance', 0.1)  # 0.1m before rho_cap
        self.stop_distance = float(self.get_parameter('stop_distance').value)

        self.loop_frequency = float(self.get_parameter('loop_frequency').value)
        self.marker_threshold_range = float(self.get_parameter('marker_threshold_range').value)
        self.scan_threshold_range = float(self.get_parameter('scan_threshold_range').value)
        self.rho_0 = float(self.get_parameter('rho_0').value)
        self.rho_cap = float(self.get_parameter('rho_cap').value)
        self.rep_from = int(self.get_parameter('rep_from').value)

        self.marker_lifetime = 1.0 / self.loop_frequency
        self.k_rep = K_REP

        # Laser properties
        self.front_minAngle = 0.0
        self.front_maxAngle = 0.0
        self.front_angIncrement = 0.0
        self.front_minRange = 0.0
        self.front_maxRange = 0.0
        self.front_ranges = []
        self.front_FoV = 0.0
        self.front_noOfScans = 0.0

        # Data arrays
        self.base_r_values = []
        self.base_theta_values = []
        self.roi1_ranges = []
        self.roi2_ranges = []
        self.centroids = []
        self.closestPoints = []
        self.rep_points = []

        # Vectors
        self.rep_resultant = Point(x=0.0, y=0.0, z=0.0)
        self.ref_signal = Point(x=0.0, y=0.0, z=0.0)
        self.vfinal_signal = Point(x=0.0, y=0.0, z=0.0)

        # Joy state
        self.deadman_switch = 0
        self.autonomy_switch = 0
        self.vfinal_joy = Joy()

        # TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ----------------------------------------------------------------------------
        # Publishers
        # ----------------------------------------------------------------------------
        self.pcld2_pub = self.create_publisher(PointCloud2, '/transformed_pcld2', 10)
        self.marker_pub = self.create_publisher(Marker, '/obstacle_marker', 10)
        self.centroid_pub = self.create_publisher(Marker, '/centroid_marker', 10)
        self.scan1_pub = self.create_publisher(LaserScan, '/ROI1_laserScan', 10)
        self.scan2_pub = self.create_publisher(LaserScan, '/ROI2_laserScan', 10)
        self.repForce_pub = self.create_publisher(MarkerArray, '/rep_marker', 10)
        self.resForce_pub = self.create_publisher(Marker, '/res_marker', 10)
        self.refSignal_pub = self.create_publisher(Marker, '/refSignal_marker', 10)
        self.vfinal_marker_pub = self.create_publisher(Marker, '/vfinal_marker', 10)

        output_joy_topic = self.get_parameter('output_joy_topic').value
        self.vfinal_joy_pub = self.create_publisher(Joy, output_joy_topic, 10)

        # ----------------------------------------------------------------------------
        # Subscribers
        # ----------------------------------------------------------------------------
        scan_topic = self.get_parameter('scan_topic').value
        joy_topic = self.get_parameter('joy_topic').value

        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.frontScan_callback,
            10
        )
        self.joy_sub = self.create_subscription(
            Joy,
            joy_topic,
            self.joy_bs_callback,
            10
        )

        # ----------------------------------------------------------------------------
        # Timer replacing ROS 1 while loop
        # ----------------------------------------------------------------------------
        self.main_timer = self.create_timer(1.0 / self.loop_frequency, self.main_loop)

        # One-time info
        self.displayLaserSpecs_once = False

    # =============================================================================
    # Callbacks
    # =============================================================================
    def frontScan_callback(self, scan_msg: LaserScan):
        self.front_minAngle = scan_msg.angle_min
        self.front_maxAngle = scan_msg.angle_max
        self.front_angIncrement = scan_msg.angle_increment
        self.front_FoV = self.front_maxAngle - self.front_minAngle
        if self.front_angIncrement != 0.0:
            self.front_noOfScans = math.degrees(self.front_FoV) / math.degrees(self.front_angIncrement)
        else:
            self.front_noOfScans = 0.0
        self.front_minRange = scan_msg.range_min
        self.front_maxRange = scan_msg.range_max
        self.front_ranges = list(scan_msg.ranges)

    def joy_bs_callback(self, joy_bs_msg: Joy):
        self.vfinal_joy = joy_bs_msg
        if len(joy_bs_msg.axes) >= 2:
            self.ref_signal.x = joy_bs_msg.axes[1]
            self.ref_signal.y = joy_bs_msg.axes[0]
        else:
            self.ref_signal.x = 0.0
            self.ref_signal.y = 0.0
        self.ref_signal.z = 0.0
        if len(joy_bs_msg.buttons) > 5:
            self.deadman_switch = joy_bs_msg.buttons[5]
        else:
            self.deadman_switch = 0

    def getAngle(self, range_index: int) -> float:
        return self.front_minAngle + (range_index * self.front_angIncrement)

    def getTransform(self, target_frame: str, source_frame: str):
        try:
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                now,
                timeout=Duration(seconds=0.5)
            )
            trans = (
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            )
            rot = (
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            )
            return trans, rot
        except Exception as e:
            self.get_logger().error(
                f'Transform lookup from {source_frame} to {target_frame} failed: {e}'
            )
            return None, None

    def publish_transformed_pointCloud(self):
        trans, rot = self.getTransform('base_link', 'lidar_frame')
        if trans is None or rot is None:
            self.get_logger().warn('Frame transformation skipped.')
            return

        # Build transform matrix ONCE outside the loop
        rot_wxyz = [rot[3], rot[0], rot[1], rot[2]]
        Rq = quat2mat(rot_wxyz)
        Tx = np.eye(4)
        Tx[:3, :3] = Rq
        Tx[:3, 3] = [trans[0], trans[1], trans[2]]

        transformed_points = []
        self.base_r_values = []
        self.base_theta_values = []

        for i, r in enumerate(self.front_ranges):
            if self.front_minRange < r < self.front_maxRange:
                theta = self.getAngle(i)
                x_lidar = r * math.cos(theta)
                y_lidar = r * math.sin(theta)
                lidar_point = np.array([x_lidar, y_lidar, 0.0, 1.0])
                base_point = np.dot(Tx, lidar_point)
                x_base = float(base_point[0])
                y_base = float(base_point[1])
                z_base = float(base_point[2])
                transformed_points.append((x_base, y_base, z_base))
                r_base = math.sqrt(x_base ** 2 + y_base ** 2)
                theta_base = math.atan2(y_base, x_base)
                self.base_r_values.append(r_base)
                self.base_theta_values.append(theta_base)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'base_link'
        pcld2_msg = point_cloud2.create_cloud_xyz32(header, transformed_points)
        self.pcld2_pub.publish(pcld2_msg)

    def displayLaserSpecs(self):
        self.get_logger().info('*** KAIROS+ Front Laser Specifications ***')
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
        cPoint = Point()
        min_distance = 5.5
        for point in points:
            distance = math.sqrt(point.x ** 2 + point.y ** 2)
            if distance <= min_distance:
                min_distance = distance
                cPoint = point
        self.closestPoints.append(cPoint)

    def compute_centroid(self, points):
        n = len(points)
        if n == 0:
            return Point(x=0.0, y=0.0, z=0.0)
        sum_x = sum(point.x for point in points)
        sum_y = sum(point.y for point in points)
        g_x = (1.0 / n) * sum_x
        g_y = (1.0 / n) * sum_y
        centroid = Point(x=g_x, y=g_y, z=0.0)
        self.centroids.append(centroid)
        return centroid

    def publish_centroids(self):
        centroid_marker = Marker()
        centroid_marker.header.frame_id = 'base_link'
        centroid_marker.header.stamp = self.get_clock().now().to_msg()
        centroid_marker.type = Marker.POINTS
        centroid_marker.scale.x = 0.05
        centroid_marker.scale.y = 0.05
        centroid_marker.color.r = 0.0
        centroid_marker.color.g = 1.0
        centroid_marker.color.b = 0.0
        centroid_marker.color.a = 1.0
        centroid_marker.points = self.centroids
        centroid_marker.pose.orientation.w = 1.0
        centroid_marker.lifetime = Duration(seconds=self.marker_lifetime).to_msg()
        self.centroid_pub.publish(centroid_marker)

    def compute_convexhull(self, points):
        if len(points) < 3:
            return points
        points_np = np.array([(p.x, p.y) for p in points])
        cvxhull = ConvexHull(points_np)
        cvxhull_points = [Point(x=p[0], y=p[1], z=0.0) for p in points_np[cvxhull.vertices]]
        return cvxhull_points

    def compute_roi1(self, points, centroid):
        for point in points:
            dir1_x = point.x - centroid.x
            dir1_y = point.y - centroid.y
            length1 = math.sqrt(dir1_x ** 2 + dir1_y ** 2)
            if length1 == 0.0:
                continue
            unit1_x = dir1_x / length1
            unit1_y = dir1_y / length1
            roi1_x = point.x + (unit1_x * self.rho_0)
            roi1_y = point.y + (unit1_y * self.rho_0)
            roi1_r = math.sqrt(roi1_x ** 2 + roi1_y ** 2)
            roi1_theta = math.atan2(roi1_y, roi1_x)
            if self.front_minAngle <= roi1_theta <= self.front_maxAngle:
                index = int((roi1_theta - self.front_minAngle) / self.front_angIncrement)
                if 0 <= index < len(self.roi1_ranges):
                    self.roi1_ranges[index] = roi1_r

    def compute_roi2(self, points, centroid):
        for point in points:
            dir2_x = centroid.x - point.x
            dir2_y = centroid.y - point.y
            length2 = math.sqrt(dir2_x ** 2 + dir2_y ** 2)
            if length2 == 0.0:
                continue
            unit2_x = dir2_x / length2
            unit2_y = dir2_y / length2
            roi2_x = point.x + (unit2_x * self.rho_0)
            roi2_y = point.y + (unit2_y * self.rho_0)
            roi2_r = math.sqrt(roi2_x ** 2 + roi2_y ** 2)
            roi2_theta = math.atan2(roi2_y, roi2_x)
            if self.front_minAngle <= roi2_theta <= self.front_maxAngle:
                index = int((roi2_theta - self.front_minAngle) / self.front_angIncrement)
                if 0 <= index < len(self.roi2_ranges):
                    self.roi2_ranges[index] = roi2_r

    def publish_roi1(self):
        scan1 = LaserScan()
        scan1.header.frame_id = 'base_link'
        scan1.header.stamp = self.get_clock().now().to_msg()
        scan1.angle_min = self.front_minAngle
        scan1.angle_max = self.front_maxAngle
        scan1.angle_increment = self.front_angIncrement
        scan1.range_min = self.front_minRange
        scan1.range_max = self.front_maxRange
        scan1.ranges = self.roi1_ranges
        self.scan1_pub.publish(scan1)

    def publish_roi2(self):
        scan2 = LaserScan()
        scan2.header.frame_id = 'base_link'
        scan2.header.stamp = self.get_clock().now().to_msg()
        scan2.angle_min = self.front_minAngle
        scan2.angle_max = self.front_maxAngle
        scan2.angle_increment = self.front_angIncrement
        scan2.range_min = self.front_minRange
        scan2.range_max = self.front_maxRange
        scan2.ranges = self.roi2_ranges
        self.scan2_pub.publish(scan2)

    def publish_potentialFields(self):
        rep_markerArray = MarkerArray()
        self.rep_points = []

        for i, cPoint in enumerate(self.closestPoints):
            rho_xy = math.sqrt(cPoint.x ** 2 + cPoint.y ** 2)
            if rho_xy == 0.0:
                continue

            rep_unit_x = -cPoint.x / rho_xy
            rep_unit_y = -cPoint.y / rho_xy
            normFactor0 = NORM_FACTOR_0

            if self.rho_cap <= rho_xy <= self.rho_0:
                rep_x = rep_unit_x * (self.k_rep / (rho_xy ** 2)) * ((1.0 / rho_xy) - (1.0 / self.rho_0)) / normFactor0
                rep_y = rep_unit_y * (self.k_rep / (rho_xy ** 2)) * ((1.0 / rho_xy) - (1.0 / self.rho_0)) / normFactor0
            elif rho_xy < self.rho_cap:
                rep_x = rep_unit_x * (self.k_rep / (self.rho_cap ** 2)) * ((1.0 / self.rho_cap) - (1.0 / self.rho_0)) / normFactor0
                rep_y = rep_unit_y * (self.k_rep / (self.rho_cap ** 2)) * ((1.0 / self.rho_cap) - (1.0 / self.rho_0)) / normFactor0
            else:
                rep_x = 0.0
                rep_y = 0.0

            rep_point = Point(x=rep_x, y=rep_y, z=0.0)
            self.rep_points.append(rep_point)

            rep_marker = Marker()
            rep_marker.header.frame_id = 'base_link'
            rep_marker.header.stamp = self.get_clock().now().to_msg()
            rep_marker.type = Marker.ARROW
            rep_marker.id = i
            rep_marker.ns = 'repulsive_forces'
            rep_marker.scale.x = 0.05
            rep_marker.scale.y = 0.1
            rep_marker.scale.z = 0.15
            rep_marker.color.r = 0.8
            rep_marker.color.g = 0.2
            rep_marker.color.b = 0.8
            rep_marker.color.a = 1.0
            rep_marker.pose.orientation.w = 1.0
            rep_marker.points = [Point(x=0.0, y=0.0, z=0.0), rep_point]
            rep_marker.lifetime = Duration(seconds=self.marker_lifetime).to_msg()
            rep_markerArray.markers.append(rep_marker)

        self.repForce_pub.publish(rep_markerArray)

    def compute_resultant(self, points):
        resultant = Point(x=0.0, y=0.0, z=0.0)
        for point in points:
            resultant.x += point.x
            resultant.y += point.y
            resultant.z += point.z
        return resultant

    def publish_repulsiveResultant(self):
        if self.rep_from == ALL_OBSTACLES:
            rep_vectors = self.rep_points
            self.rep_resultant = self.compute_resultant(rep_vectors)
        else:  # CLOSEST_OBSTACLE
            if self.rep_points:
                closest_rep_point = self.rep_points[0]
                closest_rep_magnitude = math.sqrt(
                    closest_rep_point.x ** 2 + closest_rep_point.y ** 2
                )
                for rep_point in self.rep_points:
                    rep_magnitude = math.sqrt(
                        rep_point.x ** 2 + rep_point.y ** 2
                    )
                    if rep_magnitude > closest_rep_magnitude:
                        closest_rep_point = rep_point
                        closest_rep_magnitude = rep_magnitude
                self.rep_resultant = closest_rep_point
            else:
                self.rep_resultant = Point(x=0.0, y=0.0, z=0.0)

        if len(self.rep_points) == 0:
            normFactor1 = NORM_FACTOR_1
        else:
            normFactor1 = NORM_FACTOR_1

        self.rep_resultant.x /= normFactor1
        self.rep_resultant.y /= normFactor1
        self.rep_resultant.z /= normFactor1

        resultant_marker = Marker()
        resultant_marker.header.frame_id = 'base_link'
        resultant_marker.header.stamp = self.get_clock().now().to_msg()
        resultant_marker.type = Marker.ARROW
        resultant_marker.ns = 'repulsive_resultant'
        resultant_marker.scale.x = 0.05
        resultant_marker.scale.y = 0.1
        resultant_marker.scale.z = 0.15
        resultant_marker.color.r = 1.0
        resultant_marker.color.g = 0.0
        resultant_marker.color.b = 0.0
        resultant_marker.color.a = 1.0
        resultant_marker.pose.orientation.w = 1.0
        resultant_marker.points = [Point(x=0.0, y=0.0, z=0.0), self.rep_resultant]
        self.resForce_pub.publish(resultant_marker)

    def publish_referenceSignal(self):
        reference_marker = Marker()
        reference_marker.header.frame_id = 'base_link'
        reference_marker.header.stamp = self.get_clock().now().to_msg()
        reference_marker.type = Marker.ARROW
        reference_marker.scale.x = 0.05
        reference_marker.scale.y = 0.1
        reference_marker.scale.z = 0.15
        reference_marker.color.r = 0.0
        reference_marker.color.g = 0.0
        reference_marker.color.b = 1.0
        reference_marker.color.a = 1.0
        reference_marker.pose.orientation.w = 1.0
        reference_marker.points = [Point(x=0.0, y=0.0, z=0.0), self.ref_signal]
        self.refSignal_pub.publish(reference_marker)

    def publish_finalVelocity_marker(self):
        rep_magnitude = math.sqrt(self.rep_resultant.x**2 + self.rep_resultant.y**2)
        ref_magnitude = math.sqrt(self.ref_signal.x**2 + self.ref_signal.y**2)

        # Scale rep_resultant so that at (rho_cap + stop_distance),
        # rep magnitude exactly equals ref magnitude → robot stops
        stop_threshold = self.rho_cap + self.stop_distance  # e.g. 0.75 + 0.1 = 0.85m

        # Find closest obstacle distance
        min_rho = float('inf')
        for cPoint in self.closestPoints:
            rho_xy = math.sqrt(cPoint.x**2 + cPoint.y**2)
            if rho_xy < min_rho:
                min_rho = rho_xy

        if min_rho <= stop_threshold and rep_magnitude > 0.0 and ref_magnitude > 0.0:
            # Scale rep vector to exactly match human input magnitude
            scale = ref_magnitude / rep_magnitude
            self.rep_resultant.x *= scale
            self.rep_resultant.y *= scale
            self.get_logger().warn(f'Stop zone reached at {min_rho:.2f}m — rep scaled to match human input')

        # Now combine — rep exactly cancels human input → vfinal = 0
        self.vfinal_signal = self.compute_resultant([self.ref_signal, self.rep_resultant])

        if self.rep_resultant.x == 0.0 and self.rep_resultant.y == 0.0:
            normFactor2 = NORM_FACTOR_2
        else:
            normFactor2 = 2.0

        self.vfinal_signal.x /= normFactor2
        self.vfinal_signal.y /= normFactor2
        self.vfinal_signal.z /= normFactor2

        vfinal_marker = Marker()
        vfinal_marker.header.frame_id = 'base_link'
        vfinal_marker.header.stamp = self.get_clock().now().to_msg()
        vfinal_marker.type = Marker.ARROW
        vfinal_marker.scale.x = 0.05
        vfinal_marker.scale.y = 0.1
        vfinal_marker.scale.z = 0.15
        vfinal_marker.color.r = 0.0
        vfinal_marker.color.g = 1.0
        vfinal_marker.color.b = 0.0
        vfinal_marker.color.a = 1.0
        vfinal_marker.pose.orientation.w = 1.0
        vfinal_marker.points = [Point(x=0.0, y=0.0, z=0.0), self.vfinal_signal]
        self.vfinal_marker_pub.publish(vfinal_marker)

    def publish_vfinal_joy(self):
        # Don't publish until we have a real joy message with enough axes/buttons
        if len(self.vfinal_joy.axes) < 4 or len(self.vfinal_joy.buttons) <= 10:
            return
        
        vfinal_joy_axes = list(self.vfinal_joy.axes)
        if len(vfinal_joy_axes) < 2:
            vfinal_joy_axes = [0.0, 0.0] + list(vfinal_joy_axes[2:])

        vfinal_joy_axes[0] = self.vfinal_signal.y
        vfinal_joy_axes[1] = self.vfinal_signal.x

        new_joy = Joy()
        new_joy.header = self.vfinal_joy.header
        new_joy.axes = vfinal_joy_axes
        new_joy.buttons = list(self.vfinal_joy.buttons)  # pass all buttons through (R1, R2 etc.)
        self.vfinal_joy_pub.publish(new_joy)

    def publish_obstacles(self):
        self.roi1_ranges = [5.5] * len(self.base_r_values)
        self.roi2_ranges = [6.0] * len(self.base_r_values)

        points = []
        markerNumber = 0

        for j in range(len(self.base_r_values)):
            if self.base_r_values[j] < self.marker_threshold_range:
                r = self.base_r_values[j]
                theta = self.base_theta_values[j]
                marker_point = Point()
                marker_point.x = r * math.cos(theta)
                marker_point.y = r * math.sin(theta)
                marker_point.z = 0.0
                points.append(marker_point)
            else:
                if points:
                    marker = Marker()
                    marker.header.frame_id = 'base_link'
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = 'thresholded_laserScan'
                    marker.id = markerNumber
                    marker.type = Marker.LINE_STRIP
                    marker.action = Marker.ADD
                    marker.scale.x = 0.02
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                    marker.pose.orientation.w = 1.0
                    marker.points = points
                    marker.lifetime = Duration(seconds=self.marker_lifetime).to_msg()
                    self.marker_pub.publish(marker)

                    self.record_closestPoint(points)
                    centroid = self.compute_centroid(points)
                    self.compute_roi1(points, centroid)
                    self.compute_roi2(points, centroid)
                    points = []
                    markerNumber += 1

        if points:
            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'thresholded_laserScan'
            marker.id = markerNumber
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.pose.orientation.w = 1.0
            marker.points = points
            marker.lifetime = Duration(seconds=self.marker_lifetime).to_msg()
            self.marker_pub.publish(marker)

            self.record_closestPoint(points)
            centroid = self.compute_centroid(points)
            self.compute_roi1(points, centroid)
            self.compute_roi2(points, centroid)

    # =============================================================================
    # Main loop (timer callback)
    # =============================================================================
    def main_loop(self):
        if not self.displayLaserSpecs_once and self.front_ranges:
            self.displayLaserSpecs()
            self.displayLaserSpecs_once = True

        if not self.front_ranges:
            return

        self.centroids = []
        self.closestPoints = []

        self.publish_transformed_pointCloud()
        self.publish_obstacles()
        self.publish_centroids()
        self.publish_roi1()
        self.publish_roi2()
        self.publish_potentialFields()
        self.publish_repulsiveResultant()
        self.publish_referenceSignal()
        self.publish_finalVelocity_marker()
        self.publish_vfinal_joy()


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