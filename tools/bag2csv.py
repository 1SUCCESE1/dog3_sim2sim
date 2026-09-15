#!/usr/bin/env python3
"""Convert a ROS 2 bag (recorded from the dog3 sim2sim) into CSVs PlotJuggler can
open directly.

PlotJuggler's rosbag2 loader crashes on these bags (std::out_of_range inside
TimeseriesBase::TimeCompare), so converting to CSV is the reliable path.

Usage:
  source /opt/ros/humble/setup.bash
  python3 tools/bag2csv.py <bag_dir> [out_dir]

Writes <out_dir>/joint_states.csv, imu.csv, model_pose.csv
(default out_dir: <bag_dir>_csv)
"""

import csv
import os
import sys

from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState, Imu
try:
    from gazebo_msgs.msg import ModelStates
except ImportError:  # gazebo not installed — pose columns just stay empty
    ModelStates = None


def main():
    bag = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else bag.rstrip('/') + '_csv'
    os.makedirs(out, exist_ok=True)

    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag, storage_id='sqlite3'), ConverterOptions('', ''))

    joints_t, joints_names = [], None
    imu_rows, pose_rows = [], []
    t0 = None
    while reader.has_next():
        topic, data, ts = reader.read_next()
        if t0 is None:
            t0 = ts
        t = (ts - t0) / 1e9

        if topic == '/joint_states':
            m = deserialize_message(data, JointState)
            joints_names = list(m.name)
            joints_t.append([t] + list(m.position))
        elif topic == '/imu_sensor_broadcaster/imu':
            m = deserialize_message(data, Imu)
            o = m.orientation
            imu_rows.append([t, o.x, o.y, o.z, o.w,
                             m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
        elif topic == '/model_states' and ModelStates is not None:
            m = deserialize_message(data, ModelStates)
            if 'dog3_description' in list(m.name):
                i = list(m.name).index('dog3_description')
                p, tw = m.pose[i], m.twist[i]
                pose_rows.append([t, p.position.x, p.position.y, p.position.z,
                                  tw.linear.x, tw.linear.y, tw.linear.z])

    def dump(name, header, rows):
        path = os.path.join(out, name)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"  {path}  ({len(rows)} rows)")

    if joints_names:
        dump('joint_states.csv', ['t'] + joints_names, joints_t)
    if imu_rows:
        dump('imu.csv', ['t', 'qx', 'qy', 'qz', 'qw', 'wx', 'wy', 'wz'], imu_rows)
    if pose_rows:
        dump('model_pose.csv', ['t', 'x', 'y', 'z', 'vx', 'vy', 'vz'], pose_rows)


if __name__ == '__main__':
    main()
