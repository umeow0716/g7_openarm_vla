"""List the attached RealSense cameras with their serial and USB speed.

uv run python scripts/realsense_device_list.py
"""

import pyrealsense2 as rs


def check_realsense_usb_speeds():
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("No Intel RealSense devices found.")
        return

    print(f"Checking USB protocol for {len(devices)} connected device(s):\n")

    for i, device in enumerate(devices):
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)

        # Query the USB port speed (returns a string like "3.2", "3.1", "2.1", etc.)
        usb_speed = device.get_info(rs.camera_info.usb_type_descriptor)

        # Check if it starts with '3' (USB 3.0, 3.1, 3.2, etc.)
        is_superspeed = usb_speed.startswith("3")

        print(f"Device {i + 1}: {name}")
        print(f"  - Serial:       {serial}")
        print(f"  - USB Version:  {usb_speed}")

        if is_superspeed:
            print("  - Status:       [OK] SuperSpeed (USB 3.0+)")
        else:
            print("  - Status:       [WARNING] NOT SuperSpeed (Running on USB 2.0)")
            print("                  -> Check the port and the cable are both USB 3.0.")
        print("-" * 40)


if __name__ == "__main__":
    check_realsense_usb_speeds()
