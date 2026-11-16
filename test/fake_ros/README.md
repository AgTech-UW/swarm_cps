A stand-in for rclpy and the message packages, used only by test_nodes.py on
machines without ROS. It mirrors the Humble signatures the nodes use and is
strict where real ROS is strict: unknown keyword arguments raise, message
fields are typed (a float field refuses an int, as rclpy does), publishers
only reach subscribers with the same type and a compatible QoS, and each
ROS domain is a separate bus. It is not installed with the package.
