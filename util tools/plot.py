import numpy as np
import matplotlib.pyplot as plt

theta1 = np.array([90, 33.8, -29.4, 0, -19.3])
theta2 = np.array([0, 75.5, 112, 90, 75.5])
t = np.linspace(0, 2, 100)

def q(q0, qf, t):
    return q0 + 3*(qf - q0)/4 * t**2 - 2*(qf - q0)/8 * t**3
def q_dot(q0, qf, t):
    return 6*(qf - q0)/4 * t - 6*(qf - q0)/8 * t**2

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.title("q")
for i in range(4):
    plt.plot(t + 2*i, q(theta1[i], theta1[i+1], t), "b")
    plt.plot(t + 2*i, q(theta2[i], theta2[i+1], t), "r")
plt.ylabel("Position (deg)")
plt.legend(["Joint 1", "Joint 2"])
plt.grid()

plt.subplot(2, 2, 3)
plt.title("qdot")
for i in range(4):
    plt.plot(t + 2*i, q_dot(theta1[i], theta1[i+1], t), "b")
    plt.plot(t + 2*i, q_dot(theta2[i], theta2[i+1], t), "r")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (deg/sec)")
plt.legend(["Joint 1", "Joint 2"])
plt.grid()

plt.subplot(1, 2, 2)
plt.title("end-effector")
for i in range(4):
    q1 = np.radians(q(theta1[i], theta1[i+1], t))
    q2 = np.radians(q(theta2[i], theta2[i+1], t))
    x = 2*np.cos(q1) + 2*np.cos(q1 + q2)
    y = 2*np.sin(q1) + 2*np.sin(q1 + q2)
    plt.plot(x, y, "b")
xx = [0, 1, 2, 2, 3]
yy = [4, 3, 1, 2, 1]
plt.plot(xx, yy, "x")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid()

plt.show()