import pygame
from pygame.locals import *
import numpy as np
from numpy.linalg import inv


#................... Initialize all of the Pygame modules ...................#
pygame.init()

#................ . Create the game screen and set it to 600 x 600 pixels.....#
window = pygame.display.set_mode((600, 600), DOUBLEBUF)
screen = pygame.display.get_surface()
pygame.display.flip()
scale = 600

#................... Set a caption to the window ...................#
pygame.display.set_caption("Kalman Filter for 2D Robot")

#................... Creating a tuple to hold the color values ............#
colorRED = (255, 0, 0)
colorBLUE = (0, 0, 255)
colorPINK = (255,200,200)
colorGRAY = (128,128,128)

#................... Draw robot and error line onto screen .................#
robot = pygame.Surface((6, 6), flags=0)
robot.fill(colorPINK)
error = pygame.Surface((6, 6))
error.fill(colorRED)

#................... Creating surface contains transparency .................#
image = pygame.Surface((600,600), pygame.SRCALPHA, 32).convert_alpha()


#................... Defining parameters of motion model .....................#
radius = 0.1
speed = 0.1
T = float(1/8)
ur = 1
ul = 1
U = np.array([[ur,ul]]).T
A = np.eye(2)
Wx=0.1
Wy=0.15
Q = np.diag((Wx, Wy))

#................... Defining the initial values ............................#
x0 = 0
y0 = 0
X = np.array([[x0, y0]]).T
P = np.zeros(2)

#................... Defining parameters of measurement ....................#
rx= 0.05
ry= 0.075
R = np.diag((rx, ry))
C = np.diag([1, 2])

#................... Defining the time update equations (predict) ......#
def kf_predict(X, P, A, Q, U):
    process_noise = np.array([[np.random.normal(0, Wx), np.random.normal(0, Wy)]]).T
    X = np.dot(A, X) + (T*(radius/2 * (U + process_noise)))
    P = np.dot(A, np.dot(P, A.T)) + Q
    return X, P

#................... Defining the measurement update equations (correct) .....#
def kf_correct(X, P, Z, C, R):
    K = np.dot(P, np.dot(C.T, inv(np.dot(C, np.dot(P, C.T))+ R)))
    X = X + np.dot(K, (Z-np.dot(C, X)))
    P = P - np.dot(K, np.dot(C, P))
    return (X, P, K)


#................... Create a loop that will keep the game running ..........#
true_line = [(0, 0), (0, 0)]
i = 0
run = True
screen.fill(colorGRAY)
while run:
    i += 1
    pygame.time.delay(100)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    if i<81:
        X, P = kf_predict(X, P, A, Q, U)

        rx = (X[0][0]*scale)
        ry = (X[1][0]*scale)
        Px = (P[0][0])
        Py = (P[1][1])

        if i%8 == 0:
            n = np.array([[np.random.normal(0, rx), np.random.normal(0, ry)]]).T
            Z = np.dot(C, X) + R
            X, P, K = kf_correct(X, P, Z, C, R)
            true_line[1] = (int(X[0][0]*scale), int(X[1][0]*scale))
            pygame.draw.lines(screen, (255, 0, 0), False, true_line, 3)
            true_line[0] = true_line[1]

        screen.blit(robot, (int(rx), int(ry)))

        try:
            pygame.draw.ellipse(screen, (0, 255, 0), (int(rx-(Px*60)), int(ry-(Py*60)), int(Px*120), int(Py*120)), 2)

        except:
            print(Px, Py)

    pygame.display.update()
