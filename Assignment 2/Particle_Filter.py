import pygame
from pygame.locals import *
import random, math
from numpy import dot, sum, tile, linalg, array, random, eye, zeros, diag, transpose, mean, var, cov, size
from numpy.linalg import inv, norm
import matplotlib.pyplot as plt
import itertools as it
import numpy as np
from video import make_video
#................... Initialize all of the Pygame modules ...................#
pygame.init()

#................ . Create the game screen and set it to 600 x 600 pixels.....#
window = pygame.display.set_mode((600, 600), DOUBLEBUF)
screen = pygame.display.get_surface()
pygame.display.flip()
scale = 600
scale_screen = int(scale / 20)

#................... Set a caption to the window ..........................#
pygame.display.set_caption("Particle Filter for 2D Robot")

#................... Creating a tuple to hold the color values ............#
colorRED = (255, 0, 0)
colorBLUE = (0, 0, 255)
colorWHITE = (255,255,255)
colorGRAY = (128,128,128)
colorBLACK = (0, 0, 0)

#................... Draw robot, particle and Landmark onto screen .................#
robot = pygame.Surface((10, 10))
robot.fill(colorRED)
particle = pygame.Surface((3.5, 3.5))
particle.fill(colorBLUE)
Landmark = pygame.Surface((10, 10))
Landmark.fill(colorWHITE)
L = array([[10, 10]]).T

#................... Creating surface contains transparency .................#
image = pygame.Surface((600,600), pygame.SRCALPHA, 32).convert_alpha()

#................... Defining the initial values ............................#
x0 = 8
y0 = 10
X = np.array([[x0, y0]]).T
P = np.zeros(2)
M = 100  # number of samples
bins = int(M / 10)

#................... Defining parameters of motion model .....................#
radius = 0.1
r = 0.1
rL=0.2
speed = 0.1
T = float(1/8)
ur = 4.75
ul = 5.25
U1 = np.array([[((r/2)* (ur+ul)),((r/2)* (ur+ul))]]).T

A = np.eye(2)
Wx=0.1
Wy=0.15
Q = np.diag((Wx, Wy))
wphi = 0.01
theta = np.pi / 2.0
thetaprev = 0
U = array([[radius * ((ur + ul) / 2.0) * math.cos(theta), radius * ((ur + ul) / 2.0) * math.sin(theta)]]).T

#................... Defining parameters of measurement ....................#
rx= 0.05
ry= 0.075
R = np.diag((rx, ry))
C = np.diag([1, 1])
dist_norm = 2.0


#................... Defining the motion model and sampling function....................#

def sample(X_prev, U1):
    for i in range(M):
        X = X_prev
        process_noise = np.array([[np.random.normal(0, Wx), np.random.normal(0, Wy)]]).T
        X_dot = (U1 + process_noise).reshape(2, )
        X = dot(A, X).reshape(2, ) + (T * X_dot)
    return X

#................... Create bins with boundaries ....................#
def create_bins(x_dist, y_dist):
    X_max = max(x_dist)
    X_min = min(x_dist)
    Y_max = max(y_dist)
    Y_min = min(y_dist)
    max_min = (X_max, X_min, Y_max, Y_min)
    delta = [(X_max - X_min) / 10, (Y_max - Y_min) / 10]
    return max_min, delta


def create_dist(x_dist, y_dist, delta, *max_min):
    X_max, X_min, Y_max, Y_min = max_min
    Probab_x = [0] * bins
    Probab_y = [0] * bins
    p = []
    k_list = []
    for i in range(M):
        kx = int(np.ceil((x_dist[i] - X_min) / delta[0]))
        ky = int(np.ceil((y_dist[i] - Y_min) / delta[1]))
        if x_dist[i] == X_min:
            Probab_x[0] += 1
            kx = 1
        else:
            Probab_x[kx - 1] += 1
        if y_dist[i] == Y_min:
            Probab_y[0] += 1
            ky = 1
        else:
            Probab_y[ky - 1] += 1
        k_list.append((kx - 1, ky - 1))
        p = list(zip(Probab_x, Probab_y))
    return p, k_list

#................... Obtaining probabilities ......................#
def prob(x_p, y_p, p_dist, delta, *max_min):
    X_max, X_min, Y_max, Y_min = max_min
    p = []
    for i in range(M):
        kx = (x_p[i] - X_min) / delta[0]
        ky = (y_p[i] - Y_min) / delta[1]
        if (kx < 0) or (kx > 10):
            Probab_x = 0
        else:
            kx = int(np.ceil(kx))
            if kx == 0:
                Probab_x = p_dist[0][0]
            else:
                Probab_x = p_dist[kx - 1][0]
        if (ky < 0) or (ky > 10):
            Probab_y = 0
        else:
            ky = int(np.ceil(ky))
            if ky == 0:
                Probab_y = p_dist[0][1]
            else:
                Probab_y = p_dist[ky - 1][1]
        prob = (Probab_x, Probab_y)
        p.append(prob)
    return p

#................... Defining the motion equations and sampling ...............#
Linear = True
def correct(pred):
    correct_samples = []
    for i in range(M):
        if Linear is True:
            n = np.array([np.random.normal(0, rx), np.random.normal(0, ry)]).T
            Z = np.dot(C, pred[i]).reshape(2, ) + n
            correct_samples.append(Z)
    return correct_samples

#..................... Assigning weight ........................#
def get_weights(prob_p, k_p, prob_c, k_c):
    weight = []
    for i in range(M):
        weight_x = (prob_c[i][0]) / (prob_p[k_p[i][0]][0])
        weight_y = (prob_c[i][1]) / (prob_p[k_p[i][1]][1])
        weight.append((weight_x, weight_y))
    return weight

#..................... Resample the posterior ........................#
def resample(weight, x_prior, y_prior):
    weight_x = [x[0] for x in weight]
    weight_y = [y[1] for y in weight]
    beta, X_post = [], []
    sumx = np.sum(weight_x)
    sumy = np.sum(weight_y)
    for i in range(M):
        beta.append(((np.sum(weight_x[0:i + 1]) / sumx), (np.sum(weight_y[0:i + 1]) / sumy)))

    rhox = np.random.uniform(0, 1)
    rhoy = np.random.uniform(0, 1)
    index_x = [0] * M
    index_y = [0] * M
    for i in range(M):
        indx = len(list(it.takewhile(lambda x: x < rhox, [x[0] for x in beta])))
        x_new = x_prior[indx]
        indy = len(list(it.takewhile(lambda x: x < rhoy, [y[1] for y in beta])))
        y_new = y_prior[indy]
        X_post.append((x_new, y_new))
        rhox += 1 / M
        rhoy += 1 / M
        if rhox >= 1:
            rhox = random.uniform(0, 1)
        if rhoy >= 1:
            rhoy = random.uniform(0, 1)
        index_x[indx] += 1
        index_y[indy] += 1
    index = list(zip(index_x, index_y))
    return beta, X_post, index

#..................... Setting the noise condition ........................#
Noise = True
if Noise == False:
    wphi, rx, ry = 0, 0, 0

#..................... Initializing the particles ........................#
i = 0
posterior = []
predict_samples = []
for i in range(M):
    posterior.append(((np.random.normal(x0, 2)), (np.random.normal(y0, 2))))

theta_list = []
theta_prev = [theta] * M
theta_prev_n = theta
X_n = X

#................. Initiate the video generator .........................#
save_screen = make_video(screen)
video = False  # at start: video not active

#................... Create a loop that will keep the game running ..........#
true_line = [(x0 * scale_screen, y0 * scale_screen), (x0, y0)]
cov_size = 20
cov_scale = cov_size / scale_screen
running = True
screen.fill((colorBLACK))
screen.blit(Landmark, (int(scale / 2), int(scale / 2)))


while running:

    screen.fill((colorBLACK), (0, scale // 6, scale, scale))

    for i, x in enumerate(posterior):
        theta_noise = random.normal(0, wphi)
        theta_dot = (r / rL) * ((ur - ul) * (radius / rL)) + theta_noise
        theta = (T * theta_dot) + theta_prev[i]
        U = array([[r * ((ur + ul) / 2.0) * math.cos(theta), r * ((ur + ul) / 2.0) * math.sin(theta)]]).T
        theta_list.append(theta)
        predict_samples.append(sample(x, U))

    #............... Defining the ground truth ......................#
    theta_dot_n = (r / rL) * ((ur - ul) * (radius / rL))
    theta_n = (T * theta_dot_n) + theta_prev_n
    U_n = array([r * ((ur + ul) / 2.0) * math.cos(theta_n), r * ((ur + ul) / 2.0) * math.sin(theta_n)]).T
    X_n = X_n.reshape(2, ) + (T * U_n)  # Xk equation 2 from HW2
    theta_prev_n = theta_n

    #.................... Sampling theta ...............................#
    theta_prev = theta_list
    theta_list = []
    x_pred = [x[0] for x in predict_samples]
    y_pred = [y[1] for y in predict_samples]
    max_min, delta = create_bins(x_pred, y_pred)
    prob_p, k_p = create_dist(x_pred, y_pred, delta, *max_min)

    correct_samples = correct(predict_samples)
    x_cor = [x[0] for x in correct_samples]
    y_cor = [y[1] for y in correct_samples]
    max_min_c, delta_c = create_bins(x_cor, y_cor)
    prob_c, k_c = create_dist(x_cor, y_cor, delta_c, *max_min_c)

    prob_post = prob(x_pred, y_pred, prob_c, delta_c, *max_min_c)
    weights = get_weights(prob_p, k_p, prob_post, k_c)
    beta, posterior, indices = resample(weights, x_pred, y_pred)
    x_post = [x[0] for x in posterior]
    y_post = [y[1] for y in posterior]

    indx = [x[0] for x in indices]
    indy = [y[1] for y in indices]
    max_x = max(indx)
    i_xmax = [i for i, j in enumerate(indx) if j == max_x]
    max_y = max(indy)
    i_ymax = [i for i, j in enumerate(indy) if j == max_y]
    avgx = 0
    for i in i_xmax:
        avgx += x_pred[i]
    avgx = avgx / len(i_xmax)
    avgy = 0
    for i in i_ymax:
        avgy += y_pred[i]
    avgy = avgy / len(i_ymax)
    X = array([[avgx, avgy]]).T

    if theta < 0:
        theta = math.pi * 2.0


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
            video = not video   # toggle video on/off by clicking 'v' on keyboard #


    predict_samples = []
    dist = X - L
    dist = dist[:, 0]
    dist_norm = norm(dist, ord=2)

    for i in range(M):
        screen.blit(particle, (int(posterior[i][0] * scale_screen), int(posterior[i][1] * scale_screen)))

    screen.blit(robot, (int(X_n[0] * scale_screen), int(X_n[1] * scale_screen)))
    screen.fill((colorBLACK), (0, scale // 12, scale, scale // 6))
    screen.blit(Landmark, (int(scale / 2), int(scale / 2)))

    pygame.display.update()
    if video:
        next(save_screen)  # call the generator
        print("IN main")  # delete, just for demonstration
