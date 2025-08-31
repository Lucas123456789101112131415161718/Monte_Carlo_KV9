import numpy as np
import matplotlib.pyplot as plt
import time

# all values are in cm
ly = 40.1 #y length of the platers
lx = 100.3 #x length of the plates
lz = 1.318 #the width in z of a single plate
dist_plates = [3.492, 3.402, 3.754]
#dist_plates = 2 #the distance between the z-center-axis of each plate
dist_plates2 = [0, 2.174, 2.084, 2.436]

# Simulation parameters
Event_N = 1000000



def get_random_theta():
    x = np.random.uniform(0,1)
    return np.power(x,1/3)

def init_start_pos(plate0):
    ypos = np.random.uniform(0,ly)
    xpos = np.random.uniform(0,lx)
    phi = np.random.uniform(0,2*np.pi)
    cos_theta = get_random_theta()
    return (plate0,xpos,ypos,phi,np.arccos(cos_theta))

def init_start_pos_3D():
    ypos = np.random.uniform(0,ly)
    xpos = np.random.uniform(0,lx)
    zpos = np.random.uniform(0,4*lz+sum(dist_plates2))
    phi = np.random.uniform(0,2*np.pi)
    cos_theta = get_random_theta()
    r = np.tan(np.arccos(cos_theta))*zpos
    x_0 = xpos - r * np.cos(phi)
    y_0 = ypos - r * np.sin(phi)
    return (x_0,y_0,phi,np.arccos(cos_theta))

def check_which_pierced_2D(plate0,ypos,xpos,phi,theta):
    plates = [0,1,2,3]
    output = [False, False, False, False]
    for plate in plates:
        dist = 0
        if plate > plate0:
            dist = sum(dist_plates[plate0:plate])
        elif plate < plate0:
            dist = -sum(dist_plates[plate:plate0])
        r = np.tan(theta)*dist
        x = xpos + r*np.cos(phi)
        y = ypos + r*np.sin(phi)
        if x>0 and x<lx and y>0 and y<ly:
            output[plate] = True
    return output

def check_which_pierced_3D(plate0,xpos,ypos,phi,theta):
    plates = [0,1,2,3]
    output = [False, False, False, False]
    for plate in plates:
        dist = 0
        if plate > plate0:
            dist = sum(dist_plates[plate0:plate])
        elif plate < plate0:
            dist = -sum(dist_plates[plate:plate0])
        r_up = np.tan(theta)*dist
        x_up = xpos + r_up*np.cos(phi)
        y_up = ypos + r_up*np.sin(phi)
        r_down = np.tan(theta)*(dist+lz)
        x_down = xpos + r_down*np.cos(phi)
        y_down = ypos + r_down*np.sin(phi)
        if x_up>0 and x_up<lx and y_up>0 and y_up<ly:
            output[plate] = True
        if x_down>0 and x_down<lx and y_down>0 and y_down<ly:
            output[plate] = True
    return output

def check_which_pierced_cont(x_0,y_0,phi,theta):
    plates = [0,1,2,3]
    output = [False, False, False, False]
    for plate in plates:
        r_up = np.tan(theta)*(plate*lz+sum(dist_plates2[:plate+1]))
        r_down = np.tan(theta)*(plate*lz+sum(dist_plates2[:plate+1])+lz)
        x_up = x_0 + r_up*np.cos(phi)
        x_down = x_0 + r_down*np.cos(phi)
        y_up = y_0 + r_up*np.sin(phi)
        y_down = y_0 + r_down*np.sin(phi)
        if x_up>0 and x_up<lx and y_up>0 and y_up<ly:
            output[plate] = True
        if x_down>0 and x_down<lx and y_down>0 and y_down<ly:
            output[plate] = True
    return output

def angle_loss2D_4(n):
    i = 0
    counts = 0
    while i<n:
        pierced = check_which_pierced_2D(*init_start_pos(np.random.randint(0,4)))
        if pierced[0] and pierced[2]:
            i += 1
            if pierced[3]:
                counts += 1
    return counts/n

def angle_loss2D_1(n):
    i = 0
    counts = 0
    while i<n:
        pierced = check_which_pierced_2D(*init_start_pos(np.random.randint(0,4)))
        if pierced[1] and pierced[3]:
            i += 1
            if pierced[0]:
                counts += 1
    return counts/n


def angle_loss_4(n):
    i = 0
    counts = 0
    while i<n:
        pierced = check_which_pierced_3D(*init_start_pos(np.random.randint(0,4)))
        if pierced[0] and pierced[2]:
            i += 1
            if pierced[3]:
                counts += 1
    return counts/n

def angle_loss_1(n):
    i = 0
    counts = 0
    while i<n:
        pierced = check_which_pierced_3D(*init_start_pos(np.random.randint(0,4)))
        if pierced[1] and pierced[3]:
            i += 1
            if pierced[0]:
                counts += 1
    return counts/n

def angleloss_3D(platenum, events):
    i = 0
    counts = 0
    while i<events:
        pierced = check_which_pierced_cont(*init_start_pos_3D())
        if platenum == 1:
            if pierced[1] and pierced[3]:
                i += 1
                if pierced[0]:
                    counts += 1
        if platenum == 4:
            if pierced[0] and pierced[2]:
                i += 1
                if pierced[3]:
                    counts += 1
    return counts/events


loss_1_2D = []
loss_4_2D = []
loss_1_3D = []
loss_4_3D = []
loss_1_cont = []
loss_4_cont = []
for i in range(10):
    ttt=time.time()
    loss_1_2D.append(1-angle_loss2D_1(Event_N))
    loss_4_2D.append(1-angle_loss2D_4(Event_N))
    loss_1_3D.append(1-angle_loss_1(Event_N))
    loss_4_3D.append(1-angle_loss_4(Event_N))
    loss_1_cont.append(1-angleloss_3D(1, Event_N))
    loss_4_cont.append(1-angleloss_3D(4, Event_N))
    print(f"Time round {i}: {time.time()-ttt:.2f}")

mean_loss_1_2D = np.mean(loss_1_2D)
mean_loss_1_3D = np.mean(loss_1_3D)
mean_loss_1_cont = np.mean(loss_1_cont)

mean_loss_4_2D = np.mean(loss_4_2D)
mean_loss_4_3D = np.mean(loss_4_3D)
mean_loss_4_cont = np.mean(loss_4_cont)


print("\nFor Detector 1: ")
print(f"loss_1_2D = {np.mean(loss_1_2D):.4f} +- {np.std(loss_1_2D):.4f}")
print(f"loss_1_3D = {np.mean(loss_1_3D):.4f} +- {np.std(loss_1_3D):.4f}")
print(f"loss_1_cont = {np.mean(loss_1_cont):.4f} +- {np.std(loss_1_cont):.4f}")

print("\nFor Detector 4: ")
print(f"loss_4_2D = {np.mean(loss_4_2D):.4f} +- {np.std(loss_4_2D):.4f}")
print(f"loss_4_3D = {np.mean(loss_4_3D):.4f} +- {np.std(loss_4_3D):.4f}")
print(f"loss_4_cont = {np.mean(loss_4_cont):.4f} +- {np.std(loss_4_cont):.4f}")

x_positions = [1] * 10
x_positions_2 = [2] * 10
x_positions_3 = [3] * 10




mean_loss_1 = [np.mean(loss_1_2D), np.mean(loss_1_3D), np.mean(loss_1_cont)]
std_loss_1 = [np.std(loss_1_2D), np.std(loss_1_3D), np.std(loss_1_cont)]

mean_loss_4 = [np.mean(loss_4_2D), np.mean(loss_4_3D), np.mean(loss_4_cont)]
std_loss_4 = [np.std(loss_4_2D), np.std(loss_4_3D), np.std(loss_4_cont)]




plt.figure(figsize=(10, 6))
plt.scatter(x_positions, loss_1_2D, s=100, label='2D Model')
plt.scatter(x_positions_2, loss_1_3D, s=100, label='3D Model')
plt.scatter(x_positions_3, loss_1_cont, s=100, label='Continuous Model')

plt.errorbar([1, 2, 3], mean_loss_1, yerr=std_loss_1, fmt='o', color='black', capsize=5, label='Mean ± Std')

plt.title('Angle-losses of Detector 1')
plt.xlabel('Model')
plt.ylabel('loss')
plt.xticks([1, 2, 3], ['2D Model', '3D Model', 'Continuous Model'])
plt.legend()
plt.grid(True)
plt.show()
plt.close()



plt.figure(figsize=(10, 6))
plt.scatter(x_positions, loss_4_2D, s=100, label='2D Model')
plt.scatter(x_positions_2, loss_4_3D, s=100, label='3D Model')
plt.scatter(x_positions_3, loss_4_cont, s=100, label='Continuous Model')

plt.errorbar([1, 2, 3], mean_loss_4, yerr=std_loss_4, fmt='o', color='black', capsize=5, label='Mean ± Std')

plt.title('Angle-losses of Detector 4')
plt.xlabel('Model')
plt.ylabel('loss')
plt.xticks([1, 2, 3], ['2D Model', '3D Model', 'Continuous Model'])
plt.legend()
plt.grid(True)
plt.show()
plt.close()