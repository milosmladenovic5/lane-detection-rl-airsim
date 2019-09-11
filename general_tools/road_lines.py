import keyboard
import airsim
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial.distance import euclidean


def capture_road_lines(filepath, car_client=None):
    """
    Constructs a env. map file
    Each line in the file represents a polyline of the road

    :param filepath : (str) - resulting file path
    :param car_client: - AirSim car client
    :return: nothing
    """
    
    if car_client == None:
        car_client = airsim.CarClient()
        car_client.confirmConnection()
        car_client.enableApiControl(False)
    
    print("Capturing road points.. Press \"p\" to capture the current position, or \"p\" to exit.")
    with open(filepath, "a") as f:
        line = []
        while True:  
            if keyboard.is_pressed("p"):
                cp = car_client.simGetGroundTruthKinematics().position
                line.append(cp.x_val)
                line.append(cp.y_val)
                line.append(cp.z_val)
                print("Captured!")
                if len(line) == 6:
                    f.write(f"{line[0]}\t{line[1]}\t{line[2]}\t{line[3]}\t{line[4]}\t{line[5]}\n")
                    #print(f"{line} writen to {filepath}")
                    print("Line writen to file.")
                    line = []
                    
            elif keyboard.is_pressed("q"):
                break

            time.sleep(0.12)

def capture_road_lines_cont(filepath, min_dist, car_client=None):
    """
    Constucs env. map file
    Captures point based on minimum distance parameter
    
    :param filepath: (str) - resulting file path
    :param min_dist: (float) - minimum distance from another point
    :param car_client: - AirSim car client
    :return: nothing
    """
    if car_client == None:
        car_client = airsim.CarClient()
        car_client.confirmConnection()
        car_client.enableApiControl(False)

    line = []
    
    print("Capture started... Move along the track to capture points.")
    cp = car_client.simGetGroundTruthKinematics().position
    last_position = np.array([cp.x_val, cp.y_val, cp.z_val])
    line.append(cp.x_val)
    line.append(cp.y_val)
    line.append(cp.z_val)
    print("Captured starting point.")    
    
    with open(filepath, "a") as f:
        while True:  
            cp = car_client.simGetGroundTruthKinematics().position
            cp_np = np.array([cp.x_val, cp.y_val, cp.z_val])
            if euclidean(last_position, cp_np) >= min_dist:
                last_position = cp_np 
                
                line.append(cp.x_val)
                line.append(cp.y_val)
                line.append(cp.z_val)
                print("Captured!")
                
                if len(line) == 6:
                    f.write(f"{line[0]}\t{line[1]}\t{line[2]}\t{line[3]}\t{line[4]}\t{line[5]}\n")
                    print("Line writen to file.")
                    line = []
                    # append the same point again as the start of the new line
                    line.append(cp.x_val)
                    line.append(cp.y_val)
                    line.append(cp.z_val)
                
            if keyboard.is_pressed("q"):
                break

def load_road_lines(filepath):
    """
    Loads road lines from the file
    Expected format: line_start - line_end x, y, z - tab separated

    :param filepath: (str) - file to read from
    :return: list of line points (start, end)
    """
    road_lines = []
    with open(filepath, "r") as f:
        for line in f:
            pvs = line.split("\t")
            first = np.array([float(pvs[0]), float(pvs[1]), float(pvs[2])])
            second = np.array([float(pvs[3]), float(pvs[4]), float(pvs[5])])
            road_lines.append((first, second))
            
    return road_lines


def draw_road_lines(road_lines, car_position=None):
    """
    Displays road lines with car position
    
    :param  road_lines: (list of tuples) road lines list (start, end)
    :param  car_position: AirSim car position
    :return: nothing
    """
    fig = plt.figure(figsize=(15, 15))
    for l in road_lines:
        plt.plot([l[0][0], l[1][0]], [l[0][1], l[1][1]], "k-", lw=2)
        
    if car_position != None:
        plt.plot([car_position.x_val], [car_position.y_val], "bo")
    
    plt.show()

def draw_road_lines_3D(road_lines, car_position=None):
    """
    Displays road lines with car position in 3D
    
    :param  road_lines: (list of tuples) road lines list (start, end)
    :param  car_position: AirSim car position
    :return: nothing
    """
    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca(projection="3d")

    for l in road_lines:
        ax.plot([l[0][0], l[1][0]], [l[0][1], l[1][1]], [l[0][2], l[1][2]], "k-", lw=2)

    if car_position != None:
        ax.plot([car_position.x_val], [car_position.y_val], [car_position.z_val], "bo")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Capture road lines to file.")
    parser.add_argument("file", help="Destination file.")
    parser.add_argument("-c", action="store_true", help="Minimum distance based capture.")
    parser.add_argument("min_dist", type=float, nargs="?", default=2.0, help="Minimum distance between two points.")

    args = parser.parse_args()
    if args.c:
        capture_road_lines_cont(args.file, args.min_dist)
    else:
        capture_road_lines(args.file)
