import pygame
from time import sleep

def init():
    pygame.init()
    win = pygame.display.set_mode((400, 400))
    faceRecognition = False


def getKey(keyName):
    ans = False
    for eve in pygame.event.get(): pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, 'K_{}'.format(keyName))
    # print('K_{}'.format(keyName))

    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return ans

def getKeyboardInput(me,faceRecognition):
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    
    if getKey("LEFT"):
        lr = -speed
    elif getKey("RIGHT"):
        lr = speed
    if getKey("UP"):
        fb = speed
    elif getKey("DOWN"):
        fb = -speed
    if getKey("w"):
        ud = speed
    elif getKey("s"):
        ud = -speed
    if getKey("a"):
        yv = -speed
    elif getKey("d"):
        yv = speed
    if getKey("q"):
        me.land()
        sleep(3)
    if getKey("e"):
        me.takeoff()
    if getKey("f"):
        faceRecognition = True
    elif getKey("g"):
        faceRecognition = False
    return [lr, fb, ud, yv], faceRecognition


def main():
    if getKey("LEFT"):
        print("Left key pressed")

    if getKey("RIGHT"):
        print("Right key Pressed")


if __name__ == "__main__":
    init()
    while True:
        main()