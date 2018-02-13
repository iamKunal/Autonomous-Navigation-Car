from __future__ import print_function
import RPi.GPIO as GPIO
import time
import pygame
from pygame.locals import *
import cv2

from imutils.video.pivideostream import PiVideoStream

channels = [3, 5, 7, 8]
L_F, L_B, R_F, R_B = channels
SIZE = (200, 200)
FPS = 60
directory = 'data/'

possible_keys = []


def startup():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(channels, GPIO.OUT)



def stop():
    GPIO.output(channels,0)

def fd(secs=None):
    stop()
    # print("FD")
    GPIO.output([L_F, R_F], 1)
    if secs:
        time.sleep(secs)
        stop()

def bk(secs=None):
    stop()
    # print("BK")
    GPIO.output([L_B, R_B], 1)
    if secs:
        time.sleep(secs)
        stop()

def rt(secs=None):
    stop()
    # print("RT")
    GPIO.output([L_F], 1)
    if secs:
        time.sleep(secs)
        stop()
def rt_sharp(secs=None):
    stop()
    GPIO.output([L_F, R_B], 1)
    if secs:
        time.sleep(secs)
        stop()

def lt(secs=None):
    stop()
    # print("LT")
    GPIO.output([R_F], 1)
    if secs:
        time.sleep(secs)
        stop()
def lt_sharp(secs=None):
    stop()
    GPIO.output([L_B, R_F], 1)
    if secs:
        time.sleep(secs)
        stop()

def do_action(key_input):
    cont = None
    if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
        print("Forward Right")
        rt()
        cont = 3
    elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
        print("Forward Left")
        lt()
        cont = 1
    # simple orders
    elif key_input[pygame.K_UP]:
        print("Forward")
        fd()
        cont = 2

    elif key_input[pygame.K_DOWN]:
        # print("Reverse")
        bk()
        cont = 5
    
    elif key_input[pygame.K_RIGHT]:
        print("Right")
        rt_sharp()
        cont = 4

    elif key_input[pygame.K_LEFT]:
        print("Left")
        lt_sharp()
        cont = 0

    elif key_input[pygame.K_x] or key_input[pygame.K_q]:
        print( 'exit')
        stop()
        return False
    return cont

def get_arrow(key_input):
    arrw_dst = (SIZE[0]/2, SIZE[1]/2)
    if key_input[pygame.K_UP] and key_input[pygame.K_RIGHT]:
        # print("Forward Right")
        arrw_dst = (0, SIZE[1])
    elif key_input[pygame.K_UP] and key_input[pygame.K_LEFT]:
        # print("Forward Left")
        arrw_dst = (0, 0)
    # simple orders
    elif key_input[pygame.K_UP]:
        # print("Forward")
        arrw_dst = (0,SIZE[1]/2)

    elif key_input[pygame.K_DOWN]:
        # print("Reverse")
        arrw_dst = (SIZE[0], SIZE[1]/2)
    
    elif key_input[pygame.K_RIGHT]:
        # print("Right")
        arrw_dst = (SIZE[0]/2, SIZE[1])

    elif key_input[pygame.K_LEFT]:
        # print("Left")
        arrw_dst = (SIZE[0]/2, 0)
    return arrw_dst

if __name__ == '__main__':
    startup()

    vs = PiVideoStream(SIZE, FPS).start()
    time.sleep(3.0)
    print("Camera done")
    pygame.init()
    display = pygame.display.set_mode(SIZE, 0)

    possible_keys = [pygame.K_UP, pygame.K_DOWN, pygame.K_RIGHT, pygame.K_LEFT]

    SIZE=SIZE[::-1]
    frame_no =0
    strt = time.time()
    while True:
        frame_no += 1
        cont = True
        original_frame = vs.read()
        frame = cv2.flip(cv2.rotate(original_frame, cv2.ROTATE_90_CLOCKWISE), 0)
        if frame is None:
            print("Camera Error")
            break
        pygame_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(cv2.flip(original_frame, -1), cv2.COLOR_BGR2GRAY)
        arrw_src = (SIZE[0]/2, SIZE[1]/2)
        arrw_dst = (SIZE[0]/2, SIZE[1]/2)
        went_in = False
        pygame.event.get()
        key_input = pygame.key.get_pressed()
        arrw_dst = get_arrow(key_input)
        cont = do_action(key_input)
        if cont is None or cont is False:
            stop()
            frame_no-=1
        else:
            print(frame_no)
            # cv2.imwrite(directory + str(time.time()-strt) + '_' + str(frame_no)+'_'+str(cont) + '.png', gray_frame)
        cv2.arrowedLine(pygame_frame, arrw_src, arrw_dst, color=(255,0,0), thickness=2)#, line_type, shift, tipLength)
        display.blit(pygame.surfarray.make_surface(pygame_frame), (0,0))
        pygame.display.flip()
        cv2.imshow("grey", gray_frame[100:200,])
        cv2.waitKey(1)
        # print(frame_no)
        if cont is False:
            break
    # keyboard.add_hotkey('s', bk) # unmute on keydown
    # keyboard.add_hotkey('s', stop, trigger_on_release=True) # mute on keyup

    # keyboard.add_hotkey('w', fd) # unmute on keydown
    # keyboard.add_hotkey('w', stop, trigger_on_release=True) # mute on keyup

    # keyboard.add_hotkey('w+a', lt) # unmute on keydown
    # keyboard.add_hotkey('w+a', stop, trigger_on_release=True) # mute on keyup


    # keyboard.add_hotkey('w+d', rt) # unmute on keydown
    # keyboard.add_hotkey('w+d', stop, trigger_on_release=True) # mute on keyup


    # keyboard.add_hotkey('a', lt_sharp) # unmute on keydown
    # keyboard.add_hotkey('a', stop, trigger_on_release=True) # mute on keyup


    # keyboard.add_hotkey('d', rt_sharp) # unmute on keydown
    # keyboard.add_hotkey('d', stop, trigger_on_release=True) # mute on keyup

    # keyboard.wait(' ') # wait forever
    # stop()


    # fd(2)
    # bk(2)


    GPIO.cleanup()
