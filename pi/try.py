import RPi.GPIO as GPIO
import time

channels = [3, 5, 7, 8]
L_F, L_B, R_F, R_B = channels

def startup():
	GPIO.setmode(GPIO.BOARD)
	GPIO.setup(channels, GPIO.OUT)



def stop():
    GPIO.output(channels,0)

def fd(secs=None):
	stop()
	GPIO.output([L_F, R_F], 1)
	if secs:
		time.sleep(secs)
		stop()

def bk(secs=None):
	stop()
	GPIO.output([L_B, R_B], 1)
	if secs:
		time.sleep(secs)
		stop()

def rt(secs=None):
	stop()
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


if __name__ == '__main__':
	startup()

	stop()

	bk(0.1)

	# fd(2)
	# bk(2)


	GPIO.cleanup()
