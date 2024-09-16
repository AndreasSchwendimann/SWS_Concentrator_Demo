#!/usr/bin/env python
import RPi.GPIO as GPIO
import pandas as pd
 
LightInPin = 11 # photointerrupter
OutLed = 10 # LED

def setup():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(OutLed, GPIO.OUT) # Set Led Pin mode to output
    GPIO.setup(LightInPin, GPIO.IN, pull_up_down=GPIO.PUD_UP) # Set pull up to high level(3.3V)
    GPIO.add_event_detect(LightInPin, GPIO.BOTH, callback=detect, bouncetime=8)
 
def Led(x):
    if x == 0:
        GPIO.output(OutLed, 0)
    if x == 1:
        GPIO.output(OutLed, 1)

def Print(x):
    if x == 1:
        print(' Light has been interrupted')
        print(' --------------------------')
        df = pd.DataFrame({'timestamp': [pd.Timestamp.now()]})
        df.to_csv('measured_particles.csv', mode='a', header=False, index=False)
 
def detect(chn):
    Led(GPIO.input(LightInPin))
    Print(GPIO.input(LightInPin))
 
def loop():
    while True:
        pass
 
def destroy():
    GPIO.output(OutLed, GPIO.HIGH) # Green led off
    GPIO.cleanup() # Release resource
 
if __name__ == '__main__': # Set the Program start from here
    setup()
    print("this is a test")
    try:
        loop()
    except KeyboardInterrupt: # When pressed 'Ctrl+C' child program destroy() will be executed.
        destroy()
