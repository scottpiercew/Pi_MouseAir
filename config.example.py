# Copy to config.py for local use. Never commit real credentials.
#
#
# configuration file - contains customization for exact system
#
#

SWVersion = ""    # Set in MouseAir.py
DEBUGMouseAir = True

mailUser = ''
mailPassword = ''

databasePassword = ''

notifyAddress = ''
secondaryNotifyAddress = ''
fromAddress = ''

############
# Blynk configuration
############

USEBLYNK = False
BLYNK_AUTH = ''
BLYNK_URL = 'https://example.invalid/'

############
# RGBW LEDs configuration (on use on a Raspberry Pi 3B+)
############

RGBWLEDS = True

############
# CatNotCat Parameters
############

#CatNotCatThreshold in state.py because of Blynk control
DetectCatNumberOfFrames = 10 # number of continous frames before we detect Cat


# pin configuration

MouseDetectionPin = 4
UltrasonicLevelPin = 26
PixelPin = 21
