import pyautogui
import imutils
from skimage.measure import compare_ssim
import imutils
import win32gui
# import PIL.ImageGrab
from PIL import ImageGrab
# import PIL
import numpy as np
import cv2
import time
import logging


TIME_BETWEEN_SAMPLES = 0.3

class MapleBot:
    NOT_RELEVEMT_SCREEN_Y = 70
    WINDOW_NAME = 'MapleRoyals'
    MIN_SSIN = 0.92
    
    MIN_MONSTER_SIZE = (20, 20)

    def __init__(self):
        self.x_up = None
        self.y_up = None
        self.x_length = None
        self.y_length = None
        self.last_run = 0
        self.last_image = None

    def get_grab_sizes(self):
        hwnd = win32gui.FindWindow(None, self.WINDOW_NAME)
        win32gui.SetForegroundWindow(hwnd)
        hwnd = win32gui.FindWindow(None, self.WINDOW_NAME)
        if hwnd:
            win32gui.SetForegroundWindow(hwnd)
            x, y, x1, y1 = win32gui.GetClientRect(hwnd)
            x, y = win32gui.ClientToScreen(hwnd, (x, y))
            x1, y1 = win32gui.ClientToScreen(hwnd, (x1 - x, y1 - y))

        self.x_up = x
        self.y_up = y
        self.x_length = x1
        self.y_length = y1 - self.NOT_RELEVEMT_SCREEN_Y

        logging.info("x_up = {}".format(self.x_up))
        logging.info("y_up = {}".format(self.y_up))
        logging.info("x_length = {}".format(self.x_length))
        logging.info("y_length = {}".format(self.y_length))
    
    
    def screenshot(self):
        image = np.array((ImageGrab.grab(bbox=(self.x_up, self.y_up, self.x_length, self.y_length))),dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        # image = cv2.Canny(image, threshold1=200, threshold2=300)
        # image = np.array((PIL.ImageGrab.grab(bbox=(0, 40, 800, 640))))
        # printscreen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        # print(image)
        return image

    def get_image_difference(self, delay=None):

        previousGray = self.last_image

        diff_time = time.time() - self.last_run
        
        if diff_time < TIME_BETWEEN_SAMPLES:
            time.sleep(TIME_BETWEEN_SAMPLES - diff_time)

        self.last_run = time.time()

        currentGray = self.screenshot()

        self.last_image = currentGray

        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        (score, diff) = compare_ssim(previousGray, currentGray, full=True)

        if score < self.MIN_SSIN:
            logging.warning("ssin to low: {} exit".format(score))
            return

        diff = (diff * 255).astype("uint8")

        # threshold the difference image, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)


        # print(list(map(cv2.boundingRect, cnts)))
        m =  map(cv2.boundingRect, cnts)
        f = filter(self.is_monster, map(cv2.boundingRect, cnts))

        for i in f:
            x, y, w, h = i
            cv2.rectangle(previousGray, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow("Original", previousGray)



            
        # filter(self.check_if_monstermap(cnts, cv2.boundingRect))
        # # loop over the contours
        # for c in cnts:            # compute the bounding box of the contour and then draw the
        #     # bounding box on both input images to represent where the two
        #     # images differ
        #     (x, y, w, h) = cv2.boundingRect(c)
        #     print(x, y, w, h)

        
        #         continue

        #     cv2.rectangle(grayB, (x, y), (x + w, y + h), (0, 0, 255), 2)


        # # show the output images
        # # cv2.imshow("Modified", grayB)
        # # cv2.imshow("Diff", diff)
        # # cv2.imshow("Thresh", thresh)
    
    def is_monster(self, coordinates):
        x, y, w, h = coordinates
        print("w:", w)
        print("h:", h)
        is_mon = not (w < self.MIN_MONSTER_SIZE[0] or h < self.MIN_MONSTER_SIZE[1])
        print(is_mon)
        return is_mon
    
    def process(self):
        start_loop_time = time.time()
        # screen = self.screenshot()
        # time.sleep(1)
        monsters = self.get_image_difference()
        # print(list(monsters))

        logging.debug("loop took: {}".format(time.time() - start_loop_time))

    
    # def move()


class MonsterFilter:
    pass


def main():
    logging.basicConfig(level=logging.DEBUG)
    maplebot = MapleBot()
    maplebot.get_grab_sizes()
    maplebot.last_image = maplebot.screenshot()
    maplebot.last_run = time.time()
    while True:
        maplebot.process()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()