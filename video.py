import numpy as np
import cv2
from skimage.measure import compare_ssim
import logging
import imutils

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        logging.debug()
        return result
    return timed


class MapleBotAlgo:
    MIN_SSIN = 0.92

    def get_image_difference(self, imageA, imageB):
            
        # compute the Structural Similarity Index (SSIM) between the two
        # images, ensuring that the difference image is returned
        (score, diff) = compare_ssim(imageA, imageB,  full=True)

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

        return cnts

        # m =  map(cv2.boundingRect, cnts)
        # f = filter(self.is_monster, map(cv2.boundingRect, cnts))

class VideoMapleBot:

    WINDOW_NAME = "image"
    
    def __init__(self, video):
        self._setup_window()
        self.cap = cv2.VideoCapture(video)
        self.maple_bot_algo = MapleBotAlgo()

    def _setup_window(self):
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1920,1080)
    
    def show_frame(self, frame):
        cv2.imshow(self.WINDOW_NAME, frame)
        

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.maple_bot_algo
        return gray
    
    def process_video(self):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            frame = self.process_frame(frame)
            self.show_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()



def main():
    bot = VideoMapleBot("144.mp4")
    bot.process_video()

if __name__ == '__main__':
    main()  
    


    

# while(cap.isOpened()):
#     ret, frame = cap.read()

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     cv2.imshow('image',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# def process_frame() 