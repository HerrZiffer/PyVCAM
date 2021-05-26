import numpy as np
import threading
import time

from pyvcam import pvc
from pyvcam.camera import Camera

NUM_FRAMES = 5
EXPOSE_TIME_ms = 200

def printFrameDetails(frame, fps, frame_count, desc):
    low = np.amin(frame['pixel_data'])
    high = np.amax(frame['pixel_data'])
    average = np.average(frame['pixel_data'])
    print(desc + ': Min:{}\tMax:{}\tAverage:{:.0f}\tFrame Rate: {:.1f}\tFrame Count: {:.0f}'.format(low, high, average, fps, frame_count))

class LatestThreadRun (threading.Thread):
    def __init__(self, cam):
        threading.Thread.__init__(self)
        self.cam = cam
        self.end = False

    def run(self):
        while not self.end:
            try:
                # Wait 2 frames
                time.sleep(2 * EXPOSE_TIME_ms * 1e-3)

                frame, fps, frame_count = self.cam.poll_latest_frame()
                printFrameDetails(frame, fps, frame_count, 'latest')

            except Exception as e:
                pass

    def stop(self):
        self.end = True

def main():
    # Initialize PVCAM and find the first available camera.
    pvc.init_pvcam()

    cam = [cam for cam in Camera.detect_camera()][0]
    cam.open()

    # Start a thread for executing the trigger
    t1 = LatestThreadRun(cam)
    t1.start()

    cam.start_seq(exp_time=EXPOSE_TIME_ms, num_frames=NUM_FRAMES)

    # Wait for all frames to be captured. Frames take a slightly longer than expose time, so wait a little extra time
    time.sleep(NUM_FRAMES * EXPOSE_TIME_ms * 1e-3 + 0.5)
    print('')

    cnt = 0
    while cnt < NUM_FRAMES:
        frame, fps, frame_count = cam.poll_frame()
        printFrameDetails(frame, fps, frame_count, 'oldest')

        # Save to disk here


        cnt += 1

    t1.stop()
    cam.finish()
    cam.close()
    pvc.uninit_pvcam()

    print('\nTotal frames: {}'.format(cnt))

if __name__ == "__main__":
    main()
