from pyvcam import pvc
from pyvcam.camera import Camera
import time
import npy2bdv

def main():
    pvc.init_pvcam()
    cam = next(Camera.detect_camera())
    cam.open()
    cam.start_live(exp_time=1)

    frames_streamed_to_disk = 0
    start = time.time()

    fname = "frames.h5"
    bdv_writer = npy2bdv.BdvWriter(fname, blockdim=((1, 2200, 3200),), overwrite=True)

    try:
        while True:
            frame, fps, frame_count = cam.poll_frame()

            # Stream to disk
            stack = frame['pixel_data'].reshape((1, frame['pixel_data'].shape[0], frame['pixel_data'].shape[1]))
            bdv_writer.append_view(stack, time=frames_streamed_to_disk)

            frames_streamed_to_disk += 1
            if frame_count % 100 == 0:
                print('Acquired frame rate: {:.1f}. Frames streamed to disk: {:d}.'.format(fps, frames_streamed_to_disk))

            if frames_streamed_to_disk == 100:
                break

    except KeyboardInterrupt:
        pass

    bdv_writer.write_xml_file(ntimes=frames_streamed_to_disk)
    bdv_writer.close()

    cam.finish()
    cam.close()
    pvc.uninit_pvcam()

    print('Total frames streamed to disk: {}\nFrames streamed to disk per second: {}\n'.format(frames_streamed_to_disk, (frames_streamed_to_disk/(time.time()-start))))


if __name__ == "__main__":
    main()
