#import numpy as np
#from pathlib import Path
import cv2
#import time

class CameraController:
    """This class is used to connect to a specified set of cameras. Images are captured from the cameras whenever the update is called. At the same time, it can also save the images to a specific directory and a specific name.
    
    # camera0 - webcam on the computer
    # camera2 - right mounted
    # camera3 - the free floating one
    # camera4 - the center mounted one 
    # cameras = [0, 2, 3, 4]
    # cameras = [0, 2]
    cameras = [4]    
    """
    def __init__(self, exp):
        """
        cameras: a list of numbers which correspond to the capture devices that will be captured
        dimension: the dimension to which the images are scaled down
        """
        self.exp = exp
        self.img_size = exp["saved_image_size"]
        # create the capture devices
        self.caption = "Cameras: "
        self.capture_devs = {}
        if "views" in exp:
            cameras = []
            for view_name, view_config in exp["views"].items():
                if "device" not in view_config:
                    raise ValueError(f"Camera view {view_name} is missing its device")
                cameras.append((view_name, view_config["device"]))
        else:
            cameras = [(f"dev{i}", i) for i in exp["active_camera_list"]]
        for view_name, device in cameras:
            cap = cv2.VideoCapture(device)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size[1])
            cap.set(cv2.CAP_PROP_FPS, self.exp["fps"])

            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                self.stop()
                raise RuntimeError(f"Unable to open configured camera {view_name} on device {device}")
            else:
                self.capture_devs[view_name] = cap
                self.caption += f"{view_name} "
                print(f"camera {view_name} on device {device} works")
        self.caption += "Press q to quit"
        self.images = {}
        self.visualize = True # if true, visualizes the captured images

    def stop(self):
        """When everything done, release the capture devices and close the windows"""
        for cap in self.capture_devs:
            self.capture_devs[cap].release()
        cv2.destroyAllWindows()

    def update(self):
        """
        Takes captures from all the active cameras, processes them, updates the window and optionally saves the images. Returns the key returned by waitKey()

        This one works, but it breaks down as soon as we have too many cameras
        
        If it returns True, a key to exit was pressed
        """
        for index in self.capture_devs:
            cap = self.capture_devs[index]
            success, image = cap.read()
            if not success:
                continue
            if self.img_size != None:
                image = cv2.resize(image, self.img_size)
            self.images[index] = image
        # create a list of concatenated images
        imglist = list(self.images.values())
        concatenated_image = cv2.hconcat(imglist)
        try:
            if self.visualize:
                cv2.imshow(self.caption, concatenated_image)
                key = cv2.waitKey(1)
                return (key & 0xFF) == ord('q')
        except:
            print("Error at visualization? ")
