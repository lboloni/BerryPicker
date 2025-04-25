import sys

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

sys.path.append(Config().values["conv_vae"]["code_dir"])

from sensorprocessing import sp_conv_vae

def main():
    print("Preprocess representation")
    print("Not implemented yet: specify the dataset (as a set of directories)") 
    print("Not implemented yet: run the representation and save the values") 

    #conf = read_json("/home/lboloni/Documents/Hackingwork/_Checkouts/VisionBasedRobotManipulator/src/settings-tredy2.json")
    print(Config())
    print(Config().values["robot"])
    print(Config().values["robot"]["usb_port"])
    print(Config().values["conv_vae"]["model_dir"])

    sp = sp_conv_vae.ConvVaeSensorProcessing()

if __name__ == "__main__":
    main()

