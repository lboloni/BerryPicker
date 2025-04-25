import sys
import pathlib

current_file = pathlib.Path(__file__)
sys.path.append(str(current_file.parent.parent.parent))

import unittest
import json

from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"

from robot.al5d_position_controller import RobotPosition
from behavior_cloning.demo_to_trainingdata import BCDemonstration

class TestPosition(unittest.TestCase):
    """Unit tests for the position object"""
    
    def test_to_normalized_vector(self):
        demos_top = pathlib.Path(Config()["demos"]["directory"])
        demos_dir = pathlib.Path(demos_top, "demos")
        task_dir = pathlib.Path(demos_dir, "proprioception-uncluttered")
        positions = []
        for demo in task_dir.iterdir():
            if not demo.is_dir(): continue
            for item in demo.iterdir():
                if item.suffix != ".json": continue
                if item.name == "_demonstration.json": continue                
                current = self.check_normalized_json(item)
                positions.append(current)
            
        # check the distances
        for i in range(len(positions)-1):
            rp1 = positions[i]
            rp2 = positions[i+1]
            dist = rp1.empirical_distance(rp2)
            print(f"Distance {dist}")
        print("Test passed")
        return 

    def check_normalized_json(self, jsonfile):
        with open(jsonfile) as file:
            data = json.load(file)
        datadict = data["rc-position-target"]
        rp = RobotPosition(datadict)
        normvector = rp.to_normalized_vector()
        print(f"{rp} norm {normvector}")
        rp2 = RobotPosition.from_normalized_vector(normvector)
        print(f"Reconstructed {rp2}")
        return rp        

if __name__ == '__main__':
    unittest.main()