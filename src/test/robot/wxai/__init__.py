import unittest

from exp_run_config import Config

if "wxai" not in Config().get_experiment("machine", "current")["robots"]:
    raise unittest.SkipTest("wxai is not in the robots of machine/current")
