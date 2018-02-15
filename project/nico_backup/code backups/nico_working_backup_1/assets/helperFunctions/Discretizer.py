class Discretizer():
    def __init__(self,min_val,max_val):
        self.min_val = min_val
        self.max_val = max_val
        self._check_inputs()
        #self._discretize()

    def _check_inputs(self):
        self.step_length = input("Enter desired step length of the action-space that shall be discretized [0..1]: ")
        self.step_length = float(self.step_length)
        assert self.step_length<=env.max_torque, "step length must be between 0 and {}".format(env.max_torque)
        assert self.min_val<self.max_val, "min value has to be greater or equal to the max value"
        assert self.step_length>0, "step length hast to be greater than zero!"
