import torch
from einops import rearrange
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

class AttnState:
    STORE = 0
    LOAD = 1

    def __init__(self):
        self.reset()

    @property
    def state(self):
        return self.__state

    @property
    def timestep(self):
        return self.__timestep

    def set_timestep(self, t):
        self.__timestep = t

    def reset(self):
        self.__state = AttnState.STORE
        self.__timestep = 0

    def reset_wo_timestep(self):
        self.__state = AttnState.STORE

    def to_load(self):
        self.__state = AttnState.LOAD

class CrossFrameAttnProcessor(AttnProcessor2_0):
    """
    Cross frame attention processor. Each frame attends the first frame and previous frame.

    Args:
        attn_state: Whether the model is processing the first frame or an intermediate frame
    """

    def __init__(self, attn_state: AttnState):
        super().__init__()
        self.attn_state = attn_state
        self.cur_timestep = 0
        self.first_maps = {}
        self.prev_maps = {}

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, **kwargs):

        # print("map length ", len(self.first_maps))
        # if encoder_hidden_states is None:
        #     # Is self attention

        #     tot_timestep = self.attn_state.timestep
        #     # print("tot time step is ", tot_timestep)
        #     if self.attn_state.state == AttnState.STORE:
        #         # print("come to store ", self.cur_timestep, tot_timestep)
        #         self.first_maps = hidden_states.detach()
        #         self.prev_maps = self.first_maps
        #         res = super().__call__(attn, hidden_states, encoder_hidden_states, **kwargs)
        #     else:
        #         # print("come to load ", self.cur_timestep, tot_timestep)
        #         tmp = hidden_states.detach()
        #         # cross_map = torch.cat([self.first_maps, self.prev_maps], dim=1)
        #         cross_map = self.first_maps
        #         res = super().__call__(attn, hidden_states, cross_map, **kwargs)
                

        #     self.cur_timestep += 1
        #     if (self.cur_timestep + 1) % 10 == 0:
        #         self.prev_maps = hidden_states.detach()
        # else:
        #     # Is cross attention
        #     res = super().__call__(attn, hidden_states, encoder_hidden_states, **kwargs)

        # return res

        #input one frame at a time, above is input all frames at once
        if encoder_hidden_states is None:
            # Is self attention

            tot_timestep = self.attn_state.timestep #total denoising timestep, in DDIM we use 20 or 50

            if self.attn_state.state == AttnState.STORE:
                self.first_maps[self.cur_timestep] = hidden_states.detach()
                self.prev_maps[self.cur_timestep] = hidden_states.detach()
                res = super().__call__(attn, hidden_states, encoder_hidden_states, **kwargs)
            else:
                tmp = hidden_states.detach()
                cross_map = self.first_maps
                res = super().__call__(attn, hidden_states, cross_map, **kwargs)
                self.prev_maps[self.cur_timestep] = tmp

            self.cur_timestep += 1
            if self.cur_timestep == tot_timestep:
                self.cur_timestep = 0
        else:
            # Is cross attention
            res = super().__call__(attn, hidden_states, encoder_hidden_states, **kwargs)

        return res
    

# cross_map = torch.cat([self.first_maps, self.prev_maps], dim=1)