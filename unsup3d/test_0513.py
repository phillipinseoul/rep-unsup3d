import torch
from modules import Encoder
from utils import ImageFormation

light_net = Encoder(4)
ex = torch.randn(1, 3, 64, 64)

normal_map = torch.randn(1, 3, 64, 64)

res = light_net(ex)
print(res.shape)

imageForm = ImageFormation()

sh = imageForm.normal_to_shading(normal_map, res)
print(sh.shape)