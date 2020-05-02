import numpy as np
import matplotlib.pyplot as plt

def get_class_index(c_name, classes):
  for i in range(len(classes)):
    name = classes[i].split(' ')[-1].strip()
    if name == c_name:
      return i

def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

with open('hmdb_labels.txt', 'r') as f:
    classes = f.readlines()
    f.close()


import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]

'''
3D conv layers
'''
def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


'''
Resnet Block
'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

'''
Full 3D Resnet
'''

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=51):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)

    return model



action_name = 'climb_stairs'
model_path_prev = 'prev_model.pt'
model_path_curr = 'curr_model.pt'



import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
from google.colab.patches import cv2_imshow

def get_vid_probs_and_json(action_name,model_path,video_path,save_fig_json=False,save_name_fig=None,save_name_json=None,prev=False):
    '''
    Load model
    '''
    if prev:
      model = generate(18)
    else:
      model = generate(34)

    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    '''
    Get video
    '''
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    '''
    List for the timestamp/prob and list for the clips
    '''
    full_list = []
    clip = []

    '''
    Iterate through each frame in the video
    '''
    i = 0
    still_frames = True
    while still_frames:
        i+=1
        still_frames, frame = cap.read()
        if still_frames:
            if i > 15:
              # Only append if we are past the 16th frame as the previous are not used
              full_list.append([cap.get(cv2.CAP_PROP_POS_MSEC)/1000])
        else:
            break

        '''
        Adapted Preprocessing code
        '''
        resized = center_crop(cv2.resize(frame, (171, 128)))
        tmp = resized - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)

        
        # Has to fit model shape so wait until we get 16 frames
        if len(clip) == 16:
          with torch.no_grad():
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).cuda()
            outputs = model.forward(inputs)

          probs = torch.nn.Softmax(dim=1)(outputs)

          class_index = get_class_index(action_name, classes)
          full_list[-1].append(probs[0][class_index].item())

          clip.pop(0)

    del full_list[-1]

    plt.xlabel('Time (seconds)')
    plt.ylabel('Prob of Given Action')
    plt.plot(np.array(full_list)[:,0],np.array(full_list)[:,1])
    if save_fig_json:
      plt.savefig(save_name_fig)

    full_json = {action_name: full_list}
    if save_fig_json:
        with open(save_name_json, 'w') as fp:
          json.dump(full_json, fp)

    return full_json



video_path = 'Video 1.avi'
dl = get_vid_probs_and_json(action_name, model_path_curr, video_path,True,'fig1one','json1one')



video_path = 'Video 2.avi'
dl = get_vid_probs_and_json(action_name, model_path_curr, video_path,True,'fig1two','json1two')


video_path = 'Video 3.avi'
dl = get_vid_probs_and_json(action_name, model_path_curr, video_path,True,'fig1three','json1three')


video_path = 'Video 4.avi'
dl = get_vid_probs_and_json(action_name, model_path_prev, video_path,True,'fig2prev_one','json2prev_one',prev=True)


video_path = 'Video 4.avi'
dl = get_vid_probs_and_json(action_name, model_path_curr, video_path,True,'fig2curr_one','json2curr_one')


video_path = 'Video 5.avi'
dl = get_vid_probs_and_json(action_name, model_path_prev, video_path,True,'fig2prev_two','json2prev_two',prev=True)

video_path = 'Video 5.avi'
dl = get_vid_probs_and_json(action_name, model_path_curr, video_path,True,'fig2curr_two','json2curr_two')






