import torch
import torch.nn as nn
from utils import get_mean_std

# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.model = 

#     def forward(self, X):
#         return self.model(X)
#class VGG(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.model = 

#     def forward(self, X):
#         return self.model(X)
decoder = nn.Sequential(
            nn.ReflectionPad2d(1), #reflection pad all boundaries 
            nn.Conv2d(512, 256, 3), #in, out, kernel
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), #use upsampling instead of 
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
        )

vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
# 

class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])
        self.decoder = decoder
        #freeze encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{}'.format(i+1))(input)
        return input

    def encode_stages(self, input):
        out_list = []
        for i in range(4):
            input = getattr(self, 'enc_{}'.format(i+1))(input)
            out_list.append(input)
        return out_list

    def get_content_loss(self, pred, target):
        assert(pred.size() == target.size())
        assert(target.requires_grad == False)
        return self.mse_loss(pred, target)

    def get_style_loss(self, preds, targets):
        '''takes a list of pred and target feat maps
        and calculate the mse loss of the mean and std
        '''
        assert(len(preds) == len(targets))
        style_loss = 0.
        for p_feat, t_feat in zip(preds, targets):
            assert(p_feat.size() == t_feat.size())
            assert(t_feat.requires_grad == False)
            p_mean, p_std = get_mean_std(p_feat)
            t_mean, t_std = get_mean_std(t_feat)
            style_loss += self.mse_loss(p_mean, t_mean) + self.mse_loss(p_std, t_std)

        return style_loss
    def AdaIn(self, content_feat, style_feat):
        assert(content_feat.size() == style_feat.size())
        c_mean, c_std = get_mean_std(content_feat)
        s_mean, s_std = get_mean_std(style_feat)
        norm_content_feat = (content_feat - c_mean)/c_std * s_std + s_mean
        return norm_content_feat


    def forward(self, content, style, alpha=1.0):
        content_feat = self.encode(content)
        style_feats = self.encode_stages(style)
        norm_content_feat = self.AdaIn(content_feat, style_feats[-1])
        norm_content_feat = alpha*norm_content_feat + (1-alpha)*content_feat
        decoded_img = self.decoder(norm_content_feat)
        decoded_feats = self.encode_stages(decoded_img)
        content_loss = self.get_content_loss(decoded_feats[-1], norm_content_feat)
        style_loss = self.get_style_loss(decoded_feats, style_feats)
        return content_loss, style_loss

