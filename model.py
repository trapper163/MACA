import math
import torch.nn as nn
import torch
from nets.model_utils import resnet50, FPN_Decoder, Head

class GatedConv2dWithActivation(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,batch_norm=True, activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv2dWithActivation, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.mask_conv2d = torch.nn.Conv2d(in_channels, out_channels, (1, None), stride, padding, dilation, groups, bias)
        self.batch_norm2d = torch.nn.BatchNorm2d(out_channels)
        self.sigmoid = torch.nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
    def gated(self, mask):
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.batch_norm:
            return self.batch_norm2d(x)
        else:
            return x

class dual_view_detector(nn.Module):
    def __init__(self, num_classes=15, pretrained=True, input_channel=[2048, 1024, 512, 256], output_channel):
        super(dual_view_detector, self).__init__()
        self.pretrained = pretrained
        self.input_channel = input_channel
        self.output_channel = output_channel

        self.backbone = resnet50(pretrained=pretrained)

        self.ol_decoder = FPN_Decoder()
        self.sd_decoder = FPN_Decoder()

        self.cov_ol = nn.Conv2d(self.input_channel, self.output_channel, kernel_size=1, stride=1)
        self.cov_sd = nn.Conv2d(self.input_channel, self.output_channel, kernel_size=1, stride=1)

        self.ol_head = Head(num_classes=num_classes)
        self.sd_head = Head(num_classes=num_classes)

        self._init_weights()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False


    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True


    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


    def MA(self, x):

        x_pooled_upsample_5 = torch.zeros((x.shape[0], 8, 300, 300))
        x_pooled_upsample_10 = torch.zeros((x.shape[0], 8, 300, 300))
        x_pooled_upsample_15 = torch.zeros((x.shape[0], 8, 300, 300))

        x_pooled_5 = self.AdaptiveAverPool_5(x)
        x_pooled_10 = self.AdaptiveAverPool_10(x)
        x_pooled_15 = self.AdaptiveAverPool_15(x)
        for i in range(60):
            for j in range(60):
                x_pooled_upsample_5[:, :, i * 5:(i + 1) * 5, j * 5:(j + 1) * 5] = x_pooled_5[:, :, i, j].unsqueeze(
                    -1).unsqueeze(-1)

        for i in range(30):
            for j in range(30):
                x_pooled_upsample_10[:, :, i * 10:(i + 1) * 10, j * 10:(j + 1) * 10] = x_pooled_10[:, :, i,
                                                                                       j].unsqueeze(-1).unsqueeze(-1)

        for i in range(20):
            for j in range(20):
                x_pooled_upsample_15[:, :, i * 15:(i + 1) * 15, j * 15:(j + 1) * 15] = x_pooled_15[:, :, i,
                                                                                       j].unsqueeze(-1).unsqueeze(-1)

        x_concat_5 = torch.cat((x, x_pooled_upsample_5), 1)
        x_concat_10 = torch.cat((x, x_pooled_upsample_10), 1)
        x_concat_15 = torch.cat((x, x_pooled_upsample_15), 1)

        x_concat_5_out = self.conv2d_1_rgb_red_concat_5(x_concat_5)
        x_concat_10_out = self.conv2d_1_rgb_red_concat_10(x_concat_10)
        x_concat_15_out = self.conv2d_1_rgb_red_concat_15(x_concat_15)

        x_gated_conv_input = torch.cat((x_concat_5_out, x_concat_10_out), 1)
        x_gated_conv_input = torch.cat((x_gated_conv_input, x_concat_15_out), 1)

        x_gated_conv_output = self.GatedConv2dWithActivation(x_gated_conv_input)
        return x_gated_conv_output

    def forward_features_Fusion(self, x, y):

        x = self.cov_ol(x + y)
        y = self.cov_sd(x + y)

        return x, y


    def forward(self, ol_x, sd_x):
        ol_feat = self.backbone(ol_x)
        sd_feat = self.backbone(sd_x)

        ol_feat = self.ol_decoder(ol_feat)
        sd_feat = self.sd_decoder(sd_feat)

        ol_feat, sd_feat = self.forward_features_Fusion(ol_feat, sd_feat)

        ol_hm, ol_wh, ol_offset = self.ol_head(ol_feat)
        sd_hm, sd_wh, sd_offset = self.sd_head(sd_feat)
        return ol_hm, ol_wh, ol_offset, sd_hm, sd_wh, sd_offset



if __name__ == "__main__":
    from thop import profile
    from thop import clever_format

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dual_view_detector(15, pretrained=False)
    model = model.to(device)

    flops, params = profile(model, (torch.randn(1, 3, 300, 300).to(device), torch.randn(1, 3, 300, 300).to(device)))
    flops, params = clever_format([flops, params], '%.3f')
    print(flops)
    print(params)