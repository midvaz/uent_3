from script import config
# from network_unet import train
from script import interfase

# 1034

data = config.get_configuration()
cnn_data = data['default']
print(cnn_data)
# cnn = train.Unet(
#     classes=cnn_data['classes'],
#     output_size=cnn_data['output_size'],
#     sampel_size=cnn_data['sample_size'],
#     epoch= cnn_data['epoch'],
#     count_aug = cnn_data['count_aug']
# )
if __name__ == '__main__':
    window = interfase.Interfase()
    window.rendering()

