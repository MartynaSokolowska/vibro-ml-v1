import torch
import tempfile
import os
from src.models.audio_backbones import build_audio_model

# ResNet test
model_res = build_audio_model(backbone='resnet18', mode='classification', num_classes=6, pretrained=False, input_channels=3, config={'model':{'backbone':'resnet18'}})
fd, path_res = tempfile.mkstemp(suffix='.pth')
os.close(fd)
torch.save(model_res.state_dict(), path_res)
model_res2 = build_audio_model(backbone='resnet18', mode='classification', num_classes=6, pretrained=False, input_channels=3, pretrained_weights=path_res, config={'model':{'backbone':'resnet18'}})
print('resnet loaded ok', tuple(model_res2(torch.randn(2,3,128,128)).shape))
os.remove(path_res)

# CNN14 test
model_cnn = build_audio_model(backbone='cnn14', mode='classification', num_classes=6, pretrained=False, input_channels=1, config={'model':{'backbone':'cnn14'}})
fd, path_cnn = tempfile.mkstemp(suffix='.pth')
os.close(fd)
torch.save(model_cnn.state_dict(), path_cnn)
model_cnn2 = build_audio_model(backbone='cnn14', mode='classification', num_classes=6, pretrained=False, input_channels=1, pretrained_weights=path_cnn, config={'model':{'backbone':'cnn14'}})
print('cnn14 loaded ok', tuple(model_cnn2(torch.randn(2,1,128,128)).shape))
os.remove(path_cnn)

# AST transformer test
model_ast = build_audio_model(backbone='ast', mode='classification', num_classes=6, pretrained=False, input_channels=1, config={'model':{'backbone':'ast','spectrogram_size':[128,224]}})
fd, path_ast = tempfile.mkstemp(suffix='.pth')
os.close(fd)
torch.save(model_ast.state_dict(), path_ast)
model_ast2 = build_audio_model(backbone='ast', mode='classification', num_classes=6, pretrained=False, input_channels=1, pretrained_weights=path_ast, config={'model':{'backbone':'ast','spectrogram_size':[128,224]}})
print('ast loaded ok', tuple(model_ast2(torch.randn(2,1,128,224)).shape))
os.remove(path_ast)

print('All pretrained loader tests passed')
