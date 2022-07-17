import torch
from torch import nn
from model.edsr_fortest import make_model as ours_make_model
from model.edsr_bireal_fortest import make_model as bireal_make_model
from model.edsr_residual_fortest import make_model as residual_make_model

def to_real_binary(model):
    for module in model.body.modules():
        if isinstance(module, nn.Conv2d):
            print('one')
            module.weight.data = module.weight.data.sign()/2 + 0.5
            # module.weight.data.astype(torch.float32)
    if hasattr(model, 'tail1'):
        for module in model.tail1.modules():
            if isinstance(module, nn.Conv2d):
                print('one in tail1')
                module.weight.data = module.weight.data.sign()/2 + 0.5
    if hasattr(model, 'tail11'):
        for module in model.tail11.modules():
            if isinstance(module, nn.Conv2d):
                print('one in tail11')
                module.weight.data = module.weight.data.sign()/2 + 0.5
        

#Function to Convert to ONNX 
def Convert_ONNX(model, input_size, onnx_name): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    # dummy_input = torch.randn(1, input_size, requires_grad=True)  
    dummy_input = torch.randn(input_size, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "{}.onnx".format(onnx_name),       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}})
    print(" ") 
    print('Model has been converted to ONNX') 
    print('save name : {}'.format(onnx_name))

if __name__ == '__main__':
    from option import args
    args.binary_mode = 'binary'
    args.n_resblocks = 16
    args.n_feats = 64
    args.scale = [4]
    args.n_colors = 1
    args.rgb_range = 256
    args.res_scale = 1
    
    # model = make_model(args)
    # to_real_binary(model)
    # # for name,p in model.named_parameters():
    # #     print(name, p.dtype)
    # # Convert_ONNX(model, (1,1,256,256), 'edsr_binary')
    # Convert_ONNX(model, (1,1,256,256), 'edsr_bireal_binary')

    name2model = {
        'ours' : ours_make_model,
        'bireal' : bireal_make_model,
        'residual' : residual_make_model
    }
    blocks_feats = [[16,64], [32,256]]
    scales = [2,4]

    for name in name2model:
        for block_feat in blocks_feats:
            for scale in scales:
                args.n_resblocks, args.n_feats = block_feat[0], block_feat[1]
                args.scale = [scale]
                model = name2model[name](args)

                to_real_binary(model)
                Convert_ONNX(model, (1,1,256,256), 'edsr_{}_x{}_n{}_c{}'.format(name, scale, block_feat[0], block_feat[1]))