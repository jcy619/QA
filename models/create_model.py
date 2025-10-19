
from .vit_qa import ViT
from .swin import SwinTransformer
from .t2t import T2T_ViT
from .resnet import resnet50
from .vgg import vgg19
from .mobilenet import mobilenet_v2


def create_model(img_size, n_classes, args):
    if args.model == 'vit':
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=args.sd, is_SPT=args.is_SPT, is_LSA=args.is_LSA)
    elif args.model =='swin':
        #depths = [5,4]
        #num_heads = [6,12]
        depths=[2,6,4]
        num_heads=[3,6,12]
        
        mlp_ratio = 2
        window_size = 4
        patch_size = 2 if img_size == 32 else 4
        model=SwinTransformer(img_size=img_size,embed_dim=96,window_size=window_size,drop_path_rate=args.sd,                     patch_size=patch_size,mlp_ratio=mlp_ratio,depths=depths,num_heads=num_heads,num_classes=n_classes,
                                is_SPT=args.is_SPT, is_LSA=args.is_LSA)

    elif args.model == 'resnet':
        model = resnet50(num_classes=n_classes)
    
    elif args.model == 'vgg':
        model = vgg19(num_classes = n_classes)
    elif args.model == 'mobilenet':
        model = mobilenet_v2(num_classes=n_classes)
   
    elif args.model == 't2t':
         model = T2T_ViT(img_size=img_size, num_classes=n_classes,depth=12, 
                         drop_path_rate=args.sd,is_SPT=args.is_SPT, is_LSA=args.is_LSA,embed_dim=256)
    return model
    
        
    
        
    