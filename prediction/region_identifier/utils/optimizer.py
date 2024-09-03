import torch 


def get_optimizer(model, base_args, extension_args):
    lr = base_args.learning_rate
    lr_scale = extension_args.lr_scale
    weight_decay = base_args.weight_decay

    optimizer = torch.optim.Adam([
            {'params': model.dnabert2.parameters()},
            {'params': model.classifier_method.parameters(), 'lr': lr*lr_scale}], lr=lr, weight_decay=weight_decay)
    return optimizer 
    

