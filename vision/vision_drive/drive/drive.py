import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import time
import datetime
import argparse
import random
import numpy as np
import timm
from pathlib import Path
import sys
import torchvision.models as tvm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  


def create_torch_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if name.startswith('resnet'):
        if name == 'resnet18':
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        elif name == 'resnet34':
            weights = tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        elif name == 'resnet50':
            weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        elif name == 'resnet101':
            weights = tvm.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
        else:
            raise ValueError(f"Unsupported resnet variant: {name}")

        model = getattr(tvm, name)(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    
    try:
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    except:
        raise ValueError(f"Model {name} not supported in create_torch_model.")


def set_seed(seed=43):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx

try:
    from utils import progress_bar
except ImportError:
    def progress_bar(current, total, msg=None):
        import sys
        cur_len = int(50 * current / total)
        sys.stdout.write('\r')
        sys.stdout.write(' [')
        sys.stdout.write('=' * cur_len)
        sys.stdout.write('>' + '.' * (50 - cur_len - 1))
        sys.stdout.write('] %d/%d ' % (current, total))
        if msg:
            sys.stdout.write(' - ' + msg)
        sys.stdout.flush()
        if current == total - 1:
            print()

DATASET_CONFIGS = {
    'cifar10': {
        'num_classes': 10,
        'dataset': torchvision.datasets.CIFAR10,
        'transform_train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'transform_test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        'input_size': 224,
        'uses_train_arg': True,
        'download': True
    },
    'cifar100': {
        'num_classes': 100,
        'dataset': torchvision.datasets.CIFAR100,
        'transform_train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]),
        'transform_test': transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]),
        'input_size': 224,
        'uses_train_arg': True,
        'download': True
    },
    'flowers102': {
        'num_classes': 102,
        'dataset': lambda root, train, transform, download=False: (
            torch.utils.data.ConcatDataset([
                torchvision.datasets.Flowers102(
                    root=root,
                    split='train',
                    transform=transform,
                    download=download
                ),
                torchvision.datasets.Flowers102(
                    root=root,
                    split='val',
                    transform=transform,
                    download=download
                )
            ]) if train else
            torchvision.datasets.Flowers102(
                root=root,
                split='test',
                transform=transform,
                download=download
            )
        ),
        'transform_train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'transform_test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'input_size': 224,
        'uses_train_arg': True,
        'download': True
    }
}

MODEL_CONFIGS = {
    'vit': ['vit_base_patch32_224', 'vit_base_patch16_224', 'vit_large_patch14_224'],
    'resnet': ['resnet18', 'resnet34', 'resnet50', 'resnet101'],
    'convnext': ['convnext_tiny', 'convnext_small', 'convnext_base'],
    'swin': ['swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224', 'swin_base_patch4_window7_224']
}
AVAILABLE_MODELS = [model for family in MODEL_CONFIGS.values() for model in family]

EPOCH_CONFIGS = {
    'cifar10': 20,
    'cifar100': 25,
    'flowers102': 25
}

def get_dataloaders(dataset_name, data_root='./data', batch_size=128, num_workers=4):
    config = DATASET_CONFIGS[dataset_name]
    if config['uses_train_arg']:
        trainset = config['dataset'](
            root=data_root, train=True, transform=config['transform_train'], download=config.get('download', False))
        testset = config['dataset'](
            root=data_root, train=False, transform=config['transform_test'], download=config.get('download', False))
    else:
        dataset = config['dataset'](
            root=data_root, train=False, transform=config['transform_train'], download=config.get('download', False))
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

    indexed_trainset = IndexedDataset(trainset)
    trainloader = torch.utils.data.DataLoader(
        indexed_trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    extract_loader = torch.utils.data.DataLoader(
        indexed_trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return trainloader, extract_loader, testloader, config['num_classes']

def evaluate_imagenet(net, device):
    net.eval()
    val_dataset = torchvision.datasets.ImageFolder(
        root='/data1/../weight/.._finetuning/vision/data/val',
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    correct = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            _, pred_top5 = outputs.topk(5, 1, True, True)
            pred_top5 = pred_top5.t()
            correct_top5 += pred_top5.eq(targets.view(1, -1).expand_as(pred_top5)).any(dim=0).sum().item()
            total += targets.size(0)
            progress_bar(batch_idx, len(val_loader), 'ImageNet Val: Top-1: %.3f%% | Top-5: %.3f%%'
                         % (100. * correct / total, 100. * correct_top5 / total))
    return 100. * correct / total, 100. * correct_top5 / total

def setup_logger(log_dir, args):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.lambda_mode == 2:
        log_file_name = f'lr{args.lr}_lambda{args.lambda_weight}_fixed.txt'
    elif args.lambda_mode == 0:
        log_file_name = f'lr{args.lr}_lambda_dynamic_0to1.txt'
    elif args.lambda_mode == 1:
        log_file_name = f'lr{args.lr}_lambda_dynamic_1to0.txt'
    else:
        log_file_name = f'lr{args.lr}_lambda_unknown.txt'

    log_file = log_dir / log_file_name

    def log_print(*args_print, **kwargs_print):
        print(*args_print, **kwargs_print)
        with open(log_file, 'a') as f:
            print(*args_print, file=f, **kwargs_print)
    return log_print


def solve_drive_weights(c_scores, Z, args, log_print):
    N, D = Z.shape
    Z_sq = Z.pow(2)
    w = torch.exp(c_scores / args.tau).clamp(min=0.0, max=1.0)
    epsilon = 1e-3
    
    for iteration in range(args.max_iter):
        w_prev = w.clone()
        lambda_w = torch.mv(Z_sq.t(), w) + epsilon
        inv_lambda_w = 1.0 / lambda_w
        kappa = torch.mv(Z_sq, inv_lambda_w)
        kappa = torch.clamp(kappa, max=1e4)
        new_logits = (c_scores + args.gamma * kappa) / args.tau
        w = torch.exp(new_logits / args.tau).clamp(min=0.0, max=1.0)
        diff = torch.norm(w - w_prev, p=1).item()
        if diff < 0.001:
            break
    return w

def train(net, trainloader, optimizer, criterion_noreduce, device, epoch, log_print, global_weights):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets, indices) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        indices = indices.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        losses = criterion_noreduce(outputs, targets)
        batch_weights = global_weights[indices]
        weighted_loss = torch.mean(batch_weights * losses)
        weighted_loss.backward()
        optimizer.step()

        train_loss += weighted_loss.item() 
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    acc = 100. * correct / total
    log_print(f'Epoch: {epoch} | Train Loss: {train_loss / (batch_idx + 1):.3f} | Train Acc: {acc:.3f}%')
    return acc

def test(net, testloader, device, epoch, best_acc, checkpoint_dir, log_print):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%%'
                         % (test_loss / (batch_idx + 1), 100. * correct / total))

    acc = 100. * correct / total
    log_print(f'Epoch: {epoch} | Test Loss: {test_loss / (batch_idx + 1):.3f} | Test Acc: {acc:.3f}%')

    if acc > best_acc:
        log_print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, checkpoint_dir / 'ckpt_best.pth')
        return acc
    return best_acc

def save_original_parameters(model):
    original_params = {
        'fc': {'weight': model.fc.weight.data.clone(), 'bias': model.fc.bias.data.clone()},
        'bn': {}
    }
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            original_params['bn'][name] = {
                'weight': module.weight.data.clone(),
                'bias': module.bias.data.clone(),
                'running_mean': module.running_mean.clone(),
                'running_var': module.running_var.clone()
            }
    return original_params


def main():
    parser = argparse.ArgumentParser(description='DRIVE with PCA Approximation')
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--model', type=str, default='resnet18', choices=AVAILABLE_MODELS)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-root', default='./data', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--seed', default=43, type=int)
    
    parser.add_argument('--temp', default=1.0, type=float, help='temperature for weighting losses (tau)')
    parser.add_argument('--lambda-weight', '--_lambda', dest='lambda_weight', default=0.5, type=float,
                        help="Initial or fixed lambda value (only used if mode=2)")
    
    parser.add_argument('--lambda-mode', type=int, default=2, choices=[0, 1, 2],
                        help='0: dynamic start 0 -> end 1; 1: dynamic start 1 -> end 0; 2: fixed value')

    parser.add_argument('--gamma', default=1, type=float, help='Strength of geometric regularization')
    parser.add_argument('--max-iter', default=10, type=int, help='Max iterations for weight solver')
    
    parser.add_argument('--checkpoint-dir', default='./checkpoints/div_and_dyn', type=str)
    parser.add_argument('--lp-dir', default='./checkpoints/linear_probing', type=str)
    parser.add_argument('--log-dir', default='./logs/div_and_dyn', type=str)
    
    args = parser.parse_args()
    args.tau = args.temp

    epochs = EPOCH_CONFIGS[args.dataset]
    log_dir = Path(args.log_dir) / f'{args.model}/{args.dataset}'
    log_print = setup_logger(log_dir, args)

    log_print("\n=== Configuration ===")
    for k, v in vars(args).items():
        log_print(f"{k}: {v}")
    log_print("===================\n")

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir = Path(args.checkpoint_dir)
    lp_dir = Path(args.lp_dir)
    LOCAL_WEIGHT_PATH = '/data1/../weight/..finetuning/vision/pretrains-models/resnet50.pth' 

    pretrained_model = torchvision.models.resnet50()
    try:
        pretrained_model.load_state_dict(torch.load(LOCAL_WEIGHT_PATH), strict=False)
    except FileNotFoundError:
        log_print(f"Warning: Local weights not found at {LOCAL_WEIGHT_PATH}, using random init for dummy.")
    pretrained_model = pretrained_model.to(device)
    
    if isinstance(pretrained_model, torch.nn.DataParallel):
        original_params = save_original_parameters(pretrained_model.module)
    else:
        original_params = save_original_parameters(pretrained_model)

    log_print(f'\n==> Preparing {args.dataset} dataset')
    trainloader, extract_loader, testloader, num_classes = get_dataloaders(
        args.dataset, args.data_root, args.batch_size, args.workers
    )
    N = len(trainloader.dataset)
    log_print(f"Number of classes: {num_classes}, Train Samples: {N}")

    net = create_torch_model(args.model, num_classes=num_classes, pretrained=False).to(device)
    lp_model = create_torch_model(args.model, num_classes=num_classes, pretrained=False).to(device)

    lp_path = lp_dir / f'{args.model}/{args.dataset}/ckpt_linear_best.pth'
    if lp_path.exists():
        log_print(f"Loading linear-probe checkpoint from {lp_path}")
        lp_checkpoint = torch.load(lp_path, map_location=device)
        net.load_state_dict(lp_checkpoint['net'], strict=False)
        lp_model.load_state_dict(lp_checkpoint['net'], strict=False)
    else:
        log_print(f"WARNING: Linear probe checkpoint not found at {lp_path}. Using ImageNet weights.")
        net = create_torch_model(args.model, num_classes=num_classes, pretrained=True).to(device)
        lp_model = create_torch_model(args.model, num_classes=num_classes, pretrained=True).to(device)

    lp_model.eval()

    feature_storage = {}
    def get_features_hook(module, input, output):
        if len(output.shape) == 4:
            feature_storage['feat'] = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(output, (1, 1)), 1)
        else:
            feature_storage['feat'] = output

    handle = lp_model.fc.register_forward_hook(get_features_hook)
    f_old = torch.zeros(N, device=device)
    features_list = []
    criterion_noreduce = nn.CrossEntropyLoss(reduction='none')

    log_print("==> Pre-computing f_old and extracting features for PCA...")
    with torch.no_grad():
        for batch_idx, (inputs, targets, indices) in enumerate(extract_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            indices = indices.to(device, non_blocking=True)
            outputs = lp_model(inputs)
            losses = criterion_noreduce(outputs, targets)
            f_old[indices] = losses
            if 'feat' in feature_storage:
                features_list.append(feature_storage['feat'])
            else:
                 feat = lp_model.layer4(lp_model.layer3(lp_model.layer2(lp_model.layer1(lp_model.maxpool(lp_model.relu(lp_model.bn1(lp_model.conv1(inputs))))))))
                 feat = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(feat, (1,1)), 1)
                 features_list.append(feat)
            progress_bar(batch_idx, len(extract_loader), 'Extracting')
    handle.remove()
    
    log_print("==> Performing PCA on extracted features...")
    all_features = torch.cat(features_list, dim=0)
    N_samples, D_dim = all_features.shape
    mu = torch.mean(all_features, dim=0, keepdim=True)
    X_bar = all_features - mu
    Sigma = (X_bar.t() @ X_bar) / N_samples
    L, U = torch.linalg.eigh(Sigma)
    Z = X_bar @ U
    Z = torch.nn.functional.normalize(Z, p=2, dim=1) 
    del all_features, X_bar, Sigma, L, U
    torch.cuda.empty_cache()
    
    f_new_prev = f_old.clone()
    backbone_lr_scale = 0.5
    head_params = list(map(id, net.fc.parameters()))
    backbone_params = filter(lambda p: id(p) not in head_params, net.parameters())

    optimizer = optim.SGD([
        {'params': backbone_params, 'lr': args.lr * backbone_lr_scale},
        {'params': net.fc.parameters(), 'lr': args.lr}
    ], momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log_print(f'\n==> Training {args.model} on {args.dataset} with DRIVE-PCA(div and dyn)...')
    log_print(f'==> Lambda Strategy Mode: {args.lambda_mode}')
    best_acc = 0
    start_time = time.time()

    for epoch in range(epochs):
        log_print(f'\nEpoch: {epoch + 1}/{epochs}')

        if args.lambda_mode == 2:
            current_lambda = args.lambda_weight
        else:
            if epochs > 1:
                progress = epoch / (epochs - 1)
            else:
                progress = 1.0 # Fallback for 1 epoch case

            if args.lambda_mode == 0:
                # 0 -> 1
                current_lambda = progress
            elif args.lambda_mode == 1:
                # 1 -> 0
                current_lambda = 1.0 - progress
            
            # Ensure float type
            current_lambda = float(current_lambda)

        log_print(f'    Current Lambda Weight: {current_lambda:.4f} (Mode {args.lambda_mode})')

        c_scores = current_lambda * (-f_old) + (1.0 - current_lambda) * f_new_prev

        global_weights = solve_drive_weights(c_scores, Z, args, log_print)
        
        train_acc = train(
            net, trainloader, optimizer, criterion_noreduce,
            device, epoch + 1, log_print, global_weights
        )
        
        best_acc = test(net, testloader, device, epoch + 1, best_acc, checkpoint_dir, log_print)
        scheduler.step()
        
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets, indices) in enumerate(extract_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                indices = indices.to(device, non_blocking=True)
                outputs = net(inputs)
                losses = criterion_noreduce(outputs, targets)
                f_new_prev[indices] = losses

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log_print(f'Training time: {total_time_str}')
    log_print(f'Best accuracy: {best_acc}%')

    log_print("Evaluating best checkpoint on ImageNet...")
    checkpoint_path = checkpoint_dir / f'ckpt_best.pth'
    if not checkpoint_path.exists():
        log_print(f"Checkpoint {checkpoint_path} does not exist.")
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        imagenet_model = create_torch_model(args.model, 1000, pretrained=False).to(device)
        checkpoint_dict = checkpoint['net']
        imagenet_dict = imagenet_model.state_dict()

        for k in checkpoint_dict.keys():
            if not ('fc' in k or 'head' in k or 'bn' in k or 'downsample.1' in k):
                if k in imagenet_dict:
                    imagenet_dict[k] = checkpoint_dict[k]
                elif f'module.{k}' in imagenet_dict:
                    imagenet_dict[f'module.{k}'] = checkpoint_dict[k]

        if isinstance(imagenet_model, torch.nn.DataParallel):
            imagenet_dict['module.fc.weight'] = original_params['fc']['weight'].to(device)
            imagenet_dict['module.fc.bias'] = original_params['fc']['bias'].to(device)
            for name, params in original_params['bn'].items():
                imagenet_dict[f'module.{name}.weight'] = params['weight'].to(device)
                imagenet_dict[f'module.{name}.bias'] = params['bias'].to(device)
                imagenet_dict[f'module.{name}.running_mean'] = params['running_mean'].to(device)
                imagenet_dict[f'module.{name}.running_var'] = params['running_var'].to(device)
        else:
            imagenet_dict['fc.weight'] = original_params['fc']['weight']
            imagenet_dict['fc.bias'] = original_params['fc']['bias']
            for name, params in original_params['bn'].items():
                imagenet_dict[f'{name}.weight'] = params['weight']
                imagenet_dict[f'{name}.bias'] = params['bias']
                imagenet_dict[f'{name}.running_mean'] = params['running_mean']
                imagenet_dict[f'{name}.running_var'] = params['running_var']

        imagenet_model.load_state_dict(imagenet_dict)
        top1_acc, top5_acc = evaluate_imagenet(imagenet_model, device)
        log_print("\n=== Final Results ===")
        log_print(f"Dataset: {args.dataset}")
        log_print(f"Model: {args.model}")
        log_print(f"Mode: {args.lambda_mode}")
        log_print(f"Best accuracy on {args.dataset}: {best_acc:.2f}%")
        log_print(f"ImageNet accuracy: Top-1: {top1_acc:.2f}% | Top-5: {top5_acc:.2f}%")
        log_print("===================")

if __name__ == '__main__':
    main()