import argparse
import torch
import torch.optim
import torch.utils.data
import shutil
import utils

from models import Model
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def save_checkpoint(state, is_best, path):
    torch.save(state, path)
    if is_best:
        shutil.copy(path, '/'.join(path.split('/')[:-1]) + '/kr_aihub_best_checkpoint_210618.pt')


def load_checkpoint(args):
    checkpoint = torch.load(args.resume)

    model = Model(num_class=31)
    model.load_state_dict(checkpoint['state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer


def run(args):
    device = torch.device('cuda')
    train_set = utils.dataset('train', 20, trunc=args.trunc)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, pin_memory=True, shuffle=True,
                              collate_fn=utils.collate_fn)
    val_set = utils.dataset('val')
    val_loader = DataLoader(val_set, batch_size=args.valid_batch_size, pin_memory=True, shuffle=False,
                            collate_fn=utils.collate_fn)

    if args.resume:
        model, optimizer = load_checkpoint(args)
    else:
        model = Model(num_class=31, vocab_size=4563)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    model = model.to(device)
    scaler = GradScaler()
    pbar = tqdm(total=args.steps, ncols=120)

    step = 0
    val_acc = 0
    avg_loss = 0
    acc = 0
    total = 1
    best_acc = 0

    while step < args.steps:
        for claims, bios, length in train_loader:
            claims = torch.tensor(claims).to(device)
            bios = torch.tensor(bios).to(device)
            length = torch.tensor(length).to('cpu')
            if step >= args.steps:
                break
            if step % args.pf == 0 and step != 0:
                pbar.set_postfix({'loss': avg_loss, 'accuracy': acc / total, 'valid_acc': val_acc})
                pbar.update(args.pf)
                avg_loss = 0
                acc = 0
                total = 0

            optimizer.zero_grad()

            with autocast():
                logits, loss = model(claims, bios, length)

                avg_loss += loss.item()
                total += length.sum().item()
                acc += sum([(logits.argmax(2)[i][:_] == bios[i][:_]).sum().item() for i, _ in enumerate(length)])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1

            if step % args.save_step == 0 and step != 0:
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    val_correct = 0
                    val_total = 0

                    for claims, bios, length in tqdm(val_loader, desc='validation'):
                        with autocast():
                            claims = torch.tensor(claims).to(device)
                            bios = torch.tensor(bios).to(device)
                            length = torch.tensor(length).to('cpu')

                            logits, loss = model(claims, bios, length)
                            val_loss += loss.item() * claims.size(0)
                            val_total += length.sum().item()
                            val_correct += sum(
                                [(logits.argmax(2)[i][:_] == bios[i][:_]).sum().item() for i, _ in enumerate(length)])

                    val_acc = val_correct / val_total
                    valid_loss = val_loss / len(val_loader.dataset)
                    save_checkpoint({'state_dict': model.state_dict(),
                                     'steps': step,
                                     'valid_acc': val_acc,
                                     'validation_loss': valid_loss,
                                     'optimizer': optimizer.state_dict()}, False if val_acc < best_acc else True,
                                    './checkpoints/kr_aihub_best_210618%d.pt' % step)
                    if val_acc > best_acc:
                        best_acc = val_acc
                model.train()


def main():
    parser = argparse.ArgumentParser(description='Keyword Generation Model')
    parser.add_argument('--data', metavar='data_path',
                        help='path to dataset')
    parser.add_argument('--steps', default=100000, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--save_step', default=10000, type=int,
                        help='number of steps to save checkpoint')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='mini-batch size (default: 8), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--valid_batch_size', default=8, type=int,
                        help='mini-batch size (default: 8), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float,
                        help='initial learning rate')
    parser.add_argument('--pf', '--print_freq', default=10, type=int, dest='pf',
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--world_size', default=0, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', default='localhost', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=777, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--trunc', default=512, type=int,
                        help='Maximum number of words to truncate a sentence.')

    args = parser.parse_args()
    ngpus_per_node = args.world_size
    run(args)


if __name__ == '__main__':
    main()