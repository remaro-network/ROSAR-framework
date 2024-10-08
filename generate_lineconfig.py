import argparse
import random

from config import IMG_SIZE

def run(seed):
    print('Seed: ' + str(seed))
    random.seed(seed)

    lines = []
    num_groups = random.randint(30, 50)
    indices = random.sample(range(IMG_SIZE[0]), num_groups)
    indices.sort()
    for index in indices:
        num_lines = random.randint(2, 6)

        for line in range(num_lines):
            y = index + line
            if y < IMG_SIZE[0]:
                lines.append(y)

    lines = set(lines)
    with open('line_indices.cfg', 'w') as f:
        for idx in lines:
            print(idx, file=f)

        f.close()

def main(opt):
    run(**vars(opt))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=random.randint(1, 100000000000), help='random seed (int)')
    opt = parser.parse_args()
    print('CMD Arguments:', opt)
    return opt

if __name__=='__main__':
    opt = parse_opt()
    main(opt)