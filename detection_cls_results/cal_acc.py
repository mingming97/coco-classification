import os

if __name__ == '__main__':
    file = 'coco_train_81_clsres.log'
    correct, tot = 0, 0
    with open(file) as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) != 0:
                x, y = line.split('/')
                correct += int(x)
                tot += int(y)

    print('acc: {:.6f}({}/{})'.format(correct / tot, correct, tot))
