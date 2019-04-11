import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Analysis Time')
    parser.add_argument('log_path', help='path of train log in json format')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    log_path = args.log_path
    logs = []
    with open(log_path, 'r') as f:
        for l in f:
            log = json.loads(l.strip())
            if log['mode'] != 'val':
                logs.append(log)

    epochs = set([log['epoch'] for log in logs])
    train_times = dict()
    all_times = list()
    for epoch in epochs:
        train_times[epoch] = [
            log['time'] for log in logs
            if log['epoch'] == epoch and log['mode'] != 'val'
        ]
        train_times[epoch].pop(0)  # first iter of each epoch may be very slow
        all_times += train_times[epoch]
    print('average iter time: {:.3f} s/iter'.format(
        sum(all_times) / float(len(all_times))))
