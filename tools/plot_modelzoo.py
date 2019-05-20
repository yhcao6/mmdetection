import matplotlib.pyplot as plt


def read_model_zoo(model_zoo_path):
    model_zoo = dict()
    with open(model_zoo_path, 'r') as f:
        meet_model = False
        for line in f:
            line = line.strip()
            if '###' in line:
                meet_model = False
                if 'R-CNN' in line or 'Retinanet' in line or 'HTC' in line:
                    meet_model = True
                    model_series = line.replace('###', '').strip()
                    if 'Fast R-CNN' in model_series:
                        model_series = 'Fast R-CNN'
                    if 'HTC' in model_series:
                        model_series = 'HTC'
                    model_zoo[model_series] = dict()

            if meet_model:
                if 'Backbone' in line:
                    metric_keys = []
                    for k in line.split('|')[1:-2]:
                        if 'Mem' in k:
                            k = 'Mem'
                        if 'Inf time' in k:
                            k = 'Inf time'
                        if 'Train time' in k:
                            k = 'Train time'
                        metric_keys.append(k.strip())
                    mem_inds = 0
                    while 'Mem' not in metric_keys[0]:
                        metric_keys.pop(0)
                        mem_inds += 1

                if 'FPN' in line:
                    if 'Fast R-CNN' in model_series and 'Mask' in line:
                        model_series = 'Fast Mask R-CNN'
                        model_zoo[model_series] = dict()

                    model_name = []
                    for i, v in enumerate(line.split('|')[1:-2][:mem_inds]):
                        model_name.append(v.strip())
                    model_name.insert(0, model_series)
                    model_name = '-'.join(model_name)

                    metric_values = []
                    for i, v in enumerate(
                            line.split('|')[1:][mem_inds:mem_inds +
                                                len(metric_keys)]):
                        if '-' not in v:
                            metric_values.append(float(v.strip()))
                        else:
                            if '2x' in model_name or '20e' in model_name:
                                metric_values.append(last_metric_values[i])
                            else:
                                metric_values.append(v.strip())
                    last_metric_values = metric_values.copy()

                    model_zoo[model_series][model_name] = dict()
                    assert len(metric_values) == len(metric_keys)
                    for k, v in zip(metric_keys, metric_values):
                        model_zoo[model_series][model_name][k] = v

    return model_zoo


def draw_box_ap(model_zoo):
    markers = ['.', '^', '1', 's', 'p', '*', 'x', 'D']

    model_count = 0
    for model_series, model_series_dict in model_zoo.items():
        print('###{}'.format(model_series))
        inf_times = []
        box_aps = []
        for model_name, metric_dict in model_series_dict.items():
            if ('pytorch-2x' not in model_name) and (
                    'pytorch-20e' not in model_name) and (
                        'Faster-2x' not in model_name) and (
                            'Mask-2x' not in model_name):
                continue
            for k, v in metric_dict.items():
                if 'Inf time' in k:
                    inf_time = v
                    inf_times.append(inf_time)
                if 'box AP' in k:
                    box_ap = v
                    box_aps.append(box_ap)
            print('{}, inf_time: {}, box_ap: {}'.format(
                model_name, inf_time, box_ap))
        plt.scatter(
            inf_times,
            box_aps,
            marker=markers[model_count],
            label=model_series)
        plt.legend()
        model_count += 1

    plt.show()


def draw_mask_ap(model_zoo):
    markers = ['.', '^', '1', 's', 'p', '*', 'x', 'D']

    model_count = 0
    for model_series, model_series_dict in model_zoo.items():
        print('###{}'.format(model_series))
        inf_times = []
        mask_aps = []
        if 'Mask' not in model_series and 'HTC' not in model_series:
            continue
        for model_name, metric_dict in model_series_dict.items():
            if ('pytorch-2x' not in model_name) and (
                    'pytorch-20e' not in model_name) and (
                        'Faster-2x' not in model_name) and (
                            'Mask-2x' not in model_name):
                continue
            for k, v in metric_dict.items():
                if 'Inf time' in k:
                    inf_time = v
                    inf_times.append(inf_time)
                if 'mask AP' in k:
                    mask_ap = v
                    mask_aps.append(mask_ap)
            print('{}, inf_time: {}, mask_ap: {}'.format(
                model_name, inf_time, mask_ap))
        plt.scatter(
            inf_times,
            mask_aps,
            marker=markers[model_count],
            label=model_series)
        plt.legend()
        model_count += 1

    plt.show()


if __name__ == '__main__':
    model_zoo = read_model_zoo('MODEL_ZOO.MD')
    draw_box_ap(model_zoo)
    draw_mask_ap(model_zoo)
