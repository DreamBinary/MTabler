def rewrite():
    import os
    import json
    if os.environ.get('DATA_PATH_B'):
        base_dir = os.environ.get('DATA_PATH_B')
    else:
        base_dir = '/bohr/form-recognition-train-b6y2/v4'
    with open(os.path.join(base_dir, 'dataset.json'), 'r') as f:
        data = json.load(f)
    with open("./data.json", 'w') as f:
        # write path to json
        new_data = []
        for d in data:
            d["path"] = os.path.join(base_dir, "test_images", d["image_path"])
            new_data.append(d)
        json.dump(new_data, f)
import multiprocessing
p = multiprocessing.Process(target=rewrite)
p.start()