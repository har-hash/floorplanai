import pickle
data_path = r"H:\EDAI 2\Interface\static\Data\data_train_converted.pkl"
output_path = r"H:\EDAI 2\Interface\python_out3.txt"

with open(output_path, 'w', encoding='utf-8') as out_f:
    def p(text):
        out_f.write(str(text) + '\n')

    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        samples = data['data']
        sample = samples[0]
        p(f"box: {sample.box}")
        p(f"edge: {sample.edge}")
        p(f"order: {sample.order}")
        if hasattr(sample, 'rBoundary'):
            for i, r in enumerate(sample.rBoundary[:2]):
                p(f"rBoundary[{i}]: {r}")

        rNum_path = r"H:\EDAI 2\Interface\static\Data\rNum_train.npy"
        import numpy as np
        if __import__('os').path.exists(rNum_path):
            rNum = np.load(rNum_path)
            p(f"rNum shape: {rNum.shape}, first 5: {rNum[:5]}")
            
    except Exception as e:
        p(f"Error: {e}")
