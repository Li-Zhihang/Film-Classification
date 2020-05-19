import os

for f in os.listdir('./moonp4arrivalp1p2p3'):
    if f.split('.')[-1] == 'mp4':
        base = f.split('.')[0]
        if not os.path.exists(os.path.join('./moonp4arrivalp1p2p3', base + '.output')):
            os.rename('./moonp4arrivalp1p2p3/' + f, './res/' + f)