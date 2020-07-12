sample_rate = 32000
clip_samples = sample_rate * 30

mel_bins = 64
fmin = 50
fmax = 14000
window_size = 1024
hop_size = 320
window = 'hann'
pad_mode = 'reflect'
center = True
device = 'cuda'
ref = 1.0
amin = 1e-10
top_db = None

labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 
    'pop', 'reggae', 'rock']
    
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}
classes_num = len(labels)