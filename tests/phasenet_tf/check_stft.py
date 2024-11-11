# %%
# %matplotlib inline
from scipy.signal import stft
import matplotlib as mpl
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# event_id='us7000mrzw'
event_id = "us6000m12i"
# sta_id='N4.M63A.00.HH'
sta_id = "CN.EFO..HH"

# H5_path='/data/hy4/kw2988/LCSN/LCSN_all/waveform_20240612_2024_firstarrival.h5'
H5_path='/nfs/quakeflow_dataset/LCSN/waveform_20240612_2024_manual.h5'

class STFT(nn.Module):
    def __init__(
        self,
        n_fft=128,
        hop_length=4,
        window_fn=torch.hann_window,
        log_transform=True,
        normalize=False,
        magnitude=True,
        phase=False,
        grad=False,
        discard_zero_freq=False,
        # select_freq=False,
        **kwargs,
    ):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_fn = window_fn
        self.log_transform = log_transform
        self.normalize = normalize
        self.magnitude = magnitude
        self.phase = phase
        self.grad = grad
        self.discard_zero_freq = discard_zero_freq
        # self.select_freq = select_freq
        self.register_buffer("window", window_fn(n_fft))
        self.window_fn = window_fn
        # if select_freq:
        #     dt = kwargs["dt"]
        #     fmax = 1.0 / 2.0 / dt
        #     freq = torch.linspace(0, fmax, n_fft // 2 + 1)  # Use n_fft // 2 + 1 to get correct frequency bins
        #     idx = torch.arange(n_fft // 2 + 1)[(freq > kwargs["fmin"]) & (freq < kwargs["fmax"])]
        #     self.freq_start = idx[0].item()
        #     self.freq_length = idx.numel()

    def forward(self, x):
        bt, ch, nt = x.shape
        x = x.view(-1, nt)  # bt*ch, nt
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            window=self.window,
            hop_length=self.hop_length,
            center=True,
            return_complex=True,
        )
        stft = torch.view_as_real(stft)
        stft = stft[..., : x.shape[-1] // self.hop_length, :]  # bt*ch, nf, nt, 2
        if self.discard_zero_freq:
            stft = stft.narrow(dim=-3, start=1, length=stft.shape[-3] - 1)
        # if self.select_freq:
        #     stft = stft.narrow(dim=-3, start=self.freq_start, length=self.freq_length)
        if self.magnitude:
            stft_mag = torch.norm(stft, dim=-1, keepdim=True)  # bt* ch, nf, nt
            if self.log_transform:
                stft_mag = torch.log(1 + F.relu(stft_mag))
            if self.phase:
                components = stft.split(1, dim=-1)
                stft_phase = torch.atan2(components[1].squeeze(-1), components[0].squeeze(-1))
                stft = torch.stack([stft_mag, stft_phase], dim=-1)
            else:
                stft = stft_mag
        else:
            nf, nt, ni = stft.shape[-3:]
            stft = stft.view(bt, ch, nf, nt, ni)  # bt, ch, nf, nt, 2
            stft = stft.permute(0, 1, 4, 3, 2).contiguous().view(bt, ch * ni, nt, nf)  # bt, ch*2, nt, nf
            if self.log_transform:
                stft = torch.log(1 + F.relu(stft)) - torch.log(1 + F.relu(-stft))
        if self.normalize:
            vmax = torch.max(torch.abs(stft), dim=-3, keepdim=True)[0]
            vmax[vmax == 0.0] = 1.0
            stft = stft / vmax
        return stft

stft_func = STFT(
    n_fft=64,
    hop_length=4,
    # hop_length=32,
    window_fn=torch.hann_window,
    log_transform=False,
    magnitude=False,
    phase=False,
    discard_zero_freq=True,
)

with h5py.File(H5_path,'r') as f:

    print("Keys: %s" % f.keys())
    # print(len(f.keys()))
    # print(type(f[event_id]))
    print("Keys: %s" % f[event_id].keys()) 
    # print(len(f[event_id].keys()))
    # print(f[event_id].attrs.keys())
    # print(f[event_id][sta_id].attrs.get('snr'))
    component=f[event_id][sta_id].attrs.get('component')
    instrument=f[event_id][sta_id].attrs.get('instrument')
    data=f[event_id].get(sta_id)[:]
    phase_type=f[event_id][sta_id].attrs.get('phase_type')
    phase_index=f[event_id][sta_id].attrs.get('phase_index')
    begin_time=f[event_id].attrs.get('begin_time')
    end_time=f[event_id].attrs.get('end_time')

plt.rcParams.update({'font.size': 22})
t=np.arange(0,data.shape[1])/40
fig=plt.figure(figsize=(15,25))
ax1=plt.subplot(611)
plt.plot(t,data[0,:],'k',label=instrument+component[0])
plt.legend(loc='upper right')
for i in range(len(phase_index)):
    if 'P' in phase_type[i]:
        plt.axvline(phase_index[i]/40,color='b',linestyle='--')
    elif 'S' in phase_type[i]:
        plt.axvline(phase_index[i]/40,color='r',linestyle='--')
# ev_newpick=picks_FP[(picks_FP['station_id']==sta_id)*picks_FP['event_id']==event_id]
# if len(ev_newpick)>0:
#     for i in range(len(ev_newpick)):
#         if 'P' in ev_newpick['phase_type'].iloc[i]:
#             plt.axvline(ev_newpick['phase_index'].iloc[i]/40,color='tab:green',linestyle='--',alpha=0.8)
#         elif 'S' in ev_newpick['phase_type'].iloc[i]:
#             plt.axvline(ev_newpick['phase_index'].iloc[i]/40,color='tab:pink',linestyle='--',alpha=0.8)

# plt.xlim(0,data.shape[1]/40)

ax2=plt.subplot(612)
normalizer = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True)+ 1e-6)
f, t, Zxx = stft(data[0], fs=40, nperseg=64,nfft=64,noverlap=64-4)
# plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud',cmap='jet',norm=mpl.colors.LogNorm())
# Zxx = normalizer(np.abs(Zxx.real))
Zxx = Zxx.real
Zxx = normalizer(Zxx)
Zxx = Zxx[1:, :]
# Zxx = Zxx.real
vmax = np.max(Zxx)/10
# plt.pcolormesh( np.abs(Zxx), shading='gouraud',cmap='jet',norm=mpl.colors.LogNorm())
# plt.pcolormesh( Zxx, cmap='seismic',vmax=vmax, vmin=-vmax)
plt.imshow(Zxx, cmap='seismic',vmax=vmax, vmin=-vmax, aspect='auto', origin='lower')
plt.title('scipy') 
print(f"scipy: {Zxx.shape}")

ax2=plt.subplot(613)
x = torch.tensor(data[0])
stft_ = torch.stft(
    x,
    n_fft=64,
    window=torch.hann_window(64),
    hop_length=4,
    # hop_length=128-32,
    center=True,
    return_complex=True,
)
# stft = torch.view_as_real(stft)
# Zxx = stft.numpy()
Zxy = stft_.numpy().real
print(f"torch: {Zxy.shape}")
# Zxy = normalizer(np.abs(Zxy))
Zxy = normalizer(Zxy)
Zxy = Zxy[1:, :]
vmax = np.max(Zxy)
plt.pcolormesh(Zxy, cmap='seismic',vmax=vmax, vmin=-vmax)

# if len(ev_newpick)>0:
#     for i in range(len(ev_newpick)):
#         if 'P' in ev_newpick['phase_type'].iloc[i]:
#             plt.axvline(ev_newpick['phase_index'].iloc[i]/40,color='tab:green',linestyle='--',alpha=0.8)
#         elif 'S' in ev_newpick['phase_type'].iloc[i]:
#             plt.axvline(ev_newpick['phase_index'].iloc[i]/40,color='tab:pink',linestyle='--',alpha=0.8)

plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('torch') 


ax3=plt.subplot(614)
x = torch.tensor(data[0])
x = x.unsqueeze(0).unsqueeze(0)
stft_ = stft_func(x)
stft_ = stft_[0, 0, :, :].T
Zxz = stft_.numpy()
print(f"torch: {Zxz.shape}")
# Zxz = normalizer(np.abs(Zxz))
Zxz = normalizer(Zxz)
vmax = np.max(Zxz)/10
# plt.pcolormesh(Zxz, cmap='seismic',vmax=vmax, vmin=-vmax)
plt.imshow(Zxz, cmap='seismic',vmax=vmax, vmin=-vmax, aspect='auto', origin='lower')
plt.title('phasenet') 

plt.tight_layout() 
raise


ax3=plt.subplot(613)
t=np.arange(0,data.shape[1])/40
plt.plot(t,data[1,:],'k',label=instrument+component[1])
plt.legend(loc='upper right')
for i in range(len(phase_index)):
    if 'P' in phase_type[i]:
        plt.axvline(phase_index[i]/40,color='b',linestyle='--')
    elif 'S' in phase_type[i]:
        plt.axvline(phase_index[i]/40,color='r',linestyle='--')
# if len(ev_newpick)>0:
#     for i in range(len(ev_newpick)):
#         if 'P' in ev_newpick['phase_type'].iloc[i]:
#             plt.axvline(ev_newpick['phase_index'].iloc[i]/40,color='tab:green',linestyle='--',alpha=0.8)
#         elif 'S' in ev_newpick['phase_type'].iloc[i]:
#             plt.axvline(ev_newpick['phase_index'].iloc[i]/40,color='tab:pink',linestyle='--',alpha=0.8)
plt.xlim(0,data.shape[1]/40)

ax4=plt.subplot(614)
if data[1].max()>0:
    f, t, Zxx = stft(data[1], fs=40, nperseg=128,nfft=128,noverlap=32)
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud',cmap='jet',norm=mpl.colors.LogNorm())
    # if len(ev_newpick)>0:
    #     for i in range(len(ev_newpick)):
    #         if 'P' in ev_newpick['phase_type'].iloc[i]:
    #             plt.axvline(ev_newpick['phase_index'].iloc[i]/40,color='tab:green',linestyle='--',alpha=0.8)
    #         elif 'S' in ev_newpick['phase_type'].iloc[i]:
    #             plt.axvline(ev_newpick['phase_index'].iloc[i]/40,color='tab:pink',linestyle='--',alpha=0.8)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

ax5=plt.subplot(615)
t=np.arange(0,data.shape[1])/40
plt.plot(t,data[2,:],'k',label=instrument+component[2])
plt.legend(loc='upper right')
for i in range(len(phase_index)):
    if 'P' in phase_type[i]:
        plt.axvline(phase_index[i]/40,color='b',linestyle='--')
    elif 'S' in phase_type[i]:
        plt.axvline(phase_index[i]/40,color='r',linestyle='--')
# if len(ev_newpick)>0:
#     for i in range(len(ev_newpick)):
#         if 'P' in ev_newpick['phase_type'].iloc[i]:
#             plt.axvline(ev_newpick['phase_index'].iloc[i]/40,color='tab:green',linestyle='--',alpha=0.8)
#         elif 'S' in ev_newpick['phase_type'].iloc[i]:
#             plt.axvline(ev_newpick['phase_index'].iloc[i]/40,color='tab:pink',linestyle='--',alpha=0.8)
plt.xlim(0,data.shape[1]/40)

ax6=plt.subplot(616)
f, t, Zxx = stft(data[2], fs=40, nperseg=128,nfft=128,noverlap=32)
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud',cmap='jet',norm=mpl.colors.LogNorm())
# if len(ev_newpick)>0:
#     for i in range(len(ev_newpick)):
#         if 'P' in ev_newpick['phase_type'].iloc[i]:
#             plt.axvline(ev_newpick['phase_index'].iloc[i]/40,color='tab:green',linestyle='--',alpha=0.8)
#         elif 'S' in ev_newpick['phase_type'].iloc[i]:
#             plt.axvline(ev_newpick['phase_index'].iloc[i]/40,color='tab:pink',linestyle='--',alpha=0.8)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')

plt.show()
fig.savefig('debug.png')
# %%
