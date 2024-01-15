# %%
import eqnet
from eqnet.utils import detect_peaks, extract_picks
from dataclasses import dataclass
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Union
import torch
import torch.nn.functional as F
import pandas as pd


@dataclass
class Config:
    model = "phasenet_das"
    backbone = "unet"
    phases = ["P", "S"]
    device = "cuda"
    min_prob = 0.5
    amp = True
    dtype = torch.float32
    location = None


def padding(data, min_nt=1024, min_nx=1024):
    nt, nx = data.shape[-2:]
    pad_nt = (min_nt - nt % min_nt) % min_nt
    pad_nx = (min_nx - nx % min_nx) % min_nx
    with torch.no_grad():
        data = F.pad(data, (0, pad_nx, 0, pad_nt), mode="constant")
    return data


class Data(BaseModel):
    id: List[str]
    timestamp: List[str]
    vec: Union[List[List[List[float]]], List[List[float]]]
    dt_s: Optional[float] = 0.01


def load_model(args):
    model = eqnet.models.__dict__[args.model].build_model(
        backbone=args.backbone,
        in_channels=1,
        out_channels=(len(args.phases) + 1),
    )

    if args.model == "phasenet" and (not args.add_polarity):
        raise ("No pretrained model for phasenet, please use phasenet_polarity instead")
    elif (args.model == "phasenet") and (args.add_polarity):
        model_url = "https://github.com/AI4EPS/models/releases/download/PhaseNet-Polarity-v3/model_99.pth"
    elif args.model == "phasenet_das":
        if args.location is None:
            # model_url = "ai4eps/model-registry/PhaseNet-DAS:latest"
            model_url = "https://github.com/AI4EPS/models/releases/download/PhaseNet-DAS-v0/PhaseNet-DAS-v0.pth"
            # model_url = "https://github.com/AI4EPS/models/releases/download/PhaseNet-DAS-v1/PhaseNet-DAS-v1.pth"
        elif args.location == "forge":
            model_url = "https://github.com/AI4EPS/models/releases/download/PhaseNet-DAS-ConvertedPhase/model_99.pth"
        else:
            raise ("Missing pretrained model for this location")
    else:
        raise
    state_dict = torch.hub.load_state_dict_from_url(
        model_url, model_dir="./", progress=True, check_hash=True, map_location="cpu"
    )
    model.load_state_dict(state_dict["model"], strict=True)

    return model


###################### FastAPI ######################
app = FastAPI()
args = Config()
model = load_model(args)
model.to(args.device)
model.eval()


@app.post("/predict")
def predict(meta: Data):
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=args.amp):
            print(meta["nt"], meta["nx"])
            output = model(meta)["phase"][:, :, : meta["nt"], : meta["nx"]]
            scores = torch.softmax(output, dim=1)  # [batch, nch, nt, nsta]
            topk_scores, topk_inds = detect_peaks(scores, vmin=args.min_prob, kernel=21)

            picks = extract_picks(
                topk_inds,
                topk_scores,
                file_name=meta["id"],
                begin_time=meta["timestamp"] if "timestamp" in meta else None,
                dt=meta["dt_s"] if "dt_s" in meta else 0.01,
                vmin=args.min_prob,
                phases=args.phases,
            )

    return {"picks": picks}


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# %%
if __name__ == "__main__":
    # %%
    import h5py
    import numpy as np
    import matplotlib.pyplot as plt

    h5_file = "ci37238204.h5"

    with h5py.File(h5_file, "r") as f:
        vec = f["data"][:].T
        vec = vec[np.newaxis, :, :]
        timestamp = f["data"].attrs["begin_time"]
        data_id = f'{f["data"].attrs["event_id"]}'

    # %%
    data = torch.tensor(vec, dtype=args.dtype).unsqueeze(0)  # [batch, nch, nt, nsta]
    nt, nx = data.shape[-2:]
    data = padding(data)
    meta = {"id": [data_id], "timestamp": [timestamp], "data": data, "dt_s": 0.01, "nx": nx, "nt": nt}
    picks = predict(meta)["picks"]
    data = data[:, :, :nt, :nx]

    # %%
    picks = picks[0]  ## batch size = 1
    picks = pd.DataFrame.from_dict(picks, orient="columns")

    # %%
    plt.figure()
    vmax = torch.std(data[0, 0]) * 3
    vmin = -vmax
    plt.imshow(data[0, 0], vmin=vmin, vmax=vmax, aspect="auto", cmap="seismic", interpolation="none")
    color = picks["phase_type"].map({"P": "red", "S": "blue"})
    plt.scatter(picks["station_id"], picks["phase_index"], c=color, s=1)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig("test_v2.png")
    plt.show()

# %%
