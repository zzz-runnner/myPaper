#!/usr/bin/env python3
"""Strict-load final refiners, check a small CPU forward pass and record gate values."""
import json
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from train_teacher_anchor_refiner_from_cache import AnchorResidualUNet
from train_lpsm_anchor_refiner_from_cache import LPSMAnchorResidualUNet

torch.set_num_threads(2)
torch.manual_seed(0)
rows=[]
specs=[
("irstd_s81", "runs/anchor_refiner_multidata_probe/irstd_multi_r05_a002_s81/best.pth.tar",23,64,False),
("nuaa_s1303", "runs/local_rule_lpsm_nuaa_4gpu_screen/s1303_e2m_r006/best.pth.tar",28,56,True),
("nuaa_s1313", "runs/local_rule_lpsm_nuaa_followup_4gpu/s1313_e2m_r0055_a018/best.pth.tar",28,64,True),
]
for name,path,channels,base,lpsm in specs:
    ckpt=torch.load(ROOT/path,map_location="cpu",weights_only=False)
    state=ckpt.get("state_dict",ckpt)
    model=(LPSMAnchorResidualUNet(channels,base,"e2,mid") if lpsm else AnchorResidualUNet(channels,base)).eval()
    model.load_state_dict(state,strict=True)
    with torch.no_grad(): output=model(torch.randn(1,channels,32,32))
    assert output.shape==(1,1,32,32) and torch.isfinite(output).all()
    gates={k:float(v) for k,v in state.items() if k.endswith("alpha") and v.numel()==1}
    rows.append({"name":name,"checkpoint":path,"epoch":ckpt.get("epoch"),"strict_load":True,"forward_shape":list(output.shape),"finite":True,"direction_alpha":gates,"effective_direction_alpha":{k:max(0,min(1,v)) for k,v in gates.items()}})
    print(name,"strict load and CPU forward PASS",flush=True)
(ROOT/"results/checkpoint_validation.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
