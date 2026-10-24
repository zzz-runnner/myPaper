# Spectral Highbatch GPU1/2/3 Replacement Note

Stopped/replaced because these branches did not beat current reliable bests in early full multi-dataset high-batch training.

- IRSTD-B: best epoch2 mIoU 0.681845, below IRSTD best 0.682817/0.6828 region.
- NUAA-LCA: best epoch1 mIoU 0.791230, then epoch4 0.791003, below/tied current NUAA best 0.7913.
- NUAA-TCD: best epoch0 mIoU 0.791315, then epoch4 0.791003, no improvement and drifting down.

GPU0 IRSTD-A was kept alive because epoch2 reached 0.682817, a tiny but real improvement over the previous 0.6828 region.
