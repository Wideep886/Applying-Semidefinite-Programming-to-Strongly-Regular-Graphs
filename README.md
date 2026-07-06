# Applying-Semidefinite-Programming-to-Strongly-Regular-Graphs

# Fast K-Point SDP Bound

本儲存庫收錄碩士論文中所使用的 **k-point SDP bound** Julia 程式碼，用於球面上兩距離集合的半正定規劃上界計算，以及 SRG（強正則圖）不存在性證明。

程式基於 de Laat et al. 的 [KPointBound](https://arxiv.org/abs/1812.06045) 工作流程改寫，並依論文定理 4.5 與 Remark 4.8，將 PSD 變數建在可行狀態空間 \(|U_R|\) 上（而非完整 \(3^m\) 單項式格點再投影），以加速大規模實例。

## 狀態聲明

**本程式碼目前仍處試作階段**，API 與數值行為可能隨版本調整。作者將不定期更新程式版本，並補充執行數據與結果表格。使用前請以最新 commit 為準。

## 檔案

| 檔案 | 說明 |
|------|------|
| `fast_KPoint_bound_f.jl` | 模組化三階段 API（template / fill / solve） |

## 環境需求

- **Julia** 1.x
- Julia 套件：`Nemo`, `Combinatorics`, `IterTools`, `LinearAlgebra`, `Printf`

```julia
using Pkg
Pkg.add(["Nemo", "Combinatorics", "IterTools"])
```

- 外部求解器：**sdpa_gmp**（建議使用 patched 版本）  
  http://www.daviddelaat.nl/sdpa-gmp-7.1.3.tar.gz

將 `sdpa_gmp` 放在 `PATH`，或設定 `ENV["SDPAGMP_PATH"]`，或放在 `./bin/` / `./` 目錄。

## 快速開始

```bash
julia --startup-file=no fast_KPoint_bound_f.jl
```

或在 Julia REPL 中：

```julia
include("fast_KPoint_bound_f.jl")
using .FastKPointBoundF

# n=頂點數, k=點層級, d=Gegenbauer 截斷, D=[a,b]=兩內積
solve_k_point_bound(37, 5, 2, [-5//52, 7//26]; verbose=true)
```

### SRG 不存在性範例（論文 §5.3）

```julia
# (550, 387, 260, 301) → floor(α) = 546 < 550
solve_k_point_bound(33, 5, 2, [-1//9, 7//27])

# (703, 520, 372, 420) → floor(α) = 662 < 703
solve_k_point_bound(37, 5, 2, [-5//52, 7//26])
```

## API 概覽

```julia
tpl  = build_kpoint_template(k, d, D)   # 階段 A：與 n 無關，可快取
prob = fill_kpoint_sdp(n, tpl)          # 階段 B：代入 n，填係數
solve_kpoint_sdp(prob)                  # 階段 C：壓縮、送 SDPA-GMP、Arb 驗證
```

## 引用

若使用本程式碼，請引用：

1. **de Laat et al.** — 原始 k-point bound 方法（arXiv:1812.06045）
2. **本碩士論文** — 可行狀態空間建模與 SRG 應用（請在論文中附上本儲存庫連結）
