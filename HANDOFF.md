# Quantum 比赛交接

更新：2026-10-29。工作目录 `D:\耿世荣\Documents\Quantum`，Windows + PowerShell。旧版交接备份在 `runs/handoff_before_20261028.md`（含 10-27 的公开复验、invsqrt 负结果、环境细节，仍可查）。本次将初始化本地 Git repository；大体积电路包、缓存和凭据由 `.gitignore` 排除，保留可复现实验代码、关键 JSON 结果和源码快照。

## 1. 现状

| 项目 | 数值 |
|---|---|
| 账号最佳 Public Score | **22.37455**，submission 57732578（10-28 09:43 UTC） |
| 对应包 | `external/new_public/somejoint_2244/submission.zip`，sha256 `5b2b5023…`，**全部是公开电路**（SomeJoint 22.44 notebook 输出，Apache-2.0，已署名） |
| 金线（第 12 名） | 23.15（10-28 09:40 UTC），第 1 名 26.14；两天前金线是 18.8，每天大约涨 1–2 分 |
| 今日额度 | 10-28 UTC 已用 1 次，剩 4 次；每天 00:00 UTC 重置 |
| 提交记录 | `runs/job_state.json`（`submissions_20261028`） |

公开前沿（SomeJoint 22.44 包，官方 verifier 默认 shots 全 PASS）：

| 任务 | 成本 | 倍数 | 来源 |
|---|---:|---:|---|
| invsqrt | 102,664（63Q，C=1629.6） | 20.23 | busyaprime lookup 重综合（172 AND）+ mrmeows 63Q |
| dqi | 4.223e11（897Q，T 440,633） | 57.87 | GPT-6 Astra Pro / SomeJoint fused Schur decoder；隐藏 salt 下约 k=2 次失败 |
| dfthcsos | 3.477e8（λ 48.27，T 16,752，Q 430） | 9.65 | SomeJoint R8/B143/C62，8-bit，15 个 five-word ANGLE 电路 |

线上 22.37 比本地 22.44 低，原因是 DQI 在隐藏 salt 下的失败次数不同。要到 23.15，三项乘积需要再涨约 10%。

用户要求：以自主改进为主，开源只作参考，不能默认开源结论正确；公开组合可以提交，但要注明来源。

## 2. 本轮自主实验（chemistry，全部在 `runs/chem_v2/r7/`）

结论：**瓶颈是 8-bit 量化器，不是形状。**

| 实验 | 结果 |
|---|---|
| SomeJoint R8 格点 → 连续松弛 | 0.0195 → 0.01345（`r8_b143_c62_cont.npz`）：量化大约贡献 14 mHa |
| 我们的 λ polish（`polish_weights.py`） | SomeJoint 权重已经卡在 e2=0.02 / E_gap 下限，没有余量 |
| R8→R7 删一组（`drop_group.py`）+ Adam | C62 停在约 0.020 后崩溃（Cholesky）；C70 停在 0.0175 |
| R7/C70 L-BFGS（`cont_lbfgs.py`） | 0.0175 → 0.01661（120 iter），仍在稳定下降；交接时进程 49480 还在跑（300 iter），日志 `r7_c70_lb.log`，结果写到 `r7_c70_lb.npz` |
| R8 B143→B129 剪向量 | 0.0180；每组每删 2 个向量约 +1 mHa |
| aquant（最不敏感优先，三次方调度，400 步/阶段） | 0.0134 → 87% 冻结时 0.0168，尾部爆到 0.04 以上 |
| aquant `--direction loss`（按梯度+曲率选 floor/ceil） | 更差：stage 4 冻结后 0.053，方向选择会让误差同向叠加 |
| aquant `--order most`，均匀 64 阶段 × 400 步 | 冻结 12% 时已经 0.018，没有跑完 |
| AdaRound 联合学习舍入（`adaround.py`） | 0.100，不可用 |

成本面（`shape_scan.py`，用 SomeJoint 实测标定 loader 速率；数值是 T·Q 相对 R8/B143/C62）：R7/B143/C62 0.896、R7/B129/C70 0.893、R7/B143/C70 0.936、R8/B129/C62 0.951、R8/B135/C62 0.972。前提是这些形状能量化到 ≤0.0199，**目前做不到**。

## 3. 量化器怎么修（下一步重点）

参考 Rustad 公开源码（`external/new_public/rustad_r10_src/src/pq.py`、`chain_c44.sh`，Apache-2.0，只作参考）。他从连续 13.31 mHa 量化到 19.98 mHa（R10/B129/C44）。与我们的关键差异：

1. **影响最大的角度先冻结**（topk 最大），在补偿自由度最多时处理最难的角度；
2. 冻结比例极细：0.05%、0.1%… 前 5% 分 9 段，然后到 30%，最后粗段到 100%；
3. **算力是我们的 10 倍以上**：前 5% 每阶段 12000 Adam 步，5–30% 每阶段 9000 步，之后 2000 步；lr_th 2e-4、lr_w 5e-5，段内 lr 衰减到 0.2；GN 影响估计 8 probes；fp32；D(w) 惩罚 6e-5；最后单独优化 w 4000 步；
4. 量化后再做 finish（一体项）→ gauge → lamopt（λ 在 e2 cap 0.01998 下下降）→ finish → gauge；
5. 删组用 L2 惩罚的 homotopy 逐步收缩（`cont_lbfgs.py --shrink-group`），删 copy 用 ctrunc，连续拟合用 L-BFGS。

建议步骤：

1. 给 `experiments/chemistry_v2/aquant.py` 加 `--fracs`（显式累计冻结比例）、`--split`（分段步数）、段内 lr 衰减和 fp32 前向，默认用 `--order most --direction nearest`；
2. 先在 `r8_b143_c62_cont.npz`（0.01345）上复现 ≤0.0199 作为验收，预计约 10 小时，必须后台运行并定期写 resume；
3. 能复现后，用 homotopy 删组 + L-BFGS 把 R7（或 R8/B135）的连续误差压到约 13 mHa，再量化；
4. 量化后：`polish_weights.py`（可行回溯版）降 λ → 编译 ANGLE（SomeJoint 的 phased five-word 或 mrmeows generic_lookup J5）→ 在 Kaggle 跑官方全量验证 → 和公开 invsqrt/dqi 组合提交。

预期收益：一个 −5% 到 −10% 的 chemistry 形状，大约等于总分 +1.7% 到 +3.5%，单靠它不够到金线。

## 3.1 自主改进思路（不依赖开源路线）

按“瓶颈 → 可检验假设 → 最小实验”排列。每条都先用小实验验证再投入算力。

1. **抗量化的连续解（sharpness-aware）**：现象是 SomeJoint 格点松弛后最近舍入直接变成 0.117，说明连续极小点非常尖锐。在连续训练里加入期望舍入损失 `Σ_i h_ii·Δ²/12`（Δ=π/128，h 用 GN 对角或 Hutchinson 估计），或者每步加均匀舍入噪声训练，让解落在平坦区域。验证：同一形状下，最近舍入误差和量化后最终误差是否显著下降。这直接针对瓶颈，开源方案里没有。
2. **每向量 gauge / 树的选择以降低噪声**：投影 uu^T 对 u→−u、分量置换不变。可以为每个向量选择符号和 Givens 树根，让大幅值分量放在噪声放大最小的位置，或者用全局共享树但做幅值排序。验证：固定 U，比较各种 gauge 下的最近舍入误差。
3. **面向 loader 的量化（table-aware）**：loader 占 T 约 52%，它的成本取决于词表的 ANF 结构。在量化时加入“让低位 bit 列更规则（常量或少数 monomial）”的偏好，在误差预算内换取更少的 AND。验证：固定误差，比较编译出的 Toffoli 数。
4. **精确离散收尾**：先用渐进量化做到 95%，剩余的尾部角度用 `dsearch.py` 的闭式 ΔE 加 rank-3 更新，做 floor/ceil 的精确贪心或 beam 搜索，替代 Adam 来处理尾部。
5. **λ 与误差联合优化**：连续阶段就把 λ_eff（D(w) 项）放进目标，避免量化后再用 polish 挤余量；e2 有剩余时优先换成 λ 或更小的形状。
6. **invsqrt 新结构**：直接把相位 bit 表示成 x 的低次多项式 / parity 特征，用 LP 求解（已有 `phase_refit.py`、`dual_witness` 基础），寻找比“lookup + 乘法”更省的特征族；每省 1 qubit ≈ +0.5% 总分。

## 4. 其他方向（未开始）

- invsqrt：主 lookup（8-bit 地址 → 35 bit，172 AND，40 线池）只占约 13% 的 Toffoli；重综合上限约 +0.3% 总分。63Q 电路可以在 Windows 上用 kickmix 读取（`scratchpad/inspect_inv.py` 的方法：`Circuit.from_file(BytesIO, format="kmb")`）。mrmeows 63Q 源码在其数据集 `source/invsqrt_63q/`，本地还没下载。
- dqi：1 qubit ≈ 0.22%，T 才是大头；要改 decoder，验证成本高。
- 公开 notebook 列表在 `kernels.json`，下载的公开包在 `external/new_public/`。公开前沿每天都在变，提交前先刷新。

## 5. 代码与环境要点

- chemistry 代码：`experiments/chemistry_v2/`。本轮新增 `drop_group.py`、`cont_lbfgs.py`、`adaround.py`、`shape_scan.py`；`aquant.py` 改为显式 `qval` 格点值，并支持 `--direction`；`gtrain.refit_block` 的 Cholesky ridge 从 1e-12 改成 1e-6（防止近共线向量崩溃）。
- 运行 torch 前设 `KMP_DUPLICATE_LIB_OK=TRUE`；kickmix 打不开中文绝对路径，要 chdir 后用相对路径。
- 长任务用 `Start-Process -WindowStyle Hidden` 后台运行，stdout/stderr 写到 `runs/...log/.err`；会话结束可能杀进程，长任务必须有 resume。
- Windows 不能运行官方 Linux `kickmix.so`。正式验证在 Kaggle 私有 notebook 上做（`scripts/prepare_public_refresh.py` 生成；`scripts/verify_candidate.py` 跑全默认 shots）。
- 提交：`kaggle competitions submit -c google-verifiable-quantum-circuit-compilation -f <zip> -m "<描述，注明来源>"`。不要在命令里设 `$ErrorActionPreference='Stop'`（CLI 的 stderr 警告会中断提交）。
- 隐藏 salt：电路字节的 sha256 混入秘密 salt 决定 shot 种子，DQI 的失败次数会随之变化。不要靠改字节“刷种子”。

## 6. 给下一会话的启动提示

> 先读 HANDOFF.md。账号最佳 22.37455（公开 SomeJoint 包），金线约 23.15 且在上涨。自主主线是修复 chemistry 8-bit 量化器：按 Rustad pq.py 的思路（影响最大先冻结、极细冻结比例、每阶段数千到上万步、fp32）扩展 aquant.py，先在 runs/chem_v2/r7/r8_b143_c62_cont.npz 上复现 ≤0.0199，再用于 R7 等更便宜的形状。提交公开组合要注明来源；用户要求以自主改进为主。

## 7. 2026-10-28 晚间续接：任务三量化实验

### 已完成的代码改动

- `experiments/chemistry_v2/aquant.py` 新增 L-BFGS compensation：`--lbfgs`、`--lbfgs-inner`。
- 新增可恢复的 warmup：`--warmup-lbfgs`、`--warmup-inner`，恢复 checkpoint 后可先在当前冻结状态继续优化。
- 新增 `--seed`，固定 Hutchinson probe 的随机性。
- 新增 `--hessian`，用于 exact Hessian Hutchinson 排名实验；实测不如原有 rank-1 sensitivity，默认不要开启。
- 修正 Adam/L-BFGS 的最佳状态保存逻辑：按真实 `error_2` 保存，不按带 `D(w)` 惩罚的目标保存，避免正则项掩盖物理误差。
- `--probes 0` 现在仍计算一阶 gradient impact，不再退化成只按角度距离排序。

### 关键实验结论

1. `order=most`（先冻结高影响角度）在当前 Givens 树参数化下不稳定。不同随机 probe 可能在第 4 段后从约 0.016 跳到 0.026 甚至 0.04。
2. `order=least`（先冻结低影响角度）明显稳定：R8/B143/C62 从连续 `error_2=0.013445` 出发，冻结到 60% 时仍约 `0.0149`；继续密集尾段到 92% 时约 `0.0225`。
3. 原始尾段 `0.6,0.7,0.8,0.9,1.0` 太粗，最终全冻结后约 `0.086~0.11`。改为每 2% 一段后，92% 前明显改善，但 92%→100% 仍跳到约 `0.086`。
4. `experiments/chemistry_v2/dsearch.py` 已结束，输入 `runs/chem_v2/r7/r8_tail.npz` 的 rank-3 精确贪心收尾结果为 `runs/chem_v2/r7/r8_tail_dsearch.npz`，`error_2=0.0358119`，仍未过线。

### 当前文件与状态

- 连续起点：`runs/chem_v2/r7/r8_b143_c62_cont.npz`，R8/B143/C62，连续 `error_2≈0.013445`。
- 低影响 60% checkpoint：`runs/chem_v2/r7/r8_base60.resume.npz`。
- 全量密集尾段结果：`runs/chem_v2/r7/r8_tail.npz`，当前离散 `error_2≈0.085978`（尚未通过 0.0199 阈值）。
- 低影响分段日志：`runs/chem_v2/r7/r8_least_all.log`、`runs/chem_v2/r7/r8_tail.log`。
- 本轮 dsearch 已结束；不要删除 `.resume.npz`，也不要在没有新假设前启动另一项大规模 CUDA 优化。

### 当前疑问

- 为什么低影响优先在 92% 前稳定，而最后 8% 误差突然放大？可能是剩余角度的局部曲率高度相关，单角度 greedy 补偿不足，也可能需要联合改变多个相邻 Givens angles。
- `dsearch.py` 的 rank-3 更新已验证只能把 `error_2` 从约 `0.08598` 降到 `0.0358119`，没有拉回 `0.0199`；当前形状应停止继续投入。
- 现有 rank-1 sensitivity 的随机 probe 对冻结顺序影响很大；需要判断是否应改成确定性全量梯度、分组平均或按 Givens tree depth 分层。
- 形状 R8/B143/C62 的连续解很低，但未必存在足够平坦的 8-bit 邻域；需要和 R7/B143/C70 的连续解约 0.0159 做同样的低影响量化比较。
- 任务三的最终得分还受 `lambda_eff`、`error_1`、loader cost 影响；仅看 `error_2` 不能直接推出总分超过 25。

### 主要难题

- 最后 8% 角度冻结时自由补偿维度急剧减少，近邻 rounding 误差具有强相关性。
- `D(w)` 正则项约为 0.025，和 `error_2` 同量级；若按惩罚目标保存 checkpoint，会得到物理误差更差的状态。
- 长实验一次需要数分钟到十几分钟；必须每 4 段保存 `.resume.npz`，否则尾段失败后需要重跑前半段。
- 本地 Windows 不能运行官方 Linux `kickmix.so`；正式 `error_1`、`lambda_eff`、Toffoli/Q 和总分仍需用官方 Kaggle verifier。

### 后续优先级

1. 先读取 dsearch 输出；若 `error_2<=0.0199`，立即用 `finalize.py` 做 `w/W` polish、one-body words，再跑 `eval_npz.py` 和官方全量验证。
2. 若 dsearch 无法过线，从 60% checkpoint 重新跑尾段：
   - 先用 0.90→0.92→0.94→…→1.0 的 1% 或 0.5% 小段；
   - 对最后阶段尝试 `order=most`，但只在已经冻结 90% 以上的 checkpoint 上切换；
   - 每段增加 `--lbfgs` 或使用 `--warmup-lbfgs`，并按真实 `error_2` 保存。
3. 若 R8 仍失败，复制同一量化器到 `runs/chem_v2/r7/r7_b143_c70_cont.npz`，优先测试 `order=least` 的 60% 和 92% checkpoint；其成本更低，可能提高总分。
4. 尾段若仍不可行，尝试小范围联合离散搜索：对相邻 tree edges 做 2-angle/3-angle beam，而不是逐角 `+-1`；可复用 `dsearch.py` 的 rank-3 更新并限制候选数量。
5. 只有在任务三出现可验证收益后，再刷新公开 Task 1/2 组合并做一次完整提交；当前公开组合最佳记录仍为 22.37455，目标 25+ 尚未达到。

### 交接启动命令

```powershell
cd D:\耿世荣\Documents\Quantum
$env:KMP_DUPLICATE_LIB_OK='TRUE'
Get-Process python -ErrorAction SilentlyContinue
Get-ChildItem runs\chem_v2\r7\r8_tail_dsearch* -ErrorAction SilentlyContinue
```

随后先读取 `runs/profile_20261029.json` 和第 8 节的 profiling 结论；不要删除 `.resume.npz`。所有最终候选必须经过本地 `eval_npz.py`、`finalize.py` 和官方 Kaggle verifier；提交描述注明公开电路来源与本轮自主量化改动。

## 8. 2026-10-29 执行 profiling 后的状态

### 已执行

- 检查进程后确认没有重复的 `dsearch`：`48912` 只是 venv 的 `python.exe` launcher，真正任务已由 `18496` 运行并随后结束；目前没有残留的 Python 长任务。
- 从 SomeJoint 22.44 notebook 提取了 DQI 的 68 个公开源文件到 `runs/dqi_profile/source`，没有在 Windows 上强行重建官方 Linux `kickmix.so`。
- 生成完整结果：`runs/profile_20261029.json`。
- 完成 chemistry gauge scan：`experiments/chemistry_v2/gauge_scan.py`，结果在 `runs/chem_v2/r7/gauge_scan.json`。

### DQI profiling 结果

当前公开电路仍是 `Q=897`、平均 `T=440632.998`、最大 `T=454477`，公开默认 seed 的错误率为 0。按官方 build 记录拆分：

| 阶段 | T | 占比 | 峰值 Q |
|---|---:|---:|---:|
| `extract` | 65,881 | 14.95% | 897 |
| `synthesis` | 33,317 | 7.56% | 889 |
| `convert` | 68,352 | 15.51% | 897 |
| `scan` | 44,275 | 10.05% | 870 |
| `unconvert` | 69,455 | 15.76% | 897 |
| `unsynthesis` | 41,931 | 9.52% | 897 |
| `unextract` | 117,422 | 26.65% | 897 |

结论：没有证据表明一个普通 GF multiplier 替换就能带来大收益。最大单阶段是 `unextract`，`convert+unconvert` 合计 31.27%，`extract+unextract` 合计 41.60%；四个阶段都碰到 `Q=897`，因此只优化较低峰值阶段不会降低总 Q。下一步若继续 DQI，应对 `unextract` 和 convert 往返做源级调用计数，重点寻找少做一次乘法/逆运算或降低共同峰值的方法；先不要盲目重写 multiplier。

### invsqrt profiling 结果

- `Q=63`，总 Toffoli-equivalent `1629.591`。
- 算术/lookup 部分约占 81.89%，rotation synthesis 约占 18.09%；公开 lookup attribution 为 172 AND。
- 最坏误差已经达到容差的 `0.999324`，几乎没有降低精度的空间。

结论：如果没有 Linux/Kickmix 重综合和明确的 Q/算术结构改动，invsqrt 暂不投入；单纯减少精度会越过误差线。

### chemistry profiling 结果

- 连续起点 `R8/B143/C62` 的最近 8-bit rounding 误差为 `0.1191874`。
- 16 个随机 sign gauge 与源树结果完全相同；8 个随机合法 Givens tree 最好为 `0.1363106`，均差于源树。
- 已结束的 `dsearch` 将 `error_2` 从约 `0.08598` 降到 `0.0358119`，仍未达到 `0.0199` 阈值。

结论：简单 gauge 没有收益，随机树更差；不要仅凭这轮 scan 再启动 overnight 量化。当前形状应暂停，除非提出联合多角离散搜索或新的抗量化目标。

### 当前疑问与后续方向

1. DQI 的主要未知量是 `unextract` 内部的具体 primitive 调用数，以及 convert/unconvert 是否存在可合并的逆运算；需要做静态 call-count 或一次可运行的 Linux/Kaggle profile。
2. DQI 的失败 shot 尚未能在 Windows 本地复现隐藏 salt；若没有可重复的 syndrome 模式，不能把公开记录中的 `R` 直接当成可修复 bug。
3. invsqrt 只有在拿到可运行的 Kickmix/Linux 环境后，才值得按 region 统计 lookup、rotation 和 arithmetic 的真实可重综合空间。
4. chemistry 需要改变方法：优先尝试相邻 2/3 个 Givens angle 的小 beam search，或在连续训练中加入 rounding-noise/curvature penalty；没有新假设前不再扩大当前 dsearch。
5. 每个真正降低成本或误差的候选，先跑本地 `eval_npz.py`/`finalize.py`，再做一次官方 Kaggle full verification；目前没有新候选可提交。

### DQI 源码级实例计数补充

`build_897.py` 的入口实际使用 `final_decoder/decode.py`，不是旧的 `decode.py` 中“289 个 confluent jet”的 baseline 路径。对实际入口静态展开后：

- 17 个 external groups：8 个 ordinary、9 个 homogeneous。
- 每个方向有 56 个 ordinary `jet_forward`/`jet_reverse` 调用，正反合计 112 个；这是 `sum(2*gi for gi=0..7)` 的精确结果。
- 每个方向有 17 次 `inverse24_forward190` 或 `inverse24_reverse209`。
- `scan` 是 4025 个点 × 11 Toffoli，正好 44,275 T。

源码注释中的 658 T 是 jet kernel 的局部标称值，官方 stage 总数还包含外层 arithmetic、zero flags、scaling 和 uncomputation，因此不能直接用 `56×658` 代替实测阶段成本。后续 DQI 优化应以 `final_decoder` 的调用路径为准。

## 9. 2026-10-29 联合离散搜索补充

在 `r8_tail_dsearch.npz` 的基础上新增 `experiments/chemistry_v2/pairsearch.py`，对同一向量上共享树节点的相邻两个 Givens angle 同时尝试 `±1`，以捕捉单角 greedy 无法发现的联合下降。

- 第 1 轮：`0.035811927 → 0.035462033`，耗时约 104 秒。
- 第 2 轮：从第 1 轮结果继续，`0.035462033 → 0.035192646`，耗时约 105 秒。
- 两轮共接受约 2,276 个 moves，但仍明显高于 `0.0199` 阈值。

结论：相邻两角联合搜索有效但收益很小，不能解决当前形状的尾段崩溃；不要继续扩大该 beam 的规模。若重新投入 chemistry，应改为连续阶段的 rounding-noise/curvature penalty 或更大范围的联合重优化。

## 10. 2026-10-29 数学方向补测

### 内部 gauge 的数学结论

Givens tree 每条边可以通过中间列符号得到多个等价 angle representation。已实现两种搜索：

- `gauge_dp.py`：用树动态规划选择最小 angle rounding 误差的等价表示；结果仍为 `error_2=0.1191874`。
- `gauge_sample.py`：每个向量随机采样 32 个内部 gauge，并按 projector rounding error 选最优；结果仍为 `0.1191874`。

这说明当前 nearest rounding 对这些 exact projector symmetries 基本保持不变，单靠符号 gauge 不能解决量化崩溃。

### 权重与角度交替优化

- 相邻两角搜索两轮后：`0.0358119 → 0.0351926`。
- 固定离散角度，对全部 `w` 做 joint `LBFGS`：降到 `0.0342545`。
- 再做两轮角度 `dsearch`：降到约 `0.033584`。

这验证了 `w` 的全局优化比单次 block refit 更好，但距离 `0.0199` 仍有明显差距；当前形状不具备提交价值。

### DQI 数学检查

源码中的乘法 rank 已经是：GF(2^12) rank 42，GF(2^24) 默认 rank 105，jet 使用 rank 99。由此排除了把 schoolbook multiplier 换成 Karatsuba 作为主突破口的假设。

### 现在的研究重点

如果继续数学路线，应转向量化前连续优化的离散鲁棒目标：直接最小化 floor/ceil 组合的期望残差，或建立 angle rounding 的二阶 binary quadratic model 后做联合 bit search。继续扩大当前相邻两角 beam 的收益预计很低。

## 11. 2026-10-29 提交批次

已通过 Kaggle CLI（经本机 Clash HTTP proxy）提交四个候选，当前均为 `PENDING`：

| ref | 候选 | 本地分/已知分 | 预计公开榜区间 |
|---:|---|---:|---:|
| 57759596 | `somejoint_2244` 当前最佳组合 | 本地 22.437775；同字节历史公开 22.37455 | 约 573 |
| 57759605 | `busyaprime_2234` | 本地官方记录 22.402947 | 约 560 |
| 57759612 | `dungluong_2234b` | 本地记录约 22.40294 | 约 560 |
| 57759622 | `zongaobian_signed_digits` | 仅有预验证 archive，等待榜单 | 待定 |

本批次使用 4 次额度。上次完成提交为 ref `57732578`，公开分 `22.37455`、排名 `573`。最新榜单快照为 `runs/kaggle_leaderboard.json`，当前榜首 `26.14113`。

## 12. 2026-10-29 公开讨论复盘与下一轮方向

### 公开前沿已经变化

- 公开榜首快照为 `26.14113`；10 月 28 日的讨论快照记录 25 分以上队伍已出现，23 分以上队伍约 25 个，22.5 分以上约 370 个。当前账号最好公开分仍为 `22.37455`，本地官方几何均值为 `22.43777534520302`，与榜首差距不是一次随机 salt 能解决的。
- 公开最好组合约为 `22.70031`：invsqrt 63Q、DQI 889Q、chemistry R8/B143/C60、8-bit full-XOR five-word loader。对应讨论：`753588`（RezOps Briefing）和 `753633`（`lambda_eff` 下界）。

### 讨论中可直接采用的结论

1. **Chemistry 的主矛盾是 loader/T/Q，不是继续抠 `lambda_eff`。** 公开分析给出的通用下界约为 `lambda_eff≈46.8`，当前公开值约 48.7；即使完全消除剩余 slack，总分上限也只有约 +1.3%。真正有效的公开改动是 R8/B143/C60、full-XOR 五词表、每个低乘积只计算一次，以及十个 ANGLE 电路的重编译。我们的 gauge、GPTQ 式舍入和相邻两角搜索均未达到 `error_2≤0.0199`，不能作为提交候选。
2. **DQI 的 GF multiplier 已接近公开 rank 下界。** 公开负结果指出 GF(2^12) rank 42 已被充分利用；直接换 Karatsuba 或学校算法重写不是主线。公开 897Q→889Q 的收益来自释放/重算 scratch 的调度，约少 8Q、只多约 30T；按 `Q²T` 计这比少几百 Toffoli 更值。下一步应比较 889Q 源级调度，而不是重写 multiplier。
3. **DQI 的失败次数是电路 hash 加 salt 的固定抽样。** 同一电路会重复得到同样的隐藏盐结果，不能把本地 0 次失败当作公开榜保证。只有发现可重复的 syndrome/phase 分支错误，才值得修逻辑；否则应优先降低共同峰值 Q。
4. **invsqrt 已经接近精度饱和。** 公开 tracer 的最坏误差约为容差的 0.9993，降 rotation 精度会直接失败；63Q 的 scratch 生命周期交错，简单借线也没有收益。除非能做 Linux/Kickmix 级重排，否则暂时冻结。
5. **奖项复现要求 generator。** 公开讨论 `753588` 引用主办方说明，最终需要能生成精确电路；只保存二进制 `kmb/kmx` 不足以作为长期可复现方案。后续采用公开 Apache-2.0 组件时，必须保留来源、hash、许可证和本地修改说明。

### 当前未解决的问题

- `57759596`、`57759605`、`57759612`、`57759622` 四个提交的官方结果仍需后续查询；提交批次记录在 `runs/submission_day_20261029.json`。
- 我们还没有把公开 889Q DQI 和 R8/B143/C60 chemistry 的 generator/source 完整接入本地验证，当前 22.44 包仍是 897Q + C62 组合。
- `experiments/chemistry_v2/global_beam.py` 的全局 beam 搜索曾因候选张量占用过大而中止，尚无输出候选；不要把它当作已验证改进。
- 当前 chemistry 的连续误差很低，但 8-bit 尾段误差仍从约 0.02 爆到 0.03–0.12；单角或相邻两角 greedy 已证明不足。

### 下一轮执行顺序

1. 先拉取并核对公开 889Q DQI、R8/B143/C60 和 full-XOR ANGLE 的文件 hash、来源与许可证；在本地只做静态 cost/profile，不先启动长优化。
2. 对 DQI `final_decoder` 做源级 schedule diff，找能降低共同峰值 Q 的释放/重算窗口；保留 zero flag 和 phase correction，不尝试已知失败的 multiplier 替换。
3. 对 chemistry 复现 C60 full-XOR loader，比较每个地址的低乘积复用和 ANGLE 编译 T；只有通过 `error_2`、`E_gap`、`lambda_eff` 和成本检查后才组装 submission。
4. 每个真正改进先跑本地 `eval_npz.py`/`finalize.py` 和官方 full verification，再使用剩余额度提交；不因 public salt 的偶然涨分重复提交同一字节。

### 仓库恢复提示

```powershell
cd D:\耿世荣\Documents\Quantum
git log --oneline --decorate -5
Get-Content HANDOFF.md
Get-Content REPO_MANIFEST.json
```

仓库只保存小型源码、实验脚本、关键 JSON、DQI source snapshot 和少量可复核样本；原始 competition 数据、`*.kmb`、submission zip、缓存和 Python 环境按 `REPO_MANIFEST.json` 记录但不入 Git。提交前先检查 `git status --short`，不要把 `kaggle.json`、`.aws` 或任何 token 加入仓库。
