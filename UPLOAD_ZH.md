# 单个 ZIP 上传说明

直接将 `retail-recalibration-core.zip` 拖入 GitHub 上传页面并提交即可。上传的是一个压缩包，不需要拆成多个分卷，也不需要先解压后逐个上传。

审稿人下载 ZIP，解压后进入 `retail-recalibration` 文件夹，执行：

```text
python -m pip install -r requirements.txt
python reproduce.py
```

本包保留核心已选策略评分、E1/E2/E3、意大利主要确认及反馈比较的复算，并生成全部 15 张图。删减的是完整参数候选与部分非核心诊断的逐日数组；它们的结果汇总仍保留。完整重训模型库不在本包中，少量代表性权重的范围见 `models/README.md`。本包不是所有基础模型和全部逐商品预测的完整重建包。

代码与数据范围见 README.md 和 SCOPE.json。稿件、作者信息、历史稿件版本均不在此包内。若仓库已有旧包，使用本次命名并在仓库 README 中指向本次文件，避免让审稿人误用旧包。
