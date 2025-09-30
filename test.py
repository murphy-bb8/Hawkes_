# 查看两个文件分别多少事件
import json,sys
for f in ["data_sim.json","events.json"]:
    try:
        with open(f,"r",encoding="utf-8") as fh:
            ev=json.load(fh)
        print(f, "events:", len(ev))
    except Exception as e:
        print(f, "=>", e)