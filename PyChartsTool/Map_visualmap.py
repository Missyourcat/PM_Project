# @Time: 2024/11/2 10:11
# @Author: Shen Hao
# @File: Map_visualmap.py
# @system: Win10
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.faker import Faker

c = (
    Map()
    .add("AQI", [list(z) for z in zip(Faker.provinces, Faker.values())], "china")
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Map-VisualMap（分段型）"),
        visualmap_opts=opts.VisualMapOpts(max_=200, is_piecewise=True),
    )
    .render("../Echarts/map_visualmap_piecewise.html")
)
