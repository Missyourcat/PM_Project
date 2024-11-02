# @Time: 2024/11/2 09:58
# @Author: Shen Hao
# @File: liner_aqi.py
# @system: Win10
import pyecharts.options as opts
from pyecharts.charts import Line
def liner_aqi(Year_Data, AQI_Data):
    all_data = list(zip(Year_Data, AQI_Data))
    (
        Line()
        .add_xaxis(xaxis_data=[item[0] for item in all_data])
        .add_yaxis(
            series_name="",
            y_axis=[item[1] for item in all_data],
            yaxis_index=0,
            is_smooth=True,
            is_symbol_show=False,
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="商丘 AQI"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            datazoom_opts=[
                opts.DataZoomOpts(yaxis_index=0),
                opts.DataZoomOpts(type_="inside", yaxis_index=0),
            ],
            visualmap_opts=opts.VisualMapOpts(
                pos_top="10",
                pos_right="10",
                is_piecewise=True,
                pieces=[
                    {"gt": 0, "lte": 50, "color": "#096"},
                    {"gt": 50, "lte": 100, "color": "#ffde33"},
                    {"gt": 100, "lte": 150, "color": "#ff9933"},
                    {"gt": 150, "lte": 200, "color": "#cc0033"},
                    {"gt": 200, "lte": 300, "color": "#660099"},
                    {"gt": 300, "color": "#7e0023"},
                ],
                out_of_range={"color": "#999"},
            ),
            xaxis_opts=opts.AxisOpts(type_="category"),
            yaxis_opts=opts.AxisOpts(
                type_="value",
                name_location="start",
                min_=0,
                max_=500,
                is_scale=True,
                axistick_opts=opts.AxisTickOpts(is_inside=False),
            ),
        )
        .set_series_opts(
            markline_opts=opts.MarkLineOpts(
                data=[
                    {"yAxis": 50},
                    {"yAxis": 100},
                    {"yAxis": 150},
                    {"yAxis": 200},
                    {"yAxis": 300},
                ],
                label_opts=opts.LabelOpts(position="end"),
            )
        )
        .render("../Echarts/ShangQiu_AQI.html")
    )